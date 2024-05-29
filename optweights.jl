

"""
    getvertices(bbox::Vector{Int},h::T,test::Function) where {T}
Returns list of integer vectors `xv` where `test(xv) > 0` where `xv` 
is inside the bounding box `bbox`: `bbox[1] <= xv[1] <= bbox[2]`,
`bbox[3] <= xv[2] <= bbox[4]`
"""
function getvertices(bbox::Vector{Int},h::T,test::Function) where {T}
    [[i,j] for i in bbox[1]:bbox[2] for j in bbox[3]:bbox[4] if test(h*[i,j]) > 0]
end


function laplacian(vertices::Vector{Vector{Int}}, h::T) where {T}
    indexes = Dict{eltype(vertices),Int}()
    for i in 1:length(vertices)
        indexes[vertices[i]] = i
    end
    # create Laplacian operator on D with Neumann boundary conditions
    ilist = Int[]
    jlist = Int[]
    vlist = Float64[]
    n = length(vertices)
    for i = 1:n
        x,y = vertices[i]
        nnbrs = 0
        newpt = [x+1,y]
        if newpt in keys(indexes)
            push!(ilist,i)
            push!(jlist,indexes[newpt])
            push!(vlist,-1/h^2)
            nnbrs+=1
        end
        newpt = [x-1,y]
        if newpt in keys(indexes)
            push!(ilist,i)
            push!(jlist,indexes[newpt])
            push!(vlist,-1/h^2)
            nnbrs+=1
        end
        newpt = [x,y-1]
        if newpt in keys(indexes)
            push!(ilist,i)
            push!(jlist,indexes[newpt])
            push!(vlist,-1/h^2)
            nnbrs+=1
        end
        newpt = [x,y+1]
        if newpt in keys(indexes)
            push!(ilist,i)
            push!(jlist,indexes[newpt])
            push!(vlist,-1/h^2)
            nnbrs+=1
        end
        push!(ilist,i)
        push!(jlist,i)
        push!(vlist,nnbrs/h^2)
    end
    sparse(ilist,jlist,vlist), indexes
end


function minmax_ridgeidxs(ws::Matrix{T},vertices::Vector{Vector{Int}}, h::T, hridge::T) where {T}
    # Find min and max indices for shifts of the B-spline functions over the discretized domain D.
    minvals = [minimum(dot(ws[:,i],h*xv) for xv in vertices) for i in 1:size(ws,2)]
    maxvals = [maximum(dot(ws[:,i],h*xv) for xv in vertices) for i in 1:size(ws,2)]
    min_k = Int.(floor.(minvals/hridge)) .- 1
    max_k = Int.( ceil.(maxvals/hridge)) .+ 1
    min_k,max_k
end

"""
    function Cmatrix(w::Vector{T},h::T,hridge::T,vertices::Matrix{Int},min_k,max_k) where {T}
Returns the `C` matrix for weight vector `w` over domain given by points in `h*vertices` where `indexes[vertices[i]] == i`.
The values of `C` are given by the B-spline values at `h*dot(w,vx)/hridge+k` for integer `k`
"""
function Cmatrix(w::Vector{T},h::T,hridge::T,vertices::Vector{Vector{Int}},min_k,max_k) where {T}
    ilist = Int[]
    jlist = Int[]
    vlist = T[]
    for k = min_k:max_k
        i = k-min_k+1
        for j = 1:length(vertices)
            val = Bfn(h*dot(w,vertices[j])/hridge+k)
            if ! iszero(val)
                push!(ilist,i)
                push!(jlist,j)
                push!(vlist,val)
            end
        end
    end
    sparse(ilist,jlist,vlist,max_k-min_k+1,length(vertices))
end

function Cmatrix(ws::Matrix{T},h::T,hridge::T,vertices::Vector{Vector{Int}},min_k,max_k) where {T}
    Clist = [Cmatrix(ws[:,i],h,hridge,vertices,min_k[i],max_k[i]) for i in 1:size(ws,2)]
    rowstart = [1;cumsum(size(C,1) for C in Clist).+1]
    #=
    Clist = AbstractMatrix{T}[]
    rowstart = [1]
    for i = 1:size(ws,2)
        Ci = Cmatrix(ws[:,i],h,hridge,vertices,min_k[i],max_k[i])
        push!(Clist,Ci)
        push!(rowstart,rowstart[end]+size(Ci,1)-1)
    end
    =#
    vcat(Clist...),rowstart
end

function adjdCmatrix(w::Vector{T},h::T,hridge::T,vertices::Vector{Vector{Int}},min_k,max_k,Z::Matrix{T}) where {T}
    sum(Z[k,l]*dBfn(h*dot(w,vx)/hridge+k_min+k-1)*h/hridge*vx for k = 1:max_k-min_k+1, vx in vertices)
end

function tAinvmul(x::AbstractVector{T},Achol) where {T}
    [zero(T); Achol \ x[2:end]]
end

function tAinvmul(X::AbstractMatrix{T},Achol) where {T}
    [zeros(T,size(X,2))'; Achol \ X[2:end,:]]
end


function projnullC(C::AbstractMatrix{T},x::AbstractVector{T};η::T=zero(T),normalmx=nothing) where {T<:Real}
    if normalmx == nothing
        normalmx = cholesky(C*C'+η*I)
    end
    y = x - C'*(normalmx \ (C*x))
    y = y .- sum(y)/length(y)
end

function projnullCtr(C::AbstractMatrix{T},x::AbstractVector{T};η::T=zero(T),normalmx=nothing) where {T<:Real}
    if normalmx == nothing
        normalmx = cholesky(C*C'+η*I)
    end
    y = x .- sum(x)/length(x)
    y = y - C'*(normalmx \ (C*y))
end

function projnullC(C::AbstractMatrix{T},X::AbstractMatrix{T};η::T=zero(T),normalmx=nothing) where {T<:Real}
    if normalmx == nothing
        normalmx = cholesky(C*C'+η*I)
    end
    Y = X - C'*(normalmx \ (C*X))
    for j = 1:size(Y,2)
        offset = sum(Y[:,j])/size(Y,1)
        @. Y[:,j] = Y[:,j] - offset
    end
    Y
end

function projnullCtr(C::AbstractMatrix{T},X::AbstractMatrix{T};η::T=zero(T),normalmx=nothing) where {T<:Real}
    if normalmx == nothing
        normalmx = cholesky(C*C'+η*I)
    end
    Y = copy(X)  
    for j = 1:size(Y,2)
        offset = sum(Y[:,j])/size(Y,1)
        @. Y[:,j] = Y[:,j] - offset
    end
    Y = Y - C'*(normalmx \ (C*Y))
end

struct OptWtMatrix{T}
    solver::Function # solver from Laplacians.approxchol_sddm for A[2:end,2:end]
    normalmx # normal equations matrix, usually Cholesky factored for C.C'
    C::AbstractMatrix{T} # filter matrix
end

import Base: *, eltype, size
import LinearAlgebra: mul!, issymmetric

size(owm::OptWtMatrix{T},i::Int) where {T} = size(owm)[i]
size(owm::OptWtMatrix{T}) where {T} = (size(owm.C,2),size(owm.C,2))
eltype(owm::OptWtMatrix{T}) where {T} = T

function mul!(y::AbstractVector{T},owm::OptWtMatrix{T},x::AbstractVector{T},α::T,β::T) where {T} 
    Px = projnullC(owm.C,x,normalmx=owm.normalmx)
    y1 = [zero(T); owm.solver(Px[2:end])]
    @. y *= β
    y += α*projnullCtr(owm.C,y1,normalmx=owm.normalmx)
    y
end

function mul!(y::AbstractVector{T},owm::OptWtMatrix{T},x::AbstractVector{T}) where {T} 
    mul!(y,owm,x,one(T),zero(T))
end


function *(owm::OptWtMatrix{T},x::AbstractVector{T}) where {T} 
    Px = projnullC(owm.C,x,normalmx=owm.normalmx)
    y = [zero(T); owm.solver(Px[2:end])]
    projnullCtr(owm.C,y,normalmx=owm.normalmx)
end
    
function *(owm::OptWtMatrix{T},X::AbstractMatrix{T}) where {T} 
    reduce(hcat,owm*X[:,i] for i in 1:size(X,2)) # uses the previous version
end

function issymmetric(owm::OptWtMatrix{T}) where {T}
    true
end

function maptoD(v::AbstractVector{T},vertices::Vector{Vector{Int}}) where {T}
    minvxidxs = [minimum(vx[i] for vx in vertices) for i in 1:2]
    maxvxidxs = [maximum(vx[i] for vx in vertices) for i in 1:2]
    out = zeros(maxvxidxs[1]-minvxidxs[1]+1,maxvxidxs[2]-minvxidxs[2]+1)
    for i = 1:length(vertices)
        out[vertices[i][1]-minvxidxs[1]+1,vertices[i][2]-minvxidxs[2]+1] = v[i]
    end
    out
end

"""
    softlambdamax(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) -> T
Returns ``(1/\\alpha)\\ln(trace(e^{\\alpha B}))``.
"""
function softlambdamax(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) where {T}
    if evals == nothing
        evals,evecs = eigen(Symmetric(B))
    end
    gamma = maximum(evals)
    expevals = exp.(alpha*(evals.-gamma))
    gamma + log(sum(expevals))/alpha
end

"""
    dsoftlambdamax(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) -> D,V
Returns gradient of `softlambdamax(B,alpha)` with respect to `B` in factored form:
The actual gradient is ``VDV^T = e^{\\alpha B}/trace(e^{\\alpha B})``.
"""
function dsoftlambdamax(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) where {T}
    if evals == nothing
        evals,evecs = eigen(Symmetric(B))
    end
    gamma = maximum(evals)
    expevals = exp.(alpha*(evals.-gamma))
    D = Diagonal(expevals/sum(expevals))
    V = copy(evecs)
    (D,V)
end

"""
    softlambdamax2(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) -> T
Returns ``trace(B^m)^{1/m}``.
"""
function softlambdamax2(B::AbstractMatrix{T},m::Int) where {T}
    n = opnorm(B)
    n*trace((B/n)^m)^(1/m)
end


"""
    dsoftlambdamax2(B::AbstractMatrix{T},alpha::T;evals=nothing,evecs=nothing) -> T
Returns ``trace(B^m)^{1/m}``.
"""
function dsoftlambdamax2(B::AbstractMatrix{T},m::Int) where {T}
    n = opnorm(B)
    trace((B/n)^(m-1))^((m-1)/m)*(B/n)^(m-1)
end

"""
    softmaxL2approxerr(ws::Matrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T,alpha::T;η::T=zero(T)) where {T}
Computes and return the `softlambdamax` value for the weight vectors that are columns of `ws`.
"""
function softmaxL2approxerr(ws::Matrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T,alpha::T;η::T=zero(T)) where {T}
    A,indexes = laplacian(vertices,h)
    Achol = cholesky(A[2:end,2:end])
    min_k,max_k = minmax_ridgeidxs(ws,vertices, h, hridge)
    C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k)
    CCT = cholesky(C*C'+η*I)
    P = Matrix(projnullC(C,1.0I(size(A,1)),η=η,normalmx=CCT))
    return softlambdamax(P'*tAinvmul(P,Achol),alpha)
end

"""
    dsoftmaxL2approxerr(ws::Matrix{T},vertices::Vector{Vertor{Int}},h::T,hridge::T,alpha::T;η::T=zero(T)) where {T}
Computes and return the gradient of `softlambdamax` value for the weight vectors that are columns of `ws`.
"""
function dsoftmaxL2approxerr(ws::Matrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T,alpha::T;η::T=zero(T)) where {T}
    n = length(vertices)
    A,indexes = laplacian(vertices,h)
    @assert n == size(A,1) == size(A,2)
    Achol = cholesky(A[2:end,2:end])
    min_k,max_k = minmax_ridgeidxs(ws,vertices, h, hridge)
    C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k)
    CCT = cholesky(C*C'+η*I)
    P = Matrix(projnullC(C,1.0I(size(A,1)),η=η,normalmx=CCT))
    # return softlambdamax(P'*tAinvmul(P,Achol),alpha)
    M = P'*tAinvmul(P,Achol)
    D,V = dsoftlambdamax(M,alpha);
    Ch = cholesky(C*C'+η*I)
    G = -2*(Ch \ (((C*[0 zeros(1,n-1); zeros(n-1,1) inv(Matrix(A[2:end,2:end]))])*V)*D*(V'*P)))
    # istart = [0; cumsum(max_k-min_k.+1)]
    nrows = rowstart[2:end] - rowstart[1:end-1]
    # @show rowstart
    # @show size(G)
    gradws = reduce(hcat,sum(let vx = vertices[idx], u = h*dot(ws[:,i],vx)/hridge; 
        sum(G[rowstart[i]+k-min_k[i],idx]*dBfn(u+k) for k in min_k[i]:min_k[i]+nrows[i]-1)*(h/hridge)*vx  end 
            for idx in 1:length(vertices)) for i in 1:size(ws,2))
    gradws
end


function gradstepws!(ws::Matrix{T},dws::Matrix{T},s::T) where {T}
    for i = 1:size(ws,2)
        dws[:,i] = dws[:,i] - dot(dws[:,i],ws[:,i])/dot(ws[:,i],ws[:,i])*ws[:,i]
        ws[:,i] = ws[:,i] - s*dws[:,i]
    end
    for i = 1:size(ws,2)
        ws[:,i] = ws[:,i] / norm(ws[:,i])
    end
    ws
end

"""
    function makefuncws(ws::Matrix{T},A::AbstractMatrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T,alpha::T;η::T=zero(T),Achol=nothing) where {T}
Returns the hard-to-approximate function found for the given set of weight vectors in `ws` over the region defined by `vertices` and `h`.
"""
function makefuncws(ws::Matrix{T},A::AbstractMatrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T;η::T=zero(T),Achol=nothing) where {T}
    min_k,max_k = minmax_ridgeidxs(ws,vertices, h, hridge)
    C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k)
    CCT = cholesky(C*C'+η*I)
    P = Matrix(projnullC(C,1.0I(size(A,1)),η=η,normalmx=CCT));
    if Achol == nothing
        Achol = cholesky(A[2:end,2:end])
    end
    M = P'*tAinvmul(P,Achol)
    evals,evecs = eigen(Symmetric(M))
    maxeval,idx = findmax(evals)
    v = evecs[:,idx]
    scale = 1/(h^2*dot(v,v))
    v = scale*v;
    mu = maptoD(v,vertices);
    f = tAinvmul(v,Achol)
    scalef = 1/(h^2*dot(f,A*f))
    f = scalef*f
    f = f - sum(f)/length(f)
    funcn = maptoD(f,vertices)
    funcn,mu
end


function wstopeigvals(ws::Matrix{T},vertices::Vector{Vector{Int}},h::T,hridge::T,alpha::T;η::T=zero(T),num::Int=10) where {T}
    A,indexes = laplacian(vertices,h)
    Achol = cholesky(A[2:end,2:end])
    min_k,max_k = minmax_ridgeidxs(ws,vertices, h, hridge)
    C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k)
    CCT = cholesky(C*C'+η*I)
    P = Matrix(projnullC(C,1.0I(size(A,1)),η=η,normalmx=CCT))
    eigenvals,eigenvecs = eigen(P'*tAinvmul(P,Achol))
    eigenvals[1:num]
end