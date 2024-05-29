using LinearAlgebra, Plots, SparseArrays, Arpack, Random
include("Bfn.jl")
include("optweights.jl")
rng = MersenneTwister(234365689)
Plots.scalefontsizes(1.5)

# Create domain
N = 20
h = 1/N
vertices = [[i,j] for i = -N:N for j = -N:N if i^2+j^2 <= N^2 || (i >= 0 && j >= 0)]
fig1 = scatter([h*vx[1] for vx in vertices],[h*vx[2] for vx in vertices],label="grid",aspect_ratio=1.0,markersize=2)
display(fig1)
savefig(fig1,"domain1.pdf")

# Create Laplacian matrix
A,indexes = laplacian(vertices,h);

@time Achol = cholesky(A[2:end,2:end])

# weight vectors -- should be unit vectors
ws = [1.0 0.0 0.71; 0.0 1.0 0.71]
for i = 1:size(ws,2)
    ws[:,i] = ws[:,i]/norm(ws[:,i])
end
print("ws = "); display(ws)

hridge = 0.2
min_k,max_k = minmax_ridgeidxs(ws,vertices,h,hridge)

println("Compute C matrix")
C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k);

println("Compute normal matrix for C")
η = 1e-7
println("regulization parameter η = $η")
@time normalmx = cholesky(Matrix(C*C')+η*I);


println("Compute P1 & P2 projection matrices")
@time P1 = projnullC(C,1.0I(size(A,1)),normalmx=normalmx);
@time P2 = projnullC(C,projnullC(C,1.0I(size(A,1)),normalmx=normalmx),normalmx=normalmx);
println("Compute B = P2'.R.P2")
@time B = P2'*tAinvmul(P2,Achol);

println("Get eigenvalues/vectors of B")
@time evals,evecs = eigen(Symmetric(B));
maxeval, idx = findmax(evals)
v = evecs[:,idx]
scale = 1/(h*norm(v))
v = scale*v;

println("Largest eigenvalue of B = ",maxeval)

ws0 = ws
# now show the corresponding mu(x) ...
println("mu(x) as surface")
mu = maptoD(v,vertices);
minvxidxs = [minimum(vx[i] for vx in vertices) for i in 1:2]
maxvxidxs = [maximum(vx[i] for vx in vertices) for i in 1:2]
# ... first as a surface ...
fig2 = surface(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),mu)
display(fig2)
savefig(fig2,"mu-ws0-surf.pdf")
# ... then as a contour plot ...
println("mu(x) as contour")
fig3 = contour(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),mu,aspect_ratio=1.0,right_margin=6Plots.mm)
plot!(fig3,[cos(θ) for θ=pi/2:0.01:2pi],[sin(θ) for θ=pi/2:0.01:2pi],label=nothing,color=:black)
plot!(fig3,[1,1,0],[0,1,1],label=nothing,color=:black)
for i = 1:size(ws0,2)
    plot!(fig3,[-ws0[1,i],ws0[1,i]],[-ws0[2,i],ws0[2,i]],arrow=true,color=:black,linewidth=2,label=nothing)
end
display(fig3)
savefig(fig3,"mu-ws0.pdf")

# htaf = hard to approximate function 
htaf = tAinvmul(v,Achol)
htaf = htaf .- sum(htaf)/length(htaf)
scalef = 1/sqrt(h^2*dot(htaf,A*htaf))
htaf .= htaf .*scalef
htaf2 = maptoD(htaf,vertices)
# ... first as a surface ...
println("f(x) as surface")
fig4 = surface(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),htaf2)
display(fig4)
savefig(fig4,"htaf-ws0-surf.pdf")
# ... then as a contour plot ...
println("f(x) as contour")
fig5 = contour(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),htaf2,aspect_ratio=1.0,right_margin=6Plots.mm)
plot!(fig5,[cos(θ) for θ=pi/2:0.01:2pi],[sin(θ) for θ=pi/2:0.01:2pi],label=nothing,color=:black)
plot!(fig5,[1,1,0],[0,1,1],label=nothing,color=:black)
for i = 1:size(ws0,2)
    plot!(fig5,[-ws0[1,i],ws0[1,i]],[-ws0[2,i],ws0[2,i]],arrow=true,color=:black,linewidth=2,label=nothing)
end
display(fig5)
savefig(fig5,"htaf-ws0.pdf")

# now plot closest ridge approx'n
bestrf = C'*(normalmx \ (C*htaf))
# ... as contour plot ...
bestrf2 = maptoD(bestrf,vertices)
println("P_V(f)(x) as contour")
fig6 = contour(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),bestrf2,aspect_ratio=1.0,right_margin=6Plots.mm)
plot!(fig6,[cos(θ) for θ=pi/2:0.01:2pi],[sin(θ) for θ=pi/2:0.01:2pi],label=nothing,color=:black)
plot!(fig6,[1,1,0],[0,1,1],label=nothing,color=:black)
for i = 1:size(ws0,2)
    plot!(fig6,[-ws0[1,i],ws0[1,i]],[-ws0[2,i],ws0[2,i]],arrow=true,color=:black,linewidth=2,label=nothing)
end
display(fig6)
savefig(fig6,"bestrf-ws0.pdf")
println("f(x)-P_V(f)(x) as contour")
fig7 = contour(h*(minvxidxs[1]:maxvxidxs[1]),h*(minvxidxs[2]:maxvxidxs[2]),htaf2-bestrf2,aspect_ratio=1.0,right_margin=6Plots.mm)
plot!(fig7,[cos(θ) for θ=pi/2:0.01:2pi],[sin(θ) for θ=pi/2:0.01:2pi],label=nothing,color=:black)
plot!(fig7,[1,1,0],[0,1,1],label=nothing,color=:black)
for i = 1:size(ws0,2)
    plot!(fig7,[-ws0[1,i],ws0[1,i]],[-ws0[2,i],ws0[2,i]],arrow=true,color=:black,linewidth=2,label=nothing)
end
display(fig7)
savefig(fig7,"diff-ws0.pdf")

function createfuncns(ws::Matrix{T},vertices,h::T,hridge::T;η=zero(T)) where {T}
    A,indexes = laplacian(vertices,h);
    Achol = cholesky(A[2:end,2:end])
    min_k,max_k = minmax_ridgeidxs(ws,vertices,h,hridge)
    C,rowstart = Cmatrix(ws,h,hridge,vertices,min_k,max_k);
    normalmx = cholesky(Matrix(C*C')+η*I)
    P = projnullC(C,1.0I(size(A,1)),normalmx=normalmx)
    B = P'*tAinvmul(P,Achol)
    evals,evecs = eigen(Symmetric(B));
    maxeval, idx = findmax(evals)
    mu = evecs[:,idx]
    scale = 1/(h*norm(mu))
    mu = scale*mu
    mu2 = maptoD(mu,vertices)
    htaf = tAinvmul(v,Achol)
    htaf = htaf .- sum(htaf)/length(htaf)
    scalef = 1/sqrt(h^2*dot(htaf,A*htaf))
    htaf .= htaf .*scalef
    htaf2 = maptoD(htaf,vertices)
    bestrf = C'*(normalmx \ (C*htaf))
    bestrf2 = maptoD(bestrf,vertices)

    return maxeval,mu2,htaf2,bestrf2
end

println("ratio of L2 norms for bestrf to htaf: ",norm(bestrf)/norm(htaf))


# Now do the optimization
include("cg_prp2.jl")
include("wolfesearch.jl")
penalty(ws::Matrix{T}) where {T} = sum((dot(ws[:,i],ws[:,i])-1)^2 for i = 1:size(ws,2))
dpenalty(ws::Matrix{T}) where {T} = reduce(hcat,4*(dot(ws[:,i],ws[:,i])-1)*ws[:,i] for i = 1:size(ws,2))

alphas = 200.0*(2 .^collect(0:4))
@show typeof(alphas)
wstrialin = []
wstrialout = []
trialresults = []
M = 10.0
for trial = 1:5
    ws1 = randn(rng,size(ws))
    println(); println()
    println("==================================")
    print("ws start = "); display(ws1)
    push!(wstrialin,ws1)
    for alpha in alphas
        println("alpha = $alpha, M = $M")
        wslist = []
        results = cg_prp(ws->softmaxL2approxerr(ws,vertices,h,hridge,alpha;η=η) + M*penalty(ws),
                        ws->dsoftmaxL2approxerr(ws,vertices,h,hridge,alpha;η=η) + M*dpenalty(ws),
                        ws1,tol=1e-3,maxit=20,trace=1,xs=wslist)
        push!(trialresults,(alpha,results))
        ws1 = results[1] # extract weights
        print("ws1 = "); display(ws1)
        println()
    end
    push!(wstrialout,ws1)
end

println("That's all folks!")