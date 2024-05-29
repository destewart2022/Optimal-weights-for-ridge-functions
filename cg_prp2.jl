"""
    cg_prp(f,df,x,tol;c1=0.01,c2=0.1,maxit::Int=10_000,trace::Int=0,xs=nothing)
Polak-Ribiere conjugate gradient method (without restart)
Uses Wolfe line-search conditions (parameters c1 and c2)
* `f` function to minimize
* `df` gradient of `f`
* `x` is the current position
* `tol` is the stopping tolerance: `||df(x)|| < tol`
* `0 < c2 < c1 < 1` are the Wolfe line-search parameters
* `maxit` is the iteration limit
* `trace` If non-zero, then print out information about process

Returns `(x,fval,gval,nfe,nge)`
* `x` solution estimate
* `fval` function value at `x`
* `gval` gradient at `x`
* `nfe` number of function evaluations
* `nge` number of gradient evaluations
"""
function cg_prp(f::Function, df::Function, x; tol=1e-3, c1=0.01, c2=0.1, maxit::Int=10_000, trace::Int=0, xs=nothing)
    fval = f(x)
    g = df(x)
    nfe = 1
    nge = 1
    p = -g # initial p is negative gradient 
    g_sqr = dot(g, g)
    alpha = 1.0 # make the starting step length adaptive: use whatever the previous step was
    for k = 0:maxit
        if xs != nothing
            push!(xs, x)
        end
        if trace > 1
            println("cg_prp: cos(theta) = ", -dot(p, g) / (norm(p) * norm(g)))
            println("cg_prp: ||p||/||g|| = ", norm(p) / norm(g))
        end
        # line search using Wolfe based search
        (alpha, fval, new_g, nfe_ws, nge_ws) =
            wolfesearch(f, df, x, p, alpha, c1=c1, c2=c2, f0=fval, df0=g, trace=max(trace - 1, 0))
        nfe = nfe + nfe_ws
        nge = nge + nge_ws
        if trace > 0
            println("cg_prp: iter # $k, alpha = $alpha, function value = $fval, ||g|| = $(norm(g))")
        end
        x = x + alpha * p
        new_g_sqr = dot(new_g, new_g)
        # Polak Ribiere modification
        beta = max(dot(new_g, new_g - g) / g_sqr, 0)
        if trace > 0
            println("cg_prp: beta = $beta")
        end
        g_sqr = new_g_sqr
        g = new_g
        if norm(g) < tol
            # push!(xs,x)
            return (x, fval, g, nfe, nge)
        end
        # Restart code (if desired)
        # if k % n == n-1
        #   beta = 0;
        # end
        p = -g + beta * p
        if dot(p,g) >= 0 # emergency break!!
            if trace > 0
                println("cg_prp: p is not a descent direction; replace p with -g")
            end
            p = -g
        end
    end
    if trace > 0
        println("cg_prp: Have used $maxit iterations; returning without finding solution")
    end
    return (x, fval, g, nfe, nge)
end

