using LinearAlgebra

"""
    wolfesearch(f::Function,df::Function,x,p,alpha0;c1=0.01,c2=0.5,f0=nothing,df0=nothing,trace::Int=0)

Performs search for point satisfying the strong Wolfe conditions:
    
* `f(x+alpha p) <= f(x) + c1.alpha dot(p,df(x))`		(WC1)
* `|dot(p,df(x+alpha p))| <= c2|dot(p,df(x))|`		(WC2b)
Note that `df` is the gradient function of `f`.

The other parameters are:
* `alpha0` is the initial value of `alpha` in the search.  If `alpha0` satisfies the Wolfe conditions, then the search stops there. 
* `f0` is either `f(x)` at the initial `x` or `nothing`.
* `df0` is `df(x)` at the initial `x` or `nothing`.
* If `trace` is positive then print out information about the process
    
The algorithm follows that of Wright & Nocedal *Numerical Optimization*, pp. 60-62, section 3.4
    
Returns `(alpha,fval,dfval,nfe,nde)`
"""
function wolfesearch(f::Function,df::Function,x,p,alpha0;c1=0.01,c2=0.5,f0=nothing,df0=nothing,trace::Int=0)
        
        #= 
        Check parameters
        =#
        alpha = 0
        nfe = ( f0 == nothing ) ? 1 : 0
        nde = ( df0 == nothing ) ? 1 : 0
        f0 = ( f0 == nothing ) ? f(x) : f0
        df0 = ( df0 == nothing ) ? df(x) : df0
        fval = f0
        dfval = df0
        
        if ! ( 0 < c1 && c1 < c2 && c2 < 1 )
            println("wolfesearch: Error: Need 0 < c1 < c2 < 1");
            if trace > 0
                println("wolfesearch: c1 = $c1, c2 = $c2")
            end
            return (zero(alpha),f0,df0,0,0)
        end
        
        slope0 = dot(p,dfval);
        if slope0 >= 0
            println("wolfesearch: Error: Need a descent direction");
            if trace > 0
                println("wolfesearch: dot(p,df(x)) = $slope0");
            end
            return (zero(alpha),f0,df0,0,0)
        end
        
        #
        # Bracketing phase
        #
        if trace > 0
            println("wolfesearch: Bracketing phase: alpha = $alpha0");
        end
        alpha = alpha0;
        old_alpha = 0;
        old_fval  = f0;
        old_dfval = df0;
        old_slope = slope0;
        if trace > 0
            println("wolfesearch: f(x) = $f0, p''.df(x) = $slope0");
        end
        
        # Main loop
        firsttrip = true
        while true ### forever do...
            xplus = x+alpha*p;
            fval = f(xplus)
            dfval = df(xplus)
            nfe = nfe + 1;
            nde = nde + 1;
            if trace > 0
                println("wolfesearch: alpha = $alpha, f(x+alpha*p) = $fval");
            end
            if ( fval > f0 + c1*alpha*slope0 ) || ( ( ! firsttrip ) &&
                ( fval >= old_fval ) ) 
                if trace > 0
                    println("wolfesearch: (WC1) failed or f increased");
                end
                break;
            end
            if trace > 0
                println("wolfesearch: (WC1) holds & f decreased");
            end
            slope = dot(p,dfval);
            if trace > 0
                println("wolfesearch: dot(p,df(x+alpha*p)) = $slope");
            end
            if ( abs(slope) <= c2*abs(slope0) )
                if trace > 0
                    println("wolfesearch: (WC2) holds");
                end
                return (alpha,fval,dfval,nfe,nde)
            end
            if ( slope >= 0 )
                if trace > 0
                    println("wolfesearch: f''(alpha) >= 0");
                end
                break;
            end
            
            # Update variables -- note no upper limit on alpha
            temp = alpha;
            alpha = 2*alpha;
            old_alpha = temp;
            old_fval = fval;
            old_slope = slope;
        end
        
        if ( trace > 0 )
            println("wolfesearch: Entering ''zoom'' phase.")
        end
        
        #
        # "Zoom" phase
        #
        alpha_lo = old_alpha;
        alpha_hi =     alpha;
        f_lo     = old_fval;
        f_hi     =     fval;
        df_lo    = old_dfval;
        slope_lo = old_slope;
        
        if trace > 0
            println("wolfesearch: zoom phase: alpha_lo = $alpha_lo, alpha_hi = $alpha_hi")
        end
        
        iter_cnt = 0;
        while abs(alpha_hi-alpha_lo) > 1e-15*alpha0
            
            # form quadratic interpolant of function values on alpha_lo, alpha_hi
            # and the derivative at alpha_lo...
            # and find min of interpolant within interval [alpha_lo,alpha_hi]
            a = f_lo;
            b = slope_lo;
            dalpha = alpha_hi-alpha_lo;
            c = (f_hi - f_lo - dalpha*slope_lo)/dalpha^2;
            if ( ( c <= 0 ) | ( mod(iter_cnt,3) == 2 ) )
                # Use bisection
                alpha = alpha_lo + 0.5*dalpha;
                if trace > 0
                    println("wolfesearch: using bisection: c=$c");
                end
            else
                # Use min of quadratic
                alpha = alpha_lo - 0.5*b/c;
                if trace > 0
                    println("wolfesearch: using quadratic: a=$a, b=$b, c=$c");
                end
            end
            if trace > 0
                println("wolfesearch: alpha = $alpha");
            end
            
            # main part of loop
            xplus = x + alpha*p;
            fval = f(xplus)
            dfval = df(xplus)
            nfe = nfe + 1;
            nde = nde + 1;
            if trace > 0
                println("wolfe_search2: f(x+alpha*p) = $fval");
            end
            if ( ( fval > f0 + c1*alpha*slope0 ) | ( fval >= f_lo ) )
                if trace > 0
                    println("wolfe_search2: zoom: (WC1) fails or f increased");
                    println("wolfe_search2: zoom: update alpha_hi");
                end
                alpha_hi = alpha;
                f_hi   = fval;
                # df_hi = feval(df,xplus);
            else
                if trace > 0
                    println("wolfe_search2: zoom: (WC1) holds and f decreased");
                end
                slope = dot(p,dfval)
                if trace > 0
                    println("wolfe_search2: dot(p,df(x+alpha*p)) = $slope");
                end
                if ( abs(slope) <= c2*abs(slope0) )
                    if trace > 0
                        println("wolfe_search2: zoom: (WC2b) holds");
                    end
                    return (alpha,fval,dfval,nfe,nde)
                end
                if ( slope*dalpha >= 0 )
                    if trace > 0
                        println("wolfe_search2: zoom: alpha_hi <- alpha_lo & update alpha_lo");
                    end
                    alpha_hi = alpha_lo;
                    f_hi     = f_lo;
                    alpha_lo = alpha;
                    f_lo     = fval;
                    df_lo    = dfval;
                    slope_lo = slope;
                else
                    if trace > 0
                        println("wolfe_search2: zoom: update alpha_lo");
                    end
                    alpha_lo = alpha;
                    f_lo     = fval;
                    df_lo    = dfval;
                    slope_lo = slope;
                end
            end
            # Update iteration count
            iter_cnt = iter_cnt + 1;
            
        end # of forever do...
        
        (alpha,fval,dfval,nfe,nde)
    end