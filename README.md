# Optimal-weights-for-ridge-functions
This contains code and a Julia notebook for a paper on finding optimal weight vectors for ridge function approximation in ``L^2``.

The files in this repository are the code for the paper *Finding Optimal Weight Vectors for Ridge Function Approximation in $L^{2}(D)$* by David E. Stewart.

Files included are:
* [`Bfn.jl`](Bfn.jl) contains code for the B-spline function and its derivatives.
* [`cg_prp2.jl`](cg_prp2.jl) has an implementation of the Polak-Ribiere conjugate gradient method
* [`wolfesearch.jl`](wolfesearch.jl) implements a Wolfe-condition based line search.
* [`optweights.jl`](optweights.jl) has most of the code specific to this paper.
* [This file](Optimal weights computation.ipynb) is the notebook showing how to use the above code, and perform the computations for the results in the paper.
