using LinearAlgebra
using SparseArrays

"""
    stencilmat(n, stencil)

Returns a sparse matrix generated from a stencil.

# Examples
```julia-repl
julia>  laplacian(1, 1)
sparse([1], [1], [4.0])
```
"""
function stencilmat(n, stencil)
    n1 = Integer(floor((length(stencil) + 1) / 2))
    diags = []
    for i in eachindex(stencil)
        push!(diags, Pair(i - n1, stencil[i] * ones(n - abs(i - n1))))
    end
    return spdiagm(n, n, diags...)
end
