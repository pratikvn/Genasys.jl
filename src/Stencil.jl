using LinearAlgebra
using SparseArrays

"""
    stencilmat(n, stencil)

Returns a sparse matrix generated from a stencil.

# Examples
```julia-repl
julia>  stencilmat(4, [-1 2 -1])
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 10 stored entries:
  2.0  -1.0    ⋅     ⋅
 -1.0   2.0  -1.0    ⋅
   ⋅   -1.0   2.0  -1.0
   ⋅     ⋅   -1.0   2.0
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
