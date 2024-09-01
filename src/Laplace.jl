using LinearAlgebra
using SparseArrays

"""
    laplacian(n₁, n₂)

Returns a sparse matrix for the Laplacian on a 2D rectangular grid of size n₁ x n₂ .

# Examples
```julia-repl
julia>  laplacian(1, 1)
sparse([1], [1], [4.0])
```
"""
function laplacian(n₁, n₂)
    return stencilmat([-1, 2, -1], n₁, n₂)
end

"""
    laplacian(n₁, n₂, n₃)

Returns a sparse matrix for the Laplacian on a 3D rectangular grid of size n₁ × n₂ × n₃ .

# Examples
```julia-repl
julia>  laplacian(2, 2, 2)
8×8 SparseArrays.SparseMatrixCSC{Float64, Int64} with 32 stored entries:
  6.0  -1.0  -1.0    ⋅   -1.0    ⋅     ⋅     ⋅
 -1.0   6.0    ⋅   -1.0    ⋅   -1.0    ⋅     ⋅
 -1.0    ⋅    6.0  -1.0    ⋅     ⋅   -1.0    ⋅
   ⋅   -1.0  -1.0   6.0    ⋅     ⋅     ⋅   -1.0
 -1.0    ⋅     ⋅     ⋅    6.0  -1.0  -1.0    ⋅
   ⋅   -1.0    ⋅     ⋅   -1.0   6.0    ⋅   -1.0
   ⋅     ⋅   -1.0    ⋅   -1.0    ⋅    6.0  -1.0
   ⋅     ⋅     ⋅   -1.0    ⋅   -1.0  -1.0   6.0
```
"""
function laplacian(n₁, n₂, n₃)
    return stencilmat([-1, 2, -1], n₁, n₂, n₃)
end
