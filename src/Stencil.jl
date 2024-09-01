using LinearAlgebra
using SparseArrays

"""
    stencilmat(stencil, n)

Returns a sparse matrix generated from a stencil on a 1D grid with `n` points.

# Examples
```julia-repl
julia>  stencilmat([-1 2 -1], 4)
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 10 stored entries:
  2.0  -1.0    ⋅     ⋅
 -1.0   2.0  -1.0    ⋅
   ⋅   -1.0   2.0  -1.0
   ⋅     ⋅   -1.0   2.0
```
"""
function stencilmat(stencil, n)
    n1 = Integer(floor((length(stencil) + 1) / 2))
    diags = []
    for i in eachindex(stencil)
        push!(diags, Pair(i - n1, stencil[i] * ones(n - abs(i - n1))))
    end
    return spdiagm(n, n, diags...)
end


"""
    stencilmat(stencil, m, n)

Returns a sparse matrix generated from a stencil on a 2D grid with `m × n` points.

# Examples
```julia-repl
julia>  stencilmat([-1 2 -1], 3, 2)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 20 stored entries:
  4.0  -1.0    ⋅   -1.0    ⋅     ⋅
 -1.0   4.0  -1.0    ⋅   -1.0    ⋅
   ⋅   -1.0   4.0    ⋅     ⋅   -1.0
 -1.0    ⋅     ⋅    4.0  -1.0    ⋅
   ⋅   -1.0    ⋅   -1.0   4.0  -1.0
   ⋅     ⋅   -1.0    ⋅   -1.0   4.0
```
"""
function stencilmat(stencil, m, n)
    n1 = Integer(floor((length(stencil) + 1) / 2))
    diagsx = []
    diagsy = []
    for i in eachindex(stencil)
        push!(diagsx, Pair(i - n1, stencil[i] * ones(m - abs(i - n1))))
        push!(diagsy, Pair(i - n1, stencil[i] * ones(n - abs(i - n1))))
    end
    Ix = sparse(I, m, m)
    Iy = sparse(I, n, n)
    return kron(Iy, spdiagm(m, m, diagsx...)) + kron(spdiagm(n, n, diagsy...), Ix)
end


"""
    stencilmat(stencil, m, n, l)

Returns a sparse matrix generated from a stencil on a 3D grid with `m × n × l` points.

# Examples
```julia-repl
julia>  stencilmat([-1 2 -1], 2, 2, 2)
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
function stencilmat(stencil, m, n, l)
    n1 = Integer(floor((length(stencil) + 1) / 2))
    diagsx = []
    diagsy = []
    diagsz = []
    for i in eachindex(stencil)
        push!(diagsx, Pair(i - n1, stencil[i] * ones(m - abs(i - n1))))
        push!(diagsy, Pair(i - n1, stencil[i] * ones(n - abs(i - n1))))
        push!(diagsz, Pair(i - n1, stencil[i] * ones(l - abs(i - n1))))
    end
    Ix = sparse(I, m, m)
    Iy = sparse(I, n, n)
    Iz = sparse(I, l, l)
    return kron(Ix, kron(Iy, spdiagm(l, l, diagsz...))) + kron(Ix, kron(spdiagm(n, n, diagsy...), Iz)) + kron(spdiagm(n, n, diagsy...), kron(Ix, Iz))
end
