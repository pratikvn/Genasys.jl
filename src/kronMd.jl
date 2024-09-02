using LinearAlgebra
using SparseArrays

"""
    grid2Dkron(Ax, Ay)

Returns a sparse matrix generated from input X and Y matrices with a kronecker product on a 2D grid.

# Examples
```julia-repl
julia>  Ax = sprand(2, 2, 0.5)
julia>  Ay = sprand(3, 3, 0.75)
julia>  grid2Dkron(Ax, Ay)
6×6 SparseMatrixCSC{Float64, Int64} with 14 stored entries:
 0.375082   ⋅        0.137573   ⋅        0.34465    ⋅
  ⋅        0.789502   ⋅        0.137573   ⋅        0.34465
 0.523429   ⋅        1.04008    ⋅         ⋅         ⋅
  ⋅        0.523429   ⋅        1.4545     ⋅         ⋅
  ⋅         ⋅        0.86456    ⋅        0.234737   ⋅
  ⋅         ⋅         ⋅        0.86456    ⋅        0.649157
```
"""
function grid2Dkron(Ax, Ay)
    Ix = sparse(I, size(Ax, 1), size(Ax, 2))
    Iy = sparse(I, size(Ay, 1), size(Ay, 2))
    return kron(Iy, Ax) + kron(Ay, Ix)
end


"""
    grid3Dkron(X, Y, Z)

Returns a sparse matrix generated from input X, Y and Z matrices with a kronecker product on a 3D grid.

# Examples
```julia-repl
julia>  Ax = sprand(2, 2, 0.5)
julia>  Ay = sprand(3, 3, 0.75)
julia>  Az = sprand(3, 3, 0.4)
julia>  grid3Dkron(Ax, Ay, Az)
18×18 SparseMatrixCSC{Float64, Int64} with 66 stored entries:
⎡⡻⢎⡢⠑⠄⠀⠀⠀⠀⎤
⎢⠈⠪⡛⣤⡀⠀⠀⠀⠀⎥
⎢⠀⠀⠈⠈⠱⣦⠢⡐⢄⎥
⎢⠀⠀⠀⠀⠈⠢⡻⢆⡀⎥
⎣⠀⠀⠀⠀⠀⠀⠈⠊⠛⎦
"""
function grid3Dkron(Ax, Ay, Az)
    Ix = sparse(I, size(Ax, 1), size(Ax, 2))
    Iy = sparse(I, size(Ay, 1), size(Ay, 2))
    Iz = sparse(I, size(Az, 1), size(Az, 2))
    return kron(Ix, kron(Iy, Az)) + kron(Ix, kron(Ay, Iz)) + kron(Ax, kron(Iy, Iz))
end
