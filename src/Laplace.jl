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
    o₁ = ones(n₁)
    ∂₁ = spdiagm(n₁ + 1, n₁, -1 => -o₁, 0 => o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm(n₂ + 1, n₂, -1 => -o₂, 0 => o₂)
    return kron(sparse(I, n₂, n₂), ∂₁' * ∂₁) + kron(∂₂' * ∂₂, sparse(I, n₁, n₁))
end

"""
    laplacian(n₁, n₂, n₃)

Returns a sparse matrix for the Laplacian on a 3D rectangular grid of size n₁ × n₂ × n₃ .

# Examples
```julia-repl
julia>  laplacian(1, 1)
sparse([1], [1], [4.0])
```
"""
function laplacian(n₁, n₂, n₃)
    o₁ = ones(n₁)
    ∂₁ = spdiagm(n₁ + 1, n₁, -1 => -o₁, 0 => o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm(n₂ + 1, n₂, -1 => -o₂, 0 => o₂)
    return kron(sparse(I, n₂, n₂), ∂₁' * ∂₁) + kron(∂₂' * ∂₂, sparse(I, n₁, n₁))
end
