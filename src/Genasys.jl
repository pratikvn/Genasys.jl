module Genasys

import LinearAlgebra
import SparseArrays

# Write your package code here.

export write, read, grid2Dkron, grid3Dkron, laplacian, stencilmat


include("Laplace.jl")
include("MtxIO.jl")
include("Stencil.jl")
include("kronMd.jl")

end
