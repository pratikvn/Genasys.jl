using Genasys
using LinearAlgebra
using SparseArrays
using Test

@testset "∇² tests" begin
    @test Genasys.∇²(0, 0) == sparse([], [], [])
    @test Genasys.∇²(1, 1) == sparse([1], [1], [4.0])
    @test Genasys.∇²(1, 2) == sparse([1, 2, 1, 2], [1, 1, 2, 2], [4.0, -1.0, -1.0, 4.0], 2, 2)
    @test Genasys.∇²(2, 1) == Genasys.∇²(1, 2)
    @test Genasys.∇²(2, 2) == sparse([1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0], 4, 4)
end

@testset "MtxIO tests" begin
    mat = sparse([1, 2], [1, 1], [2.5, 4.5])
    @test Genasys.read("data/testmat.mtx") == sparse([1], [1], [2.5])
    Genasys.write("data/testout.mtx", mat)
    @test Genasys.read("data/testout.mtx") == sparse([1, 2], [1, 1], [2.5, 4.5])
end
