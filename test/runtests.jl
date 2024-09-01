using Genasys
using LinearAlgebra
using SparseArrays
using Test

@testset "laplacian tests" begin
    @test Genasys.laplacian(0, 0) == sparse([], [], [])
    @test Genasys.laplacian(1, 1) == sparse([1], [1], [4.0])
    @test Genasys.laplacian(1, 2) == sparse([1, 2, 1, 2], [1, 1, 2, 2], [4.0, -1.0, -1.0, 4.0], 2, 2)
    @test Genasys.laplacian(2, 1) == Genasys.laplacian(1, 2)
    @test Genasys.laplacian(2, 2) == sparse([1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0], 4, 4)
end

@testset "Stencil tests" begin
    @test Genasys.stencilmat(1, [-1 2 -1]) == sparse([1], [1], [2])
    @test Genasys.stencilmat(3, [-1 2 -1]) == sparse([1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], [2, -1, -1, 2, -1, -1, 2])
end

@testset "MtxIO tests" begin
    mat = sparse([1, 2], [1, 1], [2.5, 4.5])
    @test Genasys.read("data/testmat.mtx") == sparse([1], [1], [2.5])
    Genasys.write("tmp/testout.mtx", mat)
    @test Genasys.read("tmp/testout.mtx") == sparse([1, 2], [1, 1], [2.5, 4.5])
end
