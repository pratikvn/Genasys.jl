using Genasys
using LinearAlgebra
using SparseArrays
using Test

@testset "Multi-dim Kronecker gen tests" begin
    @test Genasys.grid3Dkron(sparse(I, 3, 3), sparse(I, 2, 2), sparse(I, 3, 3)) == sparse(3 * I, 18, 18)
    @test Genasys.grid2Dkron(sparse(I, 3, 3), sparse(I, 2, 2)) == sparse(2 * I, 6, 6)
    @test Genasys.grid2Dkron(sparse(2.5 * I, 3, 3), sparse(I, 2, 2)) == sparse(3.5 * I, 6, 6)
end

@testset "Stencil tests" begin
    @test Genasys.stencilmat([-1 2 -1], 0) == sparse([], [], [])
    @test Genasys.stencilmat([-1 2 -1], 0, 0) == sparse([], [], [])
    @test Genasys.stencilmat([-1 2 -1], 1, 0, 0) == sparse([], [], [])
    @test Genasys.stencilmat([-1 2 -1], 1) == sparse([1], [1], [2])
    @test Genasys.stencilmat([-1 2 -1], 3) == sparse([1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], [2, -1, -1, 2, -1, -1, 2])
    @test Genasys.stencilmat([-1 2 -1], 2, 1) == sparse([1, 2, 1, 2], [1, 1, 2, 2], [4, -1, -1, 4])
    @test Genasys.stencilmat([-1 2 -1], 2, 3) == sparse([1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6], [1, 2, 3, 1, 2, 4, 1, 3, 4, 5, 2, 3, 4, 6, 3, 5, 6, 4, 5, 6], [4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4])
    @test Genasys.stencilmat([-1 2 -1], 2, 2, 1) == sparse([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4], [6, -1, -1, -1, 6, -1, -1, 6, -1, -1, -1, 6])
end

@testset "laplacian tests" begin
    @test Genasys.laplacian(1, 1) == sparse([1], [1], [4.0])
    @test Genasys.laplacian(1, 2) == sparse([1, 2, 1, 2], [1, 1, 2, 2], [4.0, -1.0, -1.0, 4.0], 2, 2)
    @test Genasys.laplacian(2, 1) == Genasys.laplacian(1, 2)
    @test Genasys.laplacian(2, 2) == sparse([1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0], 4, 4)
    @test Genasys.laplacian(2, 2, 1) == sparse([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4], [6, -1, -1, -1, 6, -1, -1, 6, -1, -1, -1, 6])
end

@testset "MtxIO tests" begin
    mat = sparse([1, 2], [1, 1], [2.5, 4.5])
    @test Genasys.read("data/testmat.mtx") == sparse([1], [1], [2.5])
    Genasys.write("tmp/testout.mtx", mat)
    @test Genasys.read("tmp/testout.mtx") == sparse([1, 2], [1, 1], [2.5, 4.5])
end
