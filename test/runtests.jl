#!/usr/bin/env julia

using Test

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim

@testset "Domain Chunk Paths" begin
    fixtures = normpath(joinpath(PROJECT_ROOT, "test", "fixtures", "chunks"))
    chunk = "chunk00000"
    paths = CryptoSim.DomainTypes.ChunkPaths(fixtures, chunk)
    @test isdir(paths.dir)
    @test isfile(paths.metadata)
    @test isfile(paths.chunk_config)
    @test occursin(chunk, CryptoSim.DomainTypes.default_julia_log_path(paths))
end

@testset "Chunk Loader" begin
    fixtures = normpath(joinpath(PROJECT_ROOT, "test", "fixtures", "chunks"))
    paths = CryptoSim.DomainTypes.ChunkPaths(fixtures, "chunk00000")
    chunk = CryptoSim.ChunkLoader.load_chunk(paths; data_dir=CryptoSim.DataIO.DEFAULT_DATA_DIR)
    @test chunk.config.n == length(chunk.price_vector)
    expected = CryptoSim.ChunkLoader.read_expected(paths)
    @test expected.volume > 0
end

@testset "Simulator Initialization" begin
    chunk_dir = normpath(joinpath(PROJECT_ROOT, "test", "fixtures", "chunks", "chunk00000"))
    cfg = CryptoSim.DataIO.load_config(joinpath(chunk_dir, "chunk-config.json"))
    cfg_obj = cfg.configurations[1]
    price_vec = ones(Float64, cfg_obj.n)
    state = CryptoSim.Simulator.SimulationState(cfg_obj, price_vec)
    @test length(state.trader.curve.x) == cfg_obj.n
    @test state.trader.curve.n == cfg_obj.n
    @test state.metrics.volume == 0.0
end
