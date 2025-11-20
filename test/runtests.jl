#!/usr/bin/env julia

using Test
using JSON3

drop_bottom_by_volume(trades, pct::Float64) = begin
    (pct <= 0 || isempty(trades)) && return trades
    pct >= 100 && return typeof(trades)()
    drop = clamp(floor(Int, length(trades) * pct / 100), 0, length(trades))
    drop == 0 && return trades
    perm = sortperm(eachindex(trades); by = i -> trades[i].volume, rev=false)
    keep = trues(length(trades))
    for idx in perm[1:drop]
        keep[idx] = false
    end
    return trades[keep]
end

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

@testset "Chunk Loader trim & smoke run" begin
    src_chunk = normpath(joinpath(PROJECT_ROOT, "artifacts", "chunks_ethusdt-1m-full", "chunk00000"))
    @test isdir(src_chunk)
    mktempdir() do tmp
        chunk_name = "chunk00000"
        dest_chunk = joinpath(tmp, chunk_name)
        cp(src_chunk, dest_chunk; force=true)
        meta_path = joinpath(dest_chunk, "metadata.json")
        meta = JSON3.read(read(meta_path, String), Dict{String,Any})
        meta["trim_flag"] = "trim0050"
        open(meta_path, "w") do io
            JSON3.write(io, meta)
        end
        paths = CryptoSim.DomainTypes.ChunkPaths(tmp, chunk_name)
        chunk = CryptoSim.ChunkLoader.load_chunk(paths; data_dir=CryptoSim.DataIO.DEFAULT_DATA_DIR)
        @test length(chunk.trades) == 50
        state = CryptoSim.Simulator.SimulationState(chunk.config, chunk.price_vector)
        splits = CryptoSim.Preprocessing.adapt_trades(chunk.trades)
        CryptoSim.Simulator.run_exact_simulation!(state, splits)
        metrics = CryptoSim.Metrics.summarize(state.metrics)
        @test metrics.volume > 0
    end
end

@testset "Chunk Loader ignore_bottom_pct" begin
    fixtures = normpath(joinpath(PROJECT_ROOT, "test", "fixtures", "chunks"))
    paths = CryptoSim.DomainTypes.ChunkPaths(fixtures, "chunk00000")
    meta = JSON3.read(read(paths.metadata, String), Dict{String,Any})
    data_dir = dirname(String(meta["datafiles"][1]["path"]))
    chunk = CryptoSim.ChunkLoader.load_chunk(paths; data_dir=data_dir)
    original_len = length(chunk.trades)
    filtered = drop_bottom_by_volume(chunk.trades, 10.0)
    expected_len = original_len - clamp(floor(Int, original_len * 0.10), 0, original_len)
    @test length(filtered) == expected_len
    state = CryptoSim.Simulator.SimulationState(chunk.config, chunk.price_vector)
    splits = CryptoSim.Preprocessing.adapt_trades(filtered)
    CryptoSim.Simulator.run_exact_simulation!(state, splits)
    metrics = CryptoSim.Metrics.summarize(state.metrics)
    @test metrics.volume > 0
end

@testset "Chunk Loader data sources" begin
    chunk_dir = normpath(joinpath(PROJECT_ROOT, "artifacts", "chunks_ethusdt-1m-full", "chunk00000"))
    cfg = CryptoSim.DataIO.load_config(joinpath(chunk_dir, "chunk-config.json"))
    mktempdir() do tmp
        @test_throws ErrorException CryptoSim.ChunkLoader.ensure_data_sources!(cfg, tmp)
    end
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

@testset "Chunk Summary module" begin
    Summary = CryptoSim.ChunkSummary
    base_measured = (
        volume = 1.0,
        slippage = 2.0,
        liquidity_density = 3.0,
        apy = 0.01,
    )
    base_expected = (
        volume = 1.0,
        slippage = 2.0,
        liquidity_density = 3.0,
        apy = 0.01,
    )
    summary = Summary.build_summary(
        "chunkA",
        base_measured,
        base_expected;
        log_ok=true,
        log_diff="",
        metadata=(data_dir="fixtures", ignore_bottom_pct=0.0),
    )
    dict = Summary.to_json_dict(summary)
    @test dict["chunk"] == "chunkA"
    roundtrip = Summary.from_json_dict(dict)
    @test roundtrip.chunk_id == "chunkA"
    @test roundtrip.metrics.volume.rel_err == 0.0
    @test roundtrip.metadata.data_dir == "fixtures"
    @test roundtrip.log_ok
    dict_no_meta = deepcopy(dict)
    pop!(dict_no_meta, "metadata", nothing)
    roundtrip2 = Summary.from_json_dict(dict_no_meta)
    @test isempty(roundtrip2.metadata)

    summary_with_delta(id, delta, log_ok=true) = Summary.build_summary(
        id,
        (
            volume = 1.0 + delta,
            slippage = 2.0 + delta,
            liquidity_density = 3.0 + delta,
            apy = 0.01 + delta,
        ),
        base_expected;
        log_ok=log_ok,
        log_diff="",
    )

    summaries = [
        summary,
        summary_with_delta("chunkB", 0.1, false),
        summary_with_delta("chunkC", -0.05, true),
    ]
    stats = Summary.build_stats(summaries; quantiles=(0.5, 0.99))
    @test stats.log_matches == 2
    @test stats.total == 3
    @test isapprox(Summary.quantile(stats, :volume, 0.5), 0.05; atol=1e-12)
    @test isapprox(Summary.quantile(stats, :volume, 0.99), 0.099; atol=1e-12)
    table = Summary.format_quantile_table(stats)
    @test occursin("chunkB", table)
    md_table = Summary.format_quantile_markdown(stats)
    @test occursin("| metric |", md_table)
    md_doc = Summary.render_markdown_report(
        stats;
        title="Demo Report",
        metadata=(dataset_name="ethusdt-1m-full", runner="cmd", notes=["note one"]),
    )
    @test occursin("# Demo Report", md_doc)
    @test occursin("ethusdt-1m-full", md_doc)
    @test occursin("note one", md_doc)
end

@testset "Synthetic step_for_price" begin
    cfg = CryptoSim.DataIO.SimulationConfig(
        A=0.3,
        gamma=0.0008,
        D=10.0,
        n=2,
        mid_fee=0.0004,
        out_fee=0.003,
        fee_gamma=0.0002,
        adjustment_step=1e-4,
        allowed_extra_profit=1e-9,
        ma_half_time=600,
        ext_fee=0.0,
        gas_fee=0.0,
        boost_rate=0.0,
        log=false,
    )
    price_vec = [1.0, 2000.0]
    logger = CryptoSim.Simulator.Instr.TradeLogger()
    state = CryptoSim.Simulator.SimulationState(cfg, price_vec; logger=logger)
    trader = state.trader
    dx = CryptoSim.Simulator.step_for_price(
        trader, 0.0, 2100.0, (0, 1), 0.0, 1.0, state.logger;
        stage=:dir1,
    )
    @test dx > 0
    dy = CryptoSim.Simulator.execute_trade!(trader, dx, 1, 2)
    @test dy > 0
end
