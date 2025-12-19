#!/usr/bin/env julia
"""
Grid search for stableswap (2-coin) across A and mid_fee, forcing out_fee = mid_fee.

Defaults mirror the legacy C++ grid scripts:
  * chunk_root: artifacts/chunks_ethusdt-1m-full (first 50 chunks by default)
  * data_dir: cryptopool-simulator/download
  * output: artifacts/experiments/stableswap2_grid/results.jsonl
  * A grid: logspace(0.05, 5.0, 8)
  * fee grid (mid_fee = out_fee): logspace(2e-4, 1e-2, 8)

Override via CLI flags:
  --chunk-root=PATH
  --config=PATH
  --dataset=PATH
  --data-dir=PATH
  --output=PATH
  --max-chunks=N
  --mode=per-chunk|combined
  --window-days=N --window-end=last|first
  --A-min=VAL --A-max=VAL --A-count=N
  --fee-min=VAL --fee-max=VAL --fee-count=N
"""

using JSON3
using Logging: @info
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim
const Sim = CryptoSim.Simulator
const Prep = CryptoSim.Preprocessing
const Loader = CryptoSim.ChunkLoader
const Domain = CryptoSim.DomainTypes
const DataIO = CryptoSim.DataIO
const Metrics = CryptoSim.Metrics

Base.@kwdef mutable struct Options
    chunk_root::String = normpath(joinpath(PROJECT_ROOT, "artifacts", "chunks_ethusdt-1m-full"))
    config_path::String = normpath(joinpath(PROJECT_ROOT, "configs", "ethusdt_chunk_config.json"))
    dataset_path::String = ""
    data_dir::String = DataIO.DEFAULT_DATA_DIR
    output::String = normpath(joinpath(PROJECT_ROOT, "artifacts", "experiments", "stableswap2_grid", "results.jsonl"))
    max_chunks::Int = 50
    mode::Symbol = :per_chunk
    window_days::Int = 365
    window_end::Symbol = :last
    A_min::Float64 = 0.05
    A_max::Float64 = 5.0
    A_count::Int = 8
    fee_min::Float64 = 2e-4
    fee_max::Float64 = 1e-2
    fee_count::Int = 8
end

function usage()
    println("""
    Usage: grid_search_stableswap2.jl [options]
      --chunk-root=PATH        Root directory holding chunkXXXXX/ (default: $(Options().chunk_root))
      --config=PATH            Simulator config template (default: $(Options().config_path))
      --dataset=PATH           Run on a single dataset path (no chunks); overrides config datafile(s)
      --data-dir=PATH          Where chunk JSON lives (default: $(DataIO.DEFAULT_DATA_DIR))
      --output=PATH            JSONL destination (default: $(Options().output))
      --max-chunks=N           Limit how many chunks to process (default: 50)
      --mode=per-chunk|combined Run per chunk (many rows) or on a combined trade stream (one row per grid point)
      --window-days=N          Restrict dataset run to the last/first N days (default: 365; only applies with --dataset)
      --window-end=last|first  Anchor the window at the dataset end or start (default: last; only applies with --dataset)
      --A-min=VAL              Smallest A in the log grid (default: 0.05)
      --A-max=VAL              Largest A in the log grid (default: 5.0)
      --A-count=N              Points in the A grid (default: 8)
      --fee-min=VAL            Smallest mid_fee/out_fee (default: 2e-4)
      --fee-max=VAL            Largest mid_fee/out_fee (default: 1e-2)
      --fee-count=N            Points in the fee grid (default: 8)
    """)
end

function parse_args()
    opts = Options()
    for arg in ARGS
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif startswith(arg, "--chunk-root=")
            opts.chunk_root = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--config=")
            opts.config_path = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--dataset=")
            opts.dataset_path = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--data-dir=")
            opts.data_dir = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--output=")
            opts.output = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--max-chunks=")
            opts.max_chunks = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--mode=")
            raw = lowercase(split(arg, '='; limit=2)[2])
            if raw in ("per-chunk", "per_chunk", "perchunk")
                opts.mode = :per_chunk
            elseif raw in ("combined", "concat", "bundle")
                opts.mode = :combined
            else
                error("Invalid --mode=$(raw). Expected per-chunk or combined.")
            end
        elseif startswith(arg, "--window-days=")
            opts.window_days = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--window-end=")
            raw = lowercase(split(arg, '='; limit=2)[2])
            if raw in ("last", "end", "tail")
                opts.window_end = :last
            elseif raw in ("first", "start", "head")
                opts.window_end = :first
            else
                error("Invalid --window-end=$(raw). Expected last or first.")
            end
        elseif startswith(arg, "--A-min=")
            opts.A_min = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--A-max=")
            opts.A_max = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--A-count=")
            opts.A_count = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--fee-min=")
            opts.fee_min = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--fee-max=")
            opts.fee_max = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--fee-count=")
            opts.fee_count = parse(Int, split(arg, '='; limit=2)[2])
        else
            error("Unrecognized argument: $(arg)")
        end
    end
    opts.A_min > 0 || error("A_min must be > 0 (got $(opts.A_min))")
    opts.A_max > 0 || error("A_max must be > 0 (got $(opts.A_max))")
    opts.fee_min > 0 || error("fee_min must be > 0 (got $(opts.fee_min))")
    opts.fee_max > 0 || error("fee_max must be > 0 (got $(opts.fee_max))")
    opts.A_max > opts.A_min || error("A_max must exceed A_min")
    opts.fee_max > opts.fee_min || error("fee_max must exceed fee_min")
    opts.A_count > 0 || error("A_count must be positive")
    opts.fee_count > 0 || error("fee_count must be positive")
    opts.max_chunks != 0 || error("max_chunks must be non-zero")
    (opts.window_days >= 0) || error("window_days must be >= 0")
    return opts
end

logspace(lo::Float64, hi::Float64, count::Int) =
    [10.0 ^ x for x in range(log10(lo), log10(hi); length=count)]

function list_chunks(root::AbstractString, max_chunks::Int)
    isdir(root) || error("Chunk root $(root) not found. Pass --chunk-root=PATH.")
    entries = filter(name -> startswith(name, "chunk"), sort(readdir(root)))
    isempty(entries) && error("No chunk* directories found under $(root)")
    if max_chunks > 0 && length(entries) > max_chunks
        return entries[1:max_chunks]
    elseif max_chunks < 0
        return entries
    else
        return entries
    end
end

@inline function override_config(base::DataIO.SimulationConfig, A::Float64, fee::Float64)
    return DataIO.SimulationConfig(
        A,
        base.gamma,
        base.D,
        base.n,
        fee,
        fee,
        base.fee_gamma,
        base.adjustment_step,
        base.allowed_extra_profit,
        base.ma_half_time,
        base.ext_fee,
        base.gas_fee,
        base.boost_rate,
        base.log,
    )
end

@inline function safe_dataset(meta::JSON3.Object)
    haskey(meta, "dataset") || return ("", "")
    obj = meta["dataset"]
    name = haskey(obj, "name") ? String(obj["name"]) : ""
    sha = haskey(obj, "sha256") ? String(obj["sha256"]) : ""
    return (name, sha)
end

function run_single(cfg::DataIO.SimulationConfig, price_vec, trades, paths::Domain.ChunkPaths,
                    meta::JSON3.Object, A::Float64, fee::Float64)
    state = Sim.SimulationState(cfg, price_vec)
    Sim.run_exact_simulation!(state, trades)
    metrics = Metrics.summarize(state.metrics)
    trader = state.trader
    dataset, dataset_sha = safe_dataset(meta)
    candle_range = if haskey(meta, "candle_range")
        cr = meta["candle_range"]
        (start = Int(cr["start"]), count = Int(cr["count"]))
    else
        nothing
    end
    return (
        status = "ok",
        chunk = String(paths.id),
        chunk_index = haskey(meta, "chunk_index") ? Int(meta["chunk_index"]) : nothing,
        chunk_size = haskey(meta, "chunk_size") ? Int(meta["chunk_size"]) : nothing,
        data_path = haskey(meta, "data_path") ? String(meta["data_path"]) : "",
        config_template = haskey(meta, "config_template") ? String(meta["config_template"]) : "",
        dataset = dataset,
        dataset_sha256 = dataset_sha,
        A = A,
        mid_fee = fee,
        out_fee = fee,
        gamma = cfg.gamma,
        fee_gamma = cfg.fee_gamma,
        adjustment_step = cfg.adjustment_step,
        ma_half_time = cfg.ma_half_time,
        ext_fee = cfg.ext_fee,
        gas_fee = cfg.gas_fee,
        boost_rate = cfg.boost_rate,
        metrics = metrics,
        profit = (
            dx = trader.profit.dx,
            D0 = trader.profit.D0,
            xcp = trader.profit.xcp,
            xcp_0 = trader.profit.xcp_0,
            xcp_profit = trader.profit.xcp_profit,
            xcp_profit_real = trader.profit.xcp_profit_real,
            apy = trader.profit.APY,
        ),
        tweak = (
            not_adjusted = trader.tweak.not_adjusted,
            is_light = trader.tweak.is_light,
        ),
        candle_range = candle_range,
    )
end

function run_chunk(io, paths::Domain.ChunkPaths, opts::Options, A_grid, fee_grid)
    data = Loader.load_chunk(paths; data_dir=opts.data_dir)
    data.config.n == 2 || error("Chunk $(paths) is not 2-coin (n=$(data.config.n))")
    trades = Prep.adapt_trades(data.trades) # reuse adapted trades across the grid
    for A in A_grid, fee in fee_grid
        cfg = override_config(data.config, A, fee)
        record = try
            merge(
                (mode = "per-chunk",),
                run_single(cfg, data.price_vector, trades, paths, data.metadata, A, fee),
            )
        catch err
            (
                status = "error",
                mode = "per-chunk",
                chunk = String(paths.id),
                A = A,
                mid_fee = fee,
                out_fee = fee,
                error = string(err),
            )
        end
        JSON3.write(io, record)
        println(io)
    end
end

@inline function config_signature(cfg::DataIO.SimulationConfig)
    return (
        gamma = cfg.gamma,
        D = cfg.D,
        n = cfg.n,
        fee_gamma = cfg.fee_gamma,
        adjustment_step = cfg.adjustment_step,
        allowed_extra_profit = cfg.allowed_extra_profit,
        ma_half_time = cfg.ma_half_time,
        ext_fee = cfg.ext_fee,
        gas_fee = cfg.gas_fee,
        boost_rate = cfg.boost_rate,
        log = cfg.log,
    )
end

function load_combined_bundle(chunks::Vector{String}, opts::Options)
    isempty(chunks) && error("No chunks selected for combined mode")
    base_cfg::Union{Nothing,DataIO.SimulationConfig} = nothing
    base_sig = nothing
    all_trades = DataIO.CPPTrade[]
    metas = JSON3.Object[]
    datafiles = String[]

    for chunk in chunks
        paths = Domain.ChunkPaths(opts.chunk_root, chunk)
        data = Loader.load_chunk(paths; data_dir=opts.data_dir)
        data.config.n == 2 || error("Chunk $(paths) is not 2-coin (n=$(data.config.n))")
        sig = config_signature(data.config)
        if base_sig === nothing
            base_cfg = data.config
            base_sig = sig
        elseif sig != base_sig
            error("Chunk $(paths) config differs from the first chunk (other than A/fees). Refuse to combine.")
        end
        append!(all_trades, data.trades)
        push!(metas, data.metadata)
        if haskey(data.metadata, "datafile")
            push!(datafiles, String(data.metadata["datafile"]))
        end
    end
    sort!(all_trades, by = tr -> tr.timestamp)
    cfg = base_cfg::DataIO.SimulationConfig
    price_vec = Loader.price_vector_from_trades(cfg.n, all_trades)
    trades = Prep.adapt_trades(all_trades)
    return (; cfg, price_vec, trades, metas, datafiles)
end

function run_combined(io, chunks::Vector{String}, opts::Options, A_grid, fee_grid)
    bundle = load_combined_bundle(chunks, opts)
    dataset_name, dataset_sha = safe_dataset(bundle.metas[1])
    dataset_path = if haskey(bundle.metas[1], "dataset") && haskey(bundle.metas[1]["dataset"], "path")
        String(bundle.metas[1]["dataset"]["path"])
    else
        ""
    end
    start_ts = isempty(bundle.trades) ? nothing : Int(bundle.trades[1].timestamp)
    end_ts = isempty(bundle.trades) ? nothing : Int(bundle.trades[end].timestamp)
    @info "combined bundle" chunks=length(chunks) trades=length(bundle.trades) start_ts=start_ts end_ts=end_ts

    nA = length(A_grid)
    nF = length(fee_grid)
    records = Vector{Any}(undef, nA * nF)
    Threads.@threads for idx in eachindex(records)
        iA = (idx - 1) ÷ nF + 1
        jF = (idx - 1) % nF + 1
        A = A_grid[iA]
        fee = fee_grid[jF]
        cfg = override_config(bundle.cfg, A, fee)
        records[idx] = try
            state = Sim.SimulationState(cfg, bundle.price_vec)
            Sim.run_exact_simulation!(state, bundle.trades)
            metrics = Metrics.summarize(state.metrics)
            trader = state.trader
            (
                status = "ok",
                mode = "combined",
                threads = Threads.nthreads(),
                chunk_root = opts.chunk_root,
                data_dir = opts.data_dir,
                chunks = chunks,
                datafiles = bundle.datafiles,
                dataset = dataset_name,
                dataset_path = dataset_path,
                dataset_sha256 = dataset_sha,
                A = A,
                mid_fee = fee,
                out_fee = fee,
                gamma = cfg.gamma,
                fee_gamma = cfg.fee_gamma,
                adjustment_step = cfg.adjustment_step,
                allowed_extra_profit = cfg.allowed_extra_profit,
                ma_half_time = cfg.ma_half_time,
                ext_fee = cfg.ext_fee,
                gas_fee = cfg.gas_fee,
                boost_rate = cfg.boost_rate,
                metrics = metrics,
                profit = (
                    dx = trader.profit.dx,
                    D0 = trader.profit.D0,
                    xcp = trader.profit.xcp,
                    xcp_0 = trader.profit.xcp_0,
                    xcp_profit = trader.profit.xcp_profit,
                    xcp_profit_real = trader.profit.xcp_profit_real,
                    apy = trader.profit.APY,
                ),
                tweak = (
                    not_adjusted = trader.tweak.not_adjusted,
                    is_light = trader.tweak.is_light,
                ),
            )
        catch err
            (
                status = "error",
                mode = "combined",
                threads = Threads.nthreads(),
                chunk_root = opts.chunk_root,
                data_dir = opts.data_dir,
                chunks = chunks,
                A = A,
                mid_fee = fee,
                out_fee = fee,
                error = string(err),
            )
        end
    end

    for record in records
        JSON3.write(io, record)
        println(io)
    end
end

function load_dataset_bundle(opts::Options)
    isfile(opts.config_path) || error("Config template $(opts.config_path) not found. Pass --config=PATH.")
    cfg_file = DataIO.load_config(opts.config_path)
    cfg = cfg_file.configurations[1]
    cfg.n == 2 || error("Only n=2 is supported for this script (got n=$(cfg.n))")

    source = isempty(opts.dataset_path) ? cfg_file.datafiles[1] : opts.dataset_path
    candles = DataIO.load_candles(source; pair = (0, 1), data_dir = opts.data_dir)
    isempty(candles) && error("No candles loaded from $(source)")
    sort!(candles, by = c -> c.timestamp)

    raw_start = minimum(c.timestamp for c in candles)
    raw_end = maximum(c.timestamp for c in candles)
    start_ts = raw_start
    end_ts = raw_end
    if opts.window_days > 0
        window = Int64(opts.window_days) * Int64(86400)
        if opts.window_end === :first
            start_ts = raw_start
            end_ts = raw_start + window
        else
            end_ts = raw_end
            start_ts = raw_end - window
        end
        candles = filter(c -> start_ts <= c.timestamp <= end_ts, candles)
        isempty(candles) && error("No candles remained after applying the window (start=$(start_ts), end=$(end_ts))")
    end

    trades = DataIO.CPPTrade[]
    sizehint!(trades, length(candles) * 2)
    candle_idx = 0
    for candle in candles
        DataIO.append_cpp_trades!(trades, candle_idx, candle, candle.pair)
        candle_idx += 2
    end
    sort!(trades, by = tr -> tr.timestamp)

    price_vec = Loader.price_vector_from_trades(cfg.n, trades)
    split_trades = Prep.adapt_trades(trades)
    return (; cfg, price_vec, trades = split_trades, candles = length(candles), start_ts, end_ts, source)
end

function run_dataset(io, opts::Options, A_grid, fee_grid)
    bundle = load_dataset_bundle(opts)
    @info "dataset bundle" source=bundle.source candles=bundle.candles trades=length(bundle.trades) start_ts=bundle.start_ts end_ts=bundle.end_ts threads=Threads.nthreads()

    nA = length(A_grid)
    nF = length(fee_grid)
    records = Vector{Any}(undef, nA * nF)
    Threads.@threads for idx in eachindex(records)
        iA = (idx - 1) ÷ nF + 1
        jF = (idx - 1) % nF + 1
        A = A_grid[iA]
        fee = fee_grid[jF]
        cfg = override_config(bundle.cfg, A, fee)
        records[idx] = try
            state = Sim.SimulationState(cfg, bundle.price_vec)
            Sim.run_exact_simulation!(state, bundle.trades)
            metrics = Metrics.summarize(state.metrics)
            trader = state.trader
            (
                status = "ok",
                mode = "dataset",
                threads = Threads.nthreads(),
                config_path = opts.config_path,
                data_dir = opts.data_dir,
                dataset_path = bundle.source,
                window_days = opts.window_days,
                window_end = String(opts.window_end),
                start_ts = bundle.start_ts,
                end_ts = bundle.end_ts,
                candle_count = bundle.candles,
                trade_count = length(bundle.trades),
                A = A,
                mid_fee = fee,
                out_fee = fee,
                gamma = cfg.gamma,
                fee_gamma = cfg.fee_gamma,
                adjustment_step = cfg.adjustment_step,
                allowed_extra_profit = cfg.allowed_extra_profit,
                ma_half_time = cfg.ma_half_time,
                ext_fee = cfg.ext_fee,
                gas_fee = cfg.gas_fee,
                boost_rate = cfg.boost_rate,
                metrics = metrics,
                profit = (
                    dx = trader.profit.dx,
                    D0 = trader.profit.D0,
                    xcp = trader.profit.xcp,
                    xcp_0 = trader.profit.xcp_0,
                    xcp_profit = trader.profit.xcp_profit,
                    xcp_profit_real = trader.profit.xcp_profit_real,
                    apy = trader.profit.APY,
                ),
                tweak = (
                    not_adjusted = trader.tweak.not_adjusted,
                    is_light = trader.tweak.is_light,
                ),
            )
        catch err
            (
                status = "error",
                mode = "dataset",
                threads = Threads.nthreads(),
                config_path = opts.config_path,
                data_dir = opts.data_dir,
                dataset_path = bundle.source,
                A = A,
                mid_fee = fee,
                out_fee = fee,
                error = string(err),
            )
        end
    end

    for record in records
        JSON3.write(io, record)
        println(io)
    end
end

function main()
    opts = parse_args()
    mkpath(dirname(opts.output))
    A_grid = logspace(opts.A_min, opts.A_max, opts.A_count)
    fee_grid = logspace(opts.fee_min, opts.fee_max, opts.fee_count)
    open(opts.output, "w") do io
        if !isempty(opts.dataset_path)
            @info "running grid" mode="dataset" config=opts.config_path dataset=opts.dataset_path data_dir=opts.data_dir output=opts.output
            run_dataset(io, opts, A_grid, fee_grid)
            return
        end

        chunks = list_chunks(opts.chunk_root, opts.max_chunks)
        @info "running grid" mode=String(opts.mode) chunk_root=opts.chunk_root data_dir=opts.data_dir output=opts.output chunks=length(chunks)
        if opts.mode === :combined
            run_combined(io, chunks, opts, A_grid, fee_grid)
        else
            for chunk in chunks
                paths = Domain.ChunkPaths(opts.chunk_root, chunk)
                run_chunk(io, paths, opts, A_grid, fee_grid)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
