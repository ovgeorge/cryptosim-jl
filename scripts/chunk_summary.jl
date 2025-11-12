#!/usr/bin/env julia

using JSON3

# Load local module without requiring a Julia package environment
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim

const Sim = CryptoSim.Simulator
const DataIO = CryptoSim.DataIO
const Prep = CryptoSim.Preprocessing
const Instr = Sim.Instrumentation

struct Options
    chunk_root::String
    data_dir::String
    diff_script::String
    keep_logs::Bool
    ignore_bottom_pct::Float64
end

function drop_bottom_by_volume(trades, pct::Float64)
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

function default_paths()
    project_root = normpath(joinpath(@__DIR__, ".."))
    chunk_root = normpath(joinpath(project_root, "test", "fixtures", "chunks"))
    diff_script = normpath(joinpath(project_root, "scripts", "diff_chunks.jl"))
    return (; project_root, chunk_root, diff_script)
end

function parse_args()
    paths = default_paths()
    project_path = paths.project_root
    data_dir = DataIO.DEFAULT_DATA_DIR
    keep_logs = false
    ignore_bottom_pct = 0.0
    chunks = String[]
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--root=")
            paths = merge(paths, (; chunk_root = normpath(split(arg, '='; limit=2)[2])))
        elseif startswith(arg, "--data-dir=")
            data_dir = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--diff-script=")
            paths = merge(paths, (; diff_script = normpath(split(arg, '='; limit=2)[2])))
        elseif arg == "--keep-logs"
            keep_logs = true
        elseif startswith(arg, "--ignore-bottom-pct=")
            ignore_bottom_pct = parse(Float64, split(arg, '='; limit=2)[2])
        else
            push!(chunks, arg)
        end
        i += 1
    end
    isempty(chunks) && error("Usage: chunk_summary.jl [--root=...] chunk_id ...")
    if ignore_bottom_pct < 0 || ignore_bottom_pct >= 100
        error("--ignore-bottom-pct must be in [0, 100)")
    end
    opts = Options(paths.chunk_root, data_dir, paths.diff_script, keep_logs, ignore_bottom_pct)
    return chunks, opts, project_path
end

function ensure_data_sources!(cfg::DataIO.ConfigFile, data_dir::AbstractString)
    for source in cfg.datafiles
        file = endswith(lowercase(source), ".json") ? source : string(source, ".json")
        path = joinpath(data_dir, file)
        isfile(path) || error("Missing data source $(path). Place the raw candle JSON under $(data_dir).")
    end
end

function read_metadata(chunk_dir::AbstractString)
    meta_path = joinpath(chunk_dir, "metadata.json")
    isfile(meta_path) || error("metadata.json missing in $(chunk_dir)")
    return JSON3.read(meta_path)
end

function price_vector_from_cpp_trades(n::Int, trades)
    prices = ones(Float64, n)
    seen = falses(n)
    seen[1] = true
    for trade in trades
        a, b = trade.pair
        if a == 0 && b + 1 <= n
            prices[b + 1] = trade.close
            seen[b + 1] = true
        elseif b == 0 && a + 1 <= n && trade.close != 0
            prices[a + 1] = 1 / trade.close
            seen[a + 1] = true
        end
        all(seen) && break
    end
    return prices
end

parse_trim(flag::AbstractString) = parse(Int, replace(flag, "trim" => ""))

function load_chunk(chunk_dir::AbstractString, data_dir::AbstractString)
    metadata = read_metadata(chunk_dir)
    config_path = joinpath(chunk_dir, "chunk-config.json")
    cfg_file = if isfile(config_path)
        DataIO.load_config(config_path)
    else
        # Fallback to a copied config (as produced by capture_chunks.sh)
        cfg_name = haskey(metadata, "config") ? String(metadata["config"]) : "single-run.json"
        DataIO.load_config(joinpath(chunk_dir, cfg_name))
    end
    ensure_data_sources!(cfg_file, data_dir)
    trades = if haskey(metadata, "trim_flag")
        trim = parse_trim(String(metadata["trim_flag"]))
        DataIO.build_cpp_trades(cfg_file; data_dir=data_dir, trim=trim)
    else
        DataIO.build_cpp_trades(cfg_file; data_dir=data_dir)
    end
    cfg = cfg_file.configurations[1]
    price_vec = price_vector_from_cpp_trades(cfg.n, trades)
    return (; cfg, price_vec, data=trades)
end

function read_expected(chunk_dir::AbstractString)
    results = open(joinpath(chunk_dir, "results.json"), "r") do io
        JSON3.read(io)
    end
    metrics = results["configuration"][1]["Result"]
    return (
        volume = Float64(metrics["volume"]),
        slippage = Float64(metrics["slippage"]),
        liquidity_density = Float64(metrics["liq_density"]),
        apy = Float64(metrics["APY"]),
    )
end

function dump_log(events, keep_logs::Bool, chunk_dir::AbstractString, chunk_id::AbstractString)
    path = keep_logs ? joinpath(chunk_dir, "julia_log.$(chunk_id).jsonl") : tempname()
    open(path, "w") do io
        for ev in events
            JSON3.write(io, ev)
            println(io)
        end
    end
    return path
end

function run_diff(diff_script::AbstractString, project_path::AbstractString, julia_log::AbstractString, cpp_log::AbstractString)
    cmd = Cmd(["julia", "--project=$(project_path)", diff_script, julia_log, cpp_log])
    buffer = IOBuffer()
    success = true
    try
        run(pipeline(cmd, stdout=buffer, stderr=buffer))
    catch err
        if err isa Base.ProcessFailedException
            success = false
        else
            rethrow(err)
        end
    end
    seek(buffer, 0)
    return success, String(take!(buffer))
end

function summarize_chunk(chunk::String, opts::Options, project_path::String)
    chunk_dir = isdir(chunk) ? normpath(chunk) : normpath(joinpath(opts.chunk_root, chunk))
    isdir(chunk_dir) || error("Chunk directory $(chunk_dir) not found")
    chunk_id = basename(chunk_dir)
    chunk_data = load_chunk(chunk_dir, opts.data_dir)
    trades = opts.ignore_bottom_pct > 0 ?
        drop_bottom_by_volume(chunk_data.data, opts.ignore_bottom_pct) :
        chunk_data.data
    buffer = Vector{Any}()
    logger = Instr.TradeLogger(buffer)
    state = Sim.SimulationState(chunk_data.cfg, chunk_data.price_vec; logger=logger)
    splits = Prep.adapt_trades(trades)
    Sim.run_exact_simulation!(state, splits)
    expected = read_expected(chunk_dir)
    summary = CryptoSim.Metrics.summarize(state.metrics)
    metrics = Dict{String,Any}()
    rel_errors = Dict{Symbol,Float64}()
    for name in (:volume, :slippage, :liquidity_density, :apy)
        actual = getfield(summary, name)
        ref = getfield(expected, name)
        rel = ref == 0 ? 0.0 : (actual - ref) / ref
        rel_errors[name] = rel
        metrics[string(name)] = Dict("julia" => actual, "cpp" => ref, "rel_err" => rel)
    end
    julia_log = dump_log(buffer, opts.keep_logs, chunk_dir, chunk_id)
    cpp_log = joinpath(chunk_dir, "cpp_log.jsonl")
    diff_success = isfile(cpp_log)
    diff_output = ""
    if diff_success
        diff_success, diff_output = run_diff(opts.diff_script, project_path, julia_log, cpp_log)
    else
        diff_output = "cpp_log.jsonl missing"
    end
    return Dict(
        "chunk" => chunk_id,
        "metrics" => metrics,
        "log_ok" => diff_success,
        "log_diff" => diff_success ? "" : diff_output,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    chunks, opts, project_path = parse_args()
    for chunk in chunks
        result = summarize_chunk(chunk, opts, project_path)
        JSON3.write(stdout, result)
        println()
    end
end
