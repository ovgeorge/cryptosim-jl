#!/usr/bin/env julia

using JSON3
using SHA

# Load local module without requiring a Julia package environment
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim

const Sim = CryptoSim.Simulator
const DataIO = CryptoSim.DataIO
const Prep = CryptoSim.Preprocessing
const Instr = Sim.Instrumentation
const Domain = CryptoSim.DomainTypes
const Loader = CryptoSim.ChunkLoader
const Summary = CryptoSim.ChunkSummary

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

function dump_log(events, keep_logs::Bool, paths::Domain.ChunkPaths)
    path = keep_logs ? Domain.default_julia_log_path(paths) : tempname()
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
    paths = Domain.ChunkPaths(opts.chunk_root, chunk)
    chunk_id = String(paths.id)
    chunk_data = Loader.load_chunk(paths; data_dir=opts.data_dir)
    trades = opts.ignore_bottom_pct > 0 ?
        drop_bottom_by_volume(chunk_data.trades, opts.ignore_bottom_pct) :
        chunk_data.trades
    buffer = Vector{Any}()
    logger = Instr.TradeLogger(buffer)
    state = Sim.SimulationState(chunk_data.config, chunk_data.price_vector; logger=logger)
    splits = Prep.adapt_trades(trades)
    Sim.run_exact_simulation!(state, splits)
    expected = Loader.read_expected(paths)
    julia_log = dump_log(buffer, opts.keep_logs, paths)
    cpp_log = paths.cpp_log
    diff_success = isfile(cpp_log)
    diff_output = ""
    if diff_success
        diff_success, diff_output = run_diff(opts.diff_script, project_path, julia_log, cpp_log)
    else
        diff_output = "cpp_log.jsonl missing"
    end
    summary_metrics = CryptoSim.Metrics.summarize(state.metrics)
    config_sha = isfile(paths.chunk_config) ? bytes2hex(sha256(read(paths.chunk_config))) : ""
    meta_obj = chunk_data.metadata
    trim_flag = haskey(meta_obj, "trim_flag") ? String(meta_obj["trim_flag"]) : ""
    chunk_size = haskey(meta_obj, "chunk_size") ? Int(meta_obj["chunk_size"]) : nothing
    data_sources = haskey(meta_obj, "source_datasets") ? [String(ds["name"]) for ds in meta_obj["source_datasets"]] : String[]
    meta = (
        data_dir = opts.data_dir,
        ignore_bottom_pct = opts.ignore_bottom_pct,
        config_sha256 = config_sha,
        trim_flag = trim_flag,
        chunk_size = chunk_size,
        source_datasets = data_sources,
    )
    summary = Summary.build_summary(
        chunk_id,
        summary_metrics,
        expected;
        log_ok=diff_success,
        log_diff=diff_output,
        metadata=meta,
    )
    return Summary.to_json_dict(summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    chunks, opts, project_path = parse_args()
    for chunk in chunks
        result = summarize_chunk(chunk, opts, project_path)
        JSON3.write(stdout, result)
        println()
    end
end
