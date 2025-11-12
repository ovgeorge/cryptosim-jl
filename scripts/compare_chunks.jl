#!/usr/bin/env julia

using JSON3
using Printf: @sprintf
using CryptoSim

const Sim = CryptoSim.Simulator
const DataIO = CryptoSim.DataIO
const Prep = CryptoSim.Preprocessing
const Instr = Sim.Instrumentation

const STEP_TRACE_ENV = "JULIA_STEP_TRACE"

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

struct ChunkReport
    chunk::String
    summary::NamedTuple
    expected::NamedTuple
    errors::Dict{Symbol,Float64}
    diff_success::Bool
    diff_output::String
    log_path::Union{Nothing,String}
end

function default_paths()
    project_root = normpath(joinpath(@__DIR__, ".."))
    chunk_root = normpath(joinpath(project_root, "test", "fixtures", "chunks"))
    diff_script = normpath(joinpath(project_root, "scripts", "diff_chunks.jl"))
    return (; project_root, chunk_root, diff_script)
end

function parse_args()
    paths = default_paths()
    data_dir = DataIO.DEFAULT_DATA_DIR
    tol = 1e-3
    keep_logs = false
    step_trace = nothing
    ignore_bottom_pct = 0.0
    chunks = String[]
    project_override = ""
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--root=")
            paths = merge(paths, (; chunk_root = normpath(split(arg, '='; limit=2)[2])))
        elseif startswith(arg, "--diff-script=")
            paths = merge(paths, (; diff_script = normpath(split(arg, '='; limit=2)[2])))
        elseif startswith(arg, "--data-dir=")
            data_dir = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--project=")
            project_override = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--step-trace=")
            value = split(arg, '='; limit=2)[2]
            step_trace = isempty(value) ? nothing : value
        elseif startswith(arg, "--ignore-bottom-pct=")
            ignore_bottom_pct = parse(Float64, split(arg, '='; limit=2)[2])
        elseif arg == "--keep-logs"
            keep_logs = true
        else
            push!(chunks, arg)
        end
        i += 1
    end
    if isempty(chunks)
        chunks = sort(filter(name -> startswith(name, "chunk") && isdir(joinpath(paths.chunk_root, name)),
                              readdir(paths.chunk_root)))
    end
    if ignore_bottom_pct < 0 || ignore_bottom_pct >= 100
        error("--ignore-bottom-pct must be in [0, 100)")
    end
    project_path = isempty(project_override) ? paths.project_root : project_override
    return (; chunks, chunk_root = paths.chunk_root, diff_script = paths.diff_script,
             data_dir, tol, keep_logs, project_path, step_trace, ignore_bottom_pct)
end

parse_trim(flag::AbstractString) = parse(Int, replace(flag, "trim" => ""))

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

function load_chunk(chunk_dir::AbstractString, data_dir::AbstractString)
    config_path = joinpath(chunk_dir, "chunk-config.json")
    cfg_file = DataIO.load_config(config_path)
    ensure_data_sources!(cfg_file, data_dir)
    metadata = read_metadata(chunk_dir)
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
    if keep_logs
        path = joinpath(chunk_dir, "julia_log.$(chunk_id).jsonl")
    else
        path = tempname()
    end
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

function with_step_trace(f::Function, target::Union{Nothing,String})
    target === nothing && return f()
    prev = haskey(ENV, STEP_TRACE_ENV) ? ENV[STEP_TRACE_ENV] : nothing
    if target === ""
        haskey(ENV, STEP_TRACE_ENV) && delete!(ENV, STEP_TRACE_ENV)
    else
        ENV[STEP_TRACE_ENV] = target
    end
    try
        return f()
    finally
        if prev === nothing
            haskey(ENV, STEP_TRACE_ENV) && delete!(ENV, STEP_TRACE_ENV)
        else
            ENV[STEP_TRACE_ENV] = prev
        end
    end
end

rel_err(actual, expected) = expected == 0 ? abs(actual - expected) : abs(actual - expected) / abs(expected)

function process_chunk(chunk_arg::AbstractString, opts)
    chunk_dir = isdir(chunk_arg) ? normpath(chunk_arg) : normpath(joinpath(opts.chunk_root, chunk_arg))
    isdir(chunk_dir) || error("Chunk directory $(chunk_dir) not found")
    chunk_id = basename(chunk_dir)
    chunk = load_chunk(chunk_dir, opts.data_dir)
    buffer = Any[]
    logger = Instr.TradeLogger(buffer)
    trace_target = isnothing(opts.step_trace) ? nothing : replace(opts.step_trace, "{chunk}" => chunk_id)
    state = with_step_trace(trace_target) do
        local_state = Sim.SimulationState(chunk.cfg, chunk.price_vec; logger=logger)
        trades = opts.ignore_bottom_pct > 0 ?
            drop_bottom_by_volume(chunk.data, opts.ignore_bottom_pct) :
            chunk.data
        splits = Prep.adapt_trades(trades)
        Sim.run_exact_simulation!(local_state, splits)
        local_state
    end
    summary = CryptoSim.Metrics.summarize(state.metrics)
    expected = read_expected(chunk_dir)
    errors = Dict{Symbol,Float64}()
    for name in (:volume, :slippage, :liquidity_density, :apy)
        actual = getfield(summary, name)
        target = getfield(expected, name)
        errors[name] = rel_err(actual, target)
    end
    julia_log = dump_log(buffer, opts.keep_logs, chunk_dir, chunk_id)
    cpp_log = joinpath(chunk_dir, "cpp_log.jsonl")
    diff_success, diff_output = run_diff(opts.diff_script, opts.project_path, julia_log, cpp_log)
    log_path = opts.keep_logs ? julia_log : nothing
    !opts.keep_logs && isfile(julia_log) && rm(julia_log; force=true)
    return ChunkReport(chunk_id, summary, expected, errors, diff_success, diff_output, log_path)
end

function format_kpi_line(name::Symbol, summary::NamedTuple, expected::NamedTuple, err::Float64, tol::Float64)
    status = err <= tol ? "OK" : "FAIL"
    actual = getfield(summary, name)
    target = getfield(expected, name)
    return @sprintf("  %-18s %s (julia=%0.6f cpp=%0.6f rel_err=%0.4e)", String(name), status, actual, target, err)
end

function print_report(report::ChunkReport, tol::Float64)
    println("=== Chunk $(report.chunk) ===")
    for name in (:volume, :slippage, :liquidity_density, :apy)
        line = format_kpi_line(name, report.summary, report.expected, report.errors[name], tol)
        println(line)
    end
    diff_status = report.diff_success ? "OK" : "FAIL"
    println("  logs              $(diff_status)")
    if !report.diff_success && !isempty(strip(report.diff_output))
        println("    " * replace(strip(report.diff_output), "\n" => "\n    "))
    end
    if report.log_path !== nothing
        println("  julia_log        $(report.log_path)")
    end
    println()
end

function main()
    opts = parse_args()
    isempty(opts.chunks) && error("No chunks to process.")
    reports = ChunkReport[]
    for chunk in opts.chunks
        report = process_chunk(chunk, opts)
        push!(reports, report)
        print_report(report, opts.tol)
    end
    metric_pass = all(all(err <= opts.tol for err in values(rep.errors)) for rep in reports)
    diff_pass = all(rep.diff_success for rep in reports)
    overall = metric_pass && diff_pass
    println(overall ? "All chunks within tolerance." : "One or more chunks exceed tolerances.")
    exit(overall ? 0 : 1)
end

main()
