#!/usr/bin/env julia

"""
Read `reports/ethusdt_full_chunk_summary.jsonl` and print relative-error
quantiles for the key metrics. Intended to run after
`scripts/full_parity_report.sh`.
"""

using JSON3
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const REPORT_PATH = normpath(joinpath(PROJECT_ROOT, "reports", "ethusdt_full_chunk_summary.jsonl"))
const METRIC_NAMES = (:volume, :slippage, :liquidity_density, :apy)
const QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

@inline function quantile(vals::Vector{Float64}, q::Float64)
    isempty(vals) && return NaN
    q <= 0 && return first(vals)
    q >= 1 && return last(vals)
    pos = q * (length(vals) - 1)
    lo = floor(Int, pos)
    hi = ceil(Int, pos)
    lo == hi && return vals[lo + 1]
    frac = pos - lo
    return vals[lo + 1] + (vals[hi + 1] - vals[lo + 1]) * frac
end

function build_metric_tables(results)
    values = Dict(name => Float64[] for name in METRIC_NAMES)
    worst = Dict(name => (chunk = "", rel = 0.0, abs = -Inf))
    log_ok = 0
    for result in results
        log_ok += result["log_ok"] ? 1 : 0
        metrics = result["metrics"]
        for name in METRIC_NAMES
            rel = Float64(metrics[string(name)]["rel_err"])
            abs_rel = abs(rel)
            push!(values[name], abs_rel)
            if abs_rel > worst[name].abs
                worst[name] = (chunk = String(result["chunk"]), rel = rel, abs = abs_rel)
            end
        end
    end
    for name in METRIC_NAMES
        sort!(values[name])
    end
    return values, worst, log_ok
end

function print_table(values, worst, total_chunks::Int)
    println("\nRelative-error quantiles (abs rel err) across $(total_chunks) chunks:\n")
    labels = ["metric"; ["q$(Int(q*1000)/10)" for q in QUANTILES]; ["max (chunk, rel)"]]
    widths = fill(14, length(labels))
    println(join(rpad.(labels, widths), ""))
    for name in METRIC_NAMES
        print(rpad(String(name), 14))
        for q in QUANTILES
            val = quantile(values[name], q)
            print(rpad(@sprintf("%.6e", val), 14))
        end
        worst_entry = worst[name]
        print(@sprintf("%.6e", worst_entry.abs))
        print(" ($(worst_entry.chunk), rel=")
        print(@sprintf("%+.6e", worst_entry.rel))
        print(")")
        println()
    end
end

function main()
    isfile(REPORT_PATH) || error("Summary file $(REPORT_PATH) not found. Run scripts/full_parity_report.sh first.")
    results = Vector{Dict{String,Any}}()
    open(REPORT_PATH, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(results, JSON3.read(line))
        end
    end
    isempty(results) && error("No rows found in $(REPORT_PATH)")
    values, worst, log_ok = build_metric_tables(results)
    println("Chunk logs matching C++: $(log_ok)/$(length(results))")
    print_table(values, worst, length(results))
end

main()
