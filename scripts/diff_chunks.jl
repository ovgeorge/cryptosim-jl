#!/usr/bin/env julia

using JSON3

const CPP_PREFIX = "CPPDBG "
const JULIA_PREFIX = "JULIADB "

struct LogEvent
    event::String
    candle_index::Int
    payload::JSON3.Object
end

strip_prefix(line::AbstractString) = begin
    startswith(line, CPP_PREFIX) && return line[length(CPP_PREFIX)+1:end]
    startswith(line, JULIA_PREFIX) && return line[length(JULIA_PREFIX)+1:end]
    return line
end

function read_log(path::AbstractString)
    events = LogEvent[]
    open(path, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            stripped = strip_prefix(stripped)
            payload = JSON3.read(stripped)
            event = String(payload["event"])
            candle = Int(payload["candle_index"])
            push!(events, LogEvent(event, candle, payload))
        end
    end
    return events
end

function event_key(ev::LogEvent)
    payload = ev.payload
    stage = haskey(payload, "stage") ? String(payload["stage"]) : nothing
    timestamp = haskey(payload, "timestamp") ? Int(payload["timestamp"]) : nothing
    return (ev.event, ev.candle_index, stage, timestamp)
end

function align_events(julia_events, cpp_events)
    julia_map = Dict(event_key(ev) => ev for ev in julia_events)
    cpp_map = Dict(event_key(ev) => ev for ev in cpp_events)
    matched = sort!(collect(intersect(keys(julia_map), keys(cpp_map))))
    missing_cpp = sort!(collect(setdiff(keys(julia_map), keys(cpp_map))))
    missing_julia = sort!(collect(setdiff(keys(cpp_map), keys(julia_map))))
    return [(julia_map[key], cpp_map[key]) for key in matched], missing_cpp, missing_julia
end

function values_equal(a, b; tol)
    if isa(a, Number) && isa(b, Number)
        return isapprox(Float64(a), Float64(b); atol=tol, rtol=tol)
    elseif isa(a, AbstractVector) && isa(b, AbstractVector)
        length(a) == length(b) || return false
        for i in eachindex(a)
            values_equal(a[i], b[i]; tol=tol) || return false
        end
        return true
    else
        return a == b
    end
end

function compare_payloads(julia_ev::LogEvent, cpp_ev::LogEvent; tol=1e-6)
    fields = intersect(keys(julia_ev.payload), keys(cpp_ev.payload))
    for key in fields
        val_j = julia_ev.payload[key]
        val_c = cpp_ev.payload[key]
        if !values_equal(val_j, val_c; tol=tol)
            return key, val_j, val_c
        end
    end
    return nothing
end

function summaries(events)
    counts = Dict{String,Int}()
    for ev in events
        counts[ev.event] = get(counts, ev.event, 0) + 1
    end
    return counts
end

function describe_missing(keys)
    return [string("event=", key[1], " candle=", key[2],
                   key[3] === nothing ? "" : " stage=$(key[3])",
                   key[4] === nothing ? "" : " ts=$(key[4])") for key in keys]
end

function main()
    if length(ARGS) != 2
        println("Usage: scripts/diff_chunks.jl <julia_log.jsonl> <cpp_log.jsonl>")
        exit(1)
    end
    julia_events = read_log(ARGS[1])
    cpp_events = read_log(ARGS[2])
    pairs, missing_cpp, missing_julia = align_events(julia_events, cpp_events)
    if isempty(pairs)
        println("No overlapping events found")
        exit(1)
    end
    for (j, c) in pairs
        mismatch = compare_payloads(j, c)
        if mismatch !== nothing
            key, val_j, val_c = mismatch
            println("Mismatch at event=$(j.event) candle=$(j.candle_index) field=$(key)")
            println("  Julia: $(val_j)")
            println("  C++  : $(val_c)")
            exit(2)
        end
    end
    if !isempty(missing_cpp)
        println("Warning: missing events in C++ log:")
        println("  " * join(describe_missing(missing_cpp), "\n  "))
    end
    if !isempty(missing_julia)
        println("Warning: missing events in Julia log:")
        println("  " * join(describe_missing(missing_julia), "\n  "))
    end
    println("Summaries:")
    println("  Julia: $(summaries(julia_events))")
    println("  C++  : $(summaries(cpp_events))")
    if isempty(missing_cpp) && isempty(missing_julia)
        println("Logs match for overlapping events.")
        exit(0)
    else
        exit(3)
    end
end

main()
