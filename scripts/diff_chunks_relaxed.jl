#!/usr/bin/env julia

using JSON3

const CPP_PREFIX = "CPPDBG "
const JULIA_PREFIX = "JULIADB "

struct LogEvent
    event::String
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
            push!(events, LogEvent(event, payload))
        end
    end
    return events
end

function event_key(ev::LogEvent)
    p = ev.payload
    # Use a relaxed key that tolerates differing candle_index conventions
    stage = haskey(p, "stage") ? String(p["stage"]) : (haskey(p, "mode") ? String(p["mode"]) : nothing)
    timestamp = haskey(p, "timestamp") ? Int(p["timestamp"]) : nothing
    pair = if haskey(p, "pair")
        v = p["pair"]
        try
            (Int(v[1]), Int(v[2]))
        catch
            nothing
        end
    else
        nothing
    end
    return (ev.event, timestamp, pair, stage)
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
    return [string("event=", key[1], " ts=", key[2],
                   key[3] === nothing ? "" : " pair=" * string(key[3]),
                   key[4] === nothing ? "" : " stage=" * String(key[4])) for key in keys]
end

function main()
    if length(ARGS) != 2
        println("Usage: scripts/diff_chunks_relaxed.jl <julia_log.jsonl> <cpp_log.jsonl>")
        exit(1)
    end
    julia_events = read_log(ARGS[1])
    cpp_events = read_log(ARGS[2])
    julia_map = Dict(event_key(ev) => ev for ev in julia_events)
    cpp_map = Dict(event_key(ev) => ev for ev in cpp_events)
    matched_keys = sort!(collect(intersect(keys(julia_map), keys(cpp_map))))
    if isempty(matched_keys)
        println("No overlapping events found (relaxed key)")
        println("  Julia summary: $(summaries(julia_events))")
        println("  C++   summary: $(summaries(cpp_events))")
        exit(1)
    end
    # Compare matched pairs
    for key in matched_keys
        j = julia_map[key]
        c = cpp_map[key]
        mismatch = compare_payloads(j, c)
        if mismatch !== nothing
            fld, vj, vc = mismatch
            println("Mismatch at key=$(key) field=$(fld)")
            println("  Julia: $(vj)")
            println("  C++  : $(vc)")
            exit(2)
        end
    end
    missing_cpp = sort!(collect(setdiff(keys(julia_map), keys(cpp_map))))
    missing_julia = sort!(collect(setdiff(keys(cpp_map), keys(julia_map))))
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
        println("Logs match for overlapping events (relaxed key).")
        exit(0)
    else
        exit(3)
    end
end

main()

