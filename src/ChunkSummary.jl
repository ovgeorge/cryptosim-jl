module ChunkSummary

using Printf

using ..Metrics

export MetricSnapshot, ChunkSummary, build_summary, to_json_dict, from_json_dict,
       QuantileStats, build_stats, quantile, format_quantile_table, format_quantile_markdown,
       render_markdown_report, METRIC_NAMES, DEFAULT_QUANTILES

const METRIC_NAMES = (:volume, :slippage, :liquidity_density, :apy)
const METRIC_COUNT = length(METRIC_NAMES)
const DEFAULT_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

struct MetricSnapshot{T}
    julia::T
    cpp::T
    rel_err::T
end

const MetricTuple{T} = NamedTuple{METRIC_NAMES, NTuple{METRIC_COUNT, MetricSnapshot{T}}}

struct ChunkSummary{T,Meta<:NamedTuple}
    chunk_id::String
    metrics::MetricTuple{T}
    log_ok::Bool
    log_diff::String
    metadata::Meta
end

ChunkSummary(chunk_id::String, metrics::MetricTuple{T}, log_ok::Bool, log_diff::AbstractString,
             metadata::Meta) where {T,Meta<:NamedTuple} =
    ChunkSummary{T,Meta}(chunk_id, metrics, Bool(log_ok), String(log_diff), metadata)

ChunkSummary(chunk_id::String, metrics::MetricTuple{T}, log_ok::Bool, log_diff::AbstractString) where {T} =
    ChunkSummary{T,NamedTuple{(),Tuple{}}}(chunk_id, metrics, Bool(log_ok), String(log_diff), NamedTuple())

@inline function _relative_error(actual::T, expected::T) where {T<:AbstractFloat}
    expected == zero(T) && return zero(T)
    return (actual - expected) / expected
end

@inline function _normalize_metadata(meta)
    if meta isa NamedTuple
        return meta
    elseif meta isa Base.AbstractDict
        return NamedTuple(Symbol(k) => v for (k, v) in meta)
    elseif meta === nothing
        return NamedTuple()
    else
        return NamedTuple()
    end
end

@inline _meta_get(meta::NamedTuple, key::Symbol, default=nothing) =
    haskey(meta, key) ? getfield(meta, key) : default

function build_summary(chunk_id::AbstractString,
                       measured::NamedTuple,
                       expected::NamedTuple;
                       log_ok::Bool,
                       log_diff::AbstractString="",
                       metadata::NamedTuple=NamedTuple())
    T = typeof(getfield(measured, METRIC_NAMES[1]))
    snapshots = ntuple(i -> begin
        name = METRIC_NAMES[i]
        actual = convert(T, getfield(measured, name))
        ref = convert(T, getfield(expected, name))
        MetricSnapshot{T}(actual, ref, _relative_error(actual, ref))
    end, METRIC_COUNT)
    metric_tuple = NamedTuple{METRIC_NAMES}(snapshots)
    meta = _normalize_metadata(metadata)
    return ChunkSummary(String(chunk_id), metric_tuple, log_ok, log_diff, meta)
end

function _metadata_to_dict(meta::NamedTuple)
    isempty(meta) && return Dict{String,Any}()
    dict = Dict{String,Any}()
    for (k, v) in pairs(meta)
        dict[string(k)] = v
    end
    return dict
end

function to_json_dict(summary::ChunkSummary)
    metrics = Dict{String,Any}()
    for name in METRIC_NAMES
        snap = getfield(summary.metrics, name)
        metrics[string(name)] = Dict(
            "julia" => snap.julia,
            "cpp" => snap.cpp,
            "rel_err" => snap.rel_err,
        )
    end
    dict = Dict(
        "chunk" => summary.chunk_id,
        "metrics" => metrics,
        "log_ok" => summary.log_ok,
        "log_diff" => summary.log_diff,
    )
    meta = _metadata_to_dict(summary.metadata)
    if !isempty(meta)
        dict["metadata"] = meta
    end
    return dict
end

function _lookup_metric(metrics_obj, name::Symbol)
    key = string(name)
    snap = metrics_obj[key]
    return MetricSnapshot{Float64}(
        Float64(snap["julia"]),
        Float64(snap["cpp"]),
        Float64(snap["rel_err"]),
    )
end

function from_json_dict(obj)
    metrics_obj = obj["metrics"]
    snaps = ntuple(i -> _lookup_metric(metrics_obj, METRIC_NAMES[i]), METRIC_COUNT)
    metric_tuple = NamedTuple{METRIC_NAMES}(snaps)
    log_diff = get(obj, "log_diff", "")
    metadata_obj = get(obj, "metadata", Dict{String,Any}())
    meta_nt = metadata_obj isa NamedTuple ? metadata_obj : NamedTuple(Symbol(k) => v for (k, v) in metadata_obj)
    return ChunkSummary(
        String(obj["chunk"]),
        metric_tuple,
        Bool(obj["log_ok"]),
        String(log_diff),
        meta_nt,
    )
end

struct QuantileStats
    metrics::Vector{Symbol}
    quantiles::Vector{Float64}
    sorted_values::Dict{Symbol,Vector{Float64}}
    worst::Dict{Symbol,NamedTuple{(:chunk, :rel, :abs),Tuple{String,Float64,Float64}}}
    log_matches::Int
    total::Int
end

function _collect_values(summaries, metric::Symbol)
    vals = Float64[]
    for summary in summaries
        snap = getfield(summary.metrics, metric)
        push!(vals, abs(Float64(snap.rel_err)))
    end
    sort!(vals)
    return vals
end

@inline function _quantile_from_sorted(vals::Vector{Float64}, q::Float64)
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

function build_stats(summaries::AbstractVector{<:ChunkSummary};
                     metrics=METRIC_NAMES,
                     quantiles=DEFAULT_QUANTILES)
    metrics_vec = collect(metrics)
    quant_vec = Float64.(collect(quantiles))
    values = Dict(name => Float64[] for name in metrics_vec)
    worst = Dict(name => (chunk = "", rel = 0.0, abs = -Inf) for name in metrics_vec)
    log_matches = 0
    for summary in summaries
        summary.log_ok && (log_matches += 1)
        for name in metrics_vec
            snap = getfield(summary.metrics, name)
            rel = Float64(snap.rel_err)
            abs_rel = abs(rel)
            push!(values[name], abs_rel)
            if abs_rel > worst[name].abs
                worst[name] = (chunk = summary.chunk_id, rel = rel, abs = abs_rel)
            end
        end
    end
    for name in metrics_vec
        sort!(values[name])
    end
    return QuantileStats(metrics_vec, quant_vec, values, worst, log_matches, length(summaries))
end

function quantile(stats::QuantileStats, metric::Symbol, q::Float64)
    vals = stats.sorted_values[metric]
    return _quantile_from_sorted(vals, q)
end

function format_quantile_table(stats::QuantileStats)
    io = IOBuffer()
    println(io, "\nRelative-error quantiles (abs rel err) across $(stats.total) chunks:\n")
    labels = ["metric"; ["q$(Int(q * 1000) / 10)" for q in stats.quantiles]; ["max (chunk, rel)"]]
    widths = fill(14, length(labels))
    println(io, join(rpad.(labels, widths), ""))
    for name in stats.metrics
        print(io, rpad(String(name), 14))
        for q in stats.quantiles
            val = quantile(stats, name, q)
            print(io, rpad(@sprintf("%.6e", val), 14))
        end
        worst = stats.worst[name]
        print(io, @sprintf("%.6e", worst.abs))
        print(io, " ($(worst.chunk), rel=")
        print(io, @sprintf("%+.6e", worst.rel))
        print(io, ")")
        println(io)
    end
    return String(take!(io))
end

function format_quantile_markdown(stats::QuantileStats)
    io = IOBuffer()
    header = ["metric"; ["q$(Int(q * 1000) / 10)" for q in stats.quantiles]; ["max (chunk, rel)"]]
    println(io, "| " * join(header, " | ") * " |")
    println(io, "| " * join(fill("---", length(header)), " | ") * " |")
    for name in stats.metrics
        row = Any[]
        push!(row, String(name))
        for q in stats.quantiles
            push!(row, @sprintf("%.6e", quantile(stats, name, q)))
        end
        worst = stats.worst[name]
        push!(row, @sprintf("%.6e", worst.abs) * " (`$(worst.chunk)`, " * @sprintf("rel=%+.6e", worst.rel) * ")")
        println(io, "| " * join(row, " | ") * " |")
    end
    return String(take!(io))
end

function render_markdown_report(stats::QuantileStats;
                                title::AbstractString="Parity Snapshot",
                                metadata::NamedTuple=NamedTuple())
    buf = IOBuffer()
    println(buf, "# $(title)\n")
    dataset_name = _meta_get(metadata, :dataset_name)
    dataset_path = _meta_get(metadata, :dataset_path)
    dataset_sha = _meta_get(metadata, :dataset_sha)
    chunk_root = _meta_get(metadata, :chunk_root)
    runner = _meta_get(metadata, :runner)
    extra_notes = _meta_get(metadata, :notes, String[])

    if dataset_name !== nothing || dataset_path !== nothing
        info = dataset_name !== nothing ? String(dataset_name) : String(dataset_path)
        if dataset_path !== nothing && dataset_name !== nothing
            info *= " ($(dataset_path))"
        end
        println(buf, "- **Dataset**: $(info)")
    end
    if dataset_sha !== nothing && !isempty(String(dataset_sha))
        println(buf, "- **Dataset sha256**: $(dataset_sha)")
    end
    if chunk_root !== nothing
        println(buf, "- **Chunk root**: $(chunk_root)")
    end
    println(buf, "- **Chunks analyzed**: $(stats.total)")
    println(buf, "- **Chunk logs matching C++**: $(stats.log_matches)/$(stats.total)")
    if runner !== nothing
        println(buf, "- **Runner**: $(runner)")
    end

    if extra_notes isa AbstractVector && !isempty(extra_notes)
        println(buf, "\n> **Notes**")
        for note in extra_notes
            println(buf, "> $(note)")
        end
    end

    println(buf, "\n" * format_quantile_markdown(stats))
    return String(take!(buf))
end

end # module
