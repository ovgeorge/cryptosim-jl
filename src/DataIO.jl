module DataIO

using JSON3
using Mmap
using Parsers
using StructTypes

try
    using DoubleFloats: Double64
catch
    @eval const Double64 = Float64
end

const DEFAULT_DATA_DIR = normpath(joinpath(@__DIR__, "..", "cryptopool-simulator", "download"))
const PARSER_OPTS = Parsers.Options()

export Candle, SimulationConfig, ConfigFile, CPPTrade, load_config, load_candles,
       load_trade_bundle, build_cpp_trades, initial_price_vector, DEFAULT_DATA_DIR,
       split_leg_extrema

"""
    struct Candle

Minimal representation of the raw exchange candles.
"""
struct Candle
    timestamp::Int64
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
    pair::Tuple{Int,Int}
end

"""
    split_leg_extrema(candle) -> NamedTuple

Returns the `(dir1_high, dir2_low, dir1_is_low)` triple describing which side
of the candle should be emitted first, mirroring the heuristic in
`cryptopool-simulator/get_all` (long-double math).
"""
@inline function split_leg_extrema(candle::Candle)
    open = Double64(candle.open)
    high = Double64(candle.high)
    low = Double64(candle.low)
    close = Double64(candle.close)
    lhs = abs(open - low) + abs(close - high)
    rhs = abs(open - high) + abs(close - low)
    dir1_is_low = lhs < rhs
    if dir1_is_low
        return (dir1_high = candle.low, dir2_low = candle.high, dir1_is_low = true)
    else
        return (dir1_high = candle.high, dir2_low = candle.low, dir1_is_low = false)
    end
end

"""
    struct CPPTrade

Represents a single `trade_data` entry emitted by the C++ simulator (`_tmp.*` file).
"""
struct CPPTrade
    candle_index::Int
    leg_dir::Symbol
    timestamp::Int64
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
    orig_volume::Float64
    orig_high::Float64
    orig_low::Float64
    is_last::Bool
    pair::Tuple{Int,Int}
end

Base.@kwdef struct SimulationConfig
    A::Float64
    gamma::Float64
    D::Float64
    n::Int
    mid_fee::Float64
    out_fee::Float64
    fee_gamma::Float64
    adjustment_step::Float64
    allowed_extra_profit::Float64
    ma_half_time::Int
    ext_fee::Float64
    gas_fee::Float64
    boost_rate::Float64
    log::Bool
end

struct ConfigFile
    configurations::Vector{SimulationConfig}
    datafiles::Vector{String}
    debug::Int
end

StructTypes.StructType(::Type{SimulationConfig}) = StructTypes.Struct()

"""
    load_config(path) -> ConfigFile

Reads the simulator JSON configuration (same schema as the C++ binary).
"""
function load_config(path::AbstractString)
    json = open(path, "r") do io
        JSON3.read(io)
    end
    configs = SimulationConfig[]
    if haskey(json, :configuration)
        for item in json[:configuration]
            push!(configs, _parse_config(item))
        end
    else
        error("configuration array missing in $(path)")
    end
    datafiles = if haskey(json, :datafile)
        String.(collect(json[:datafile]))
    else
        error("datafile array missing in $(path)")
    end
    length(datafiles) in (1, 2, 3) || error("datafile list must contain 1-3 entries")
    debug = haskey(json, :debug) ? Int(json[:debug]) : 0
    return ConfigFile(configs, datafiles, debug)
end

function _parse_config(obj)
    required(key, default=nothing) = begin
        if !haskey(obj, key)
            if default === nothing
                error("configuration is missing required field '$key'")
            else
                return default
            end
        end
        return obj[key]
    end
    SimulationConfig(
        Float64(required(:A)),
        Float64(required(:gamma, 0.0)),
        Float64(required(:D)),
        Int(required(:n)),
        Float64(required(:mid_fee)),
        Float64(required(:out_fee)),
        Float64(required(:fee_gamma)),
        Float64(required(:adjustment_step)),
        Float64(required(:allowed_extra_profit, 0.0)),
        Int(required(:ma_half_time, 600)),
        Float64(required(:ext_fee, 0.0)),
        Float64(required(:gas_fee, 0.0)),
        Float64(required(:boost_rate, 0.0)),
        Bool(required(:log, 0)),
    )
end

# -- Candle ingestion ---------------------------------------------------------

"""
    load_candles(name_or_path; pair=(0,1), data_dir=DEFAULT_DATA_DIR)

Streams a Binance-style `[timestamp, "open", "high", ...]` JSON file into memory.
"""
function load_candles(source::AbstractString; pair::Tuple{Int,Int}=(0, 1), data_dir::AbstractString=DEFAULT_DATA_DIR)
    path = _resolve_candle_path(source, data_dir)
    open(path, "r") do io
        mapped = Mmap.mmap(io)
        return _parse_candles(mapped, pair)
    end
end

function _resolve_candle_path(source::AbstractString, data_dir::AbstractString)
    if isfile(source)
        return source
    end
    filename = endswith(lowercase(source), ".json") ? source : string(source, ".json")
    path = joinpath(data_dir, filename)
    isfile(path) || error("candle file '$source' not found (looked in $data_dir)")
    return path
end

@inline function _is_delim(byte::UInt8)
    byte == UInt8(' ') || byte == UInt8('\n') || byte == UInt8('\r') ||
    byte == UInt8('\t') || byte == UInt8(',')
end

@inline function _skip_delims(data::Vector{UInt8}, idx::Int, len::Int)
    while idx <= len && _is_delim(data[idx])
        idx += 1
    end
    return idx
end

function _parse_number(::Type{T}, data::Vector{UInt8}, idx::Int, len::Int) where {T}
    idx = _skip_delims(data, idx, len)
    quoted = false
    if idx <= len && data[idx] == UInt8('"')
        quoted = true
        idx += 1
    end
    result = Parsers.xparse(T, data, idx, len, PARSER_OPTS)
    result.tlen > 0 || error("numeric parse failed near position $idx")
    idx += result.tlen
    idx = _skip_delims(data, idx, len)
    if quoted && idx <= len && data[idx] == UInt8('"')
        idx += 1
    end
    return result.val, idx
end

function _parse_candles(data::Vector{UInt8}, pair::Tuple{Int,Int})
    len = length(data)
    idx = findfirst(==(UInt8('[')), data)
    idx === nothing && error("invalid candle JSON payload")
    idx += 1
    candles = Vector{Candle}()
    while idx <= len
        idx = _skip_delims(data, idx, len)
        if idx > len || data[idx] == UInt8(']')
            break
        end
        if data[idx] != UInt8('[')
            idx += 1
            continue
        end
        idx += 1
        ts_raw, idx = _parse_number(Int64, data, idx, len)
        ts = ts_raw > 10_000_000_000 ? ts_raw ÷ 1000 : ts_raw
        open_, idx = _parse_number(Float64, data, idx, len)
        high, idx = _parse_number(Float64, data, idx, len)
        low, idx = _parse_number(Float64, data, idx, len)
        close, idx = _parse_number(Float64, data, idx, len)
        volume, idx = _parse_number(Float64, data, idx, len)
        idx = _advance_to_char(data, idx, len, UInt8(']')) + 1
        push!(candles, Candle(ts, open_, high, low, close, volume, pair))
    end
    return candles
end

function _advance_to_char(data::Vector{UInt8}, idx::Int, len::Int, target::UInt8)
    while idx <= len && data[idx] != target
        idx += 1
    end
    idx <= len || error("unterminated array entry while parsing candles")
    return idx
end

"""
    load_trade_bundle(cfg; data_dir=DEFAULT_DATA_DIR, trim=nothing)

Loads every datafile referenced by the config, synchronizes their time ranges,
and returns a single, globally sorted vector of candles.
"""
function load_trade_bundle(cfg::ConfigFile; data_dir::AbstractString=DEFAULT_DATA_DIR, trim::Union{Nothing,Int}=nothing)
    filenames = cfg.datafiles
    files = length(filenames)
    files in (1, 3) || error("datafile list must contain either 1 or 3 entries (got $files)")
    pair_defs = files == 1 ? [(0, 1)] : [(0, 1), (0, 2), (1, 2)]
    collections = Vector{Vector{Candle}}(undef, files)
    for idx in 1:files
        pair = pair_defs[idx]
        name = filenames[idx]
        collections[idx] = load_candles(name; pair, data_dir)
    end
    min_ts = minimum(minimum(c.timestamp for c in coll) for coll in collections)
    max_ts = maximum(maximum(c.timestamp for c in coll) for coll in collections)
    combined = Candle[]
    for coll in collections
        append!(combined, filter(c -> min_ts <= c.timestamp <= max_ts, coll))
    end
    sort!(combined, by = c -> c.timestamp)
    if trim !== nothing && length(combined) > trim
        combined = combined[end-trim+1:end]
    end
    return combined
end

function build_cpp_trades(cfg::ConfigFile; data_dir::AbstractString=DEFAULT_DATA_DIR, trim::Union{Nothing,Int}=nothing)
    filenames = cfg.datafiles
    files = length(filenames)
    files in (1, 3) || error("datafile list must contain either 1 or 3 entries (got $files)")
    pair_defs = files == 1 ? [(0, 1)] : [(0, 1), (0, 2), (1, 2)]
    collections = Vector{Vector{Candle}}(undef, files)
    for idx in 1:files
        pair = pair_defs[idx]
        name = filenames[idx]
        collections[idx] = load_candles(name; pair, data_dir)
    end
    min_ts = minimum(minimum(c.timestamp for c in coll) for coll in collections)
    max_ts = maximum(maximum(c.timestamp for c in coll) for coll in collections)
    trades = CPPTrade[]
    sizehint!(trades, sum(length(coll) for coll in collections) * 2)
    candle_idx = 0
    for (pair, coll) in zip(pair_defs, collections)
        for candle in coll
            ts = candle.timestamp
            (ts >= min_ts && ts <= max_ts) || continue
            append_cpp_trades!(trades, candle_idx, candle, pair)
            candle_idx += 2
        end
    end
    sort!(trades, by = tr -> tr.timestamp)
    if trim !== nothing && trim > 0 && length(trades) > trim
        trades = copy(trades[end - trim + 1:end])
    end
    return trades
end

function append_cpp_trades!(buf::Vector{CPPTrade}, idx::Int, candle::Candle, pair::Tuple{Int,Int})
    offset = (pair[1] + pair[2]) * 10
    lower_ts = candle.timestamp - offset + 5
    upper_ts = candle.timestamp + offset - 5
    vol_half = candle.volume / 2
    extrema = split_leg_extrema(candle)
    min_high = extrema.dir1_high
    max_low = extrema.dir2_low
    first_dir = extrema.dir1_is_low ? :dir2 : :dir1
    second_dir = extrema.dir1_is_low ? :dir1 : :dir2
    push!(buf, CPPTrade(
        idx,
        first_dir,
        lower_ts,
        candle.open,
        min_high,
        min_high,
        min_high,
        vol_half,
        candle.volume,
        candle.high,
        candle.low,
        false,
        pair,
    ))
    push!(buf, CPPTrade(
        idx + 1,
        second_dir,
        upper_ts,
        max_low,
        max_low,
        max_low,
        candle.close,
        vol_half,
        candle.volume,
        candle.high,
        candle.low,
        true,
        pair,
    ))
end

"""
    initial_price_vector(n, candles) -> Vector{Float64}

Replicates `get_price_vector` from the C++ simulator to seed Trader price state.
"""
function initial_price_vector(n::Int, candles::AbstractVector{Candle})
    n in (2, 3) || error("initial_price_vector currently supports n=2 or n=3 (got $n)")
    prices = ones(Float64, n)
    seen = falses(n)
    seen[1] = true # asset 0 is the 1.0 base
    for candle in candles
        a, b = candle.pair
        if a == 0 && b < n
            prices[b + 1] = candle.close
            seen[b + 1] = true
        elseif b == 0 && a < n
            prices[a + 1] = 1 / candle.close
            seen[a + 1] = true
        end
        all(seen) && return prices
    end
    return prices
end

end # module
