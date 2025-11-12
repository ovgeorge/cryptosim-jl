module Preprocessing

using JSON3

using ..DataIO: Candle, CPPTrade, split_leg_extrema

export SplitTrade, split_candles, adapt_trades

"""
    struct SplitTrade

Represents the bull/bear legs emitted per candle, including cached high/low
and ordering metadata so the simulator can reproduce C++ control flow.
"""
struct SplitTrade
    candle_index::Int
    leg_dir::Symbol
    timestamp::Int64
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
    orig_volume::Float64
    cap_volume::Float64
    pair::Tuple{Int,Int}
    is_last::Bool
    orig_high::Float64
    orig_low::Float64
end

"""
    split_candles(candles; pair_ts_offset_fn = default_offset) -> Vector{SplitTrade}

Recreates the exact `trade_min`/`trade_max` splitter from the legacy C++ code,
emitting at most two split trades per raw candle with deterministic timestamps.
"""
function split_candles(candles::AbstractVector{Candle})
    splits = Vector{SplitTrade}()
    sizehint!(splits, length(candles) * 2)
    idx = 0
    for candle in candles
        append_splits!(splits, idx, candle)
        idx += 2
    end
    sort!(splits, by = s -> s.timestamp)
    return splits
end

# -- internal helpers ---------------------------------------------------------

@inline function pair_offset(pair::Tuple{Int,Int})
    return (pair[1] + pair[2]) * 10
end

function append_splits!(buf::Vector{SplitTrade}, idx::Int, candle::Candle)
    offset = pair_offset(candle.pair)
    lower_ts = candle.timestamp - offset + 5
    upper_ts = candle.timestamp + offset - 5
    (min_leg, max_leg, first_dir, second_dir) = build_split_legs(candle)
    push!(buf, SplitTrade(idx, first_dir, lower_ts,
                          min_leg.open, min_leg.high, min_leg.low, min_leg.close,
                          min_leg.volume, candle.volume, min_leg.volume,
                          candle.pair, false, candle.high, candle.low))
    max_volume = get(max_leg, :volume, candle.volume / 2)
    push!(buf, SplitTrade(idx + 1, second_dir, upper_ts,
                          max_leg.open, max_leg.high, max_leg.low, max_leg.close,
                          max_volume, candle.volume, max_volume,
                          candle.pair, true, candle.high, candle.low))
end

"""
    build_split_legs(candle) -> (min_leg, max_leg)

Splits the candle according to the same heuristic used in C++:
the leg covering the low price goes first when the total deviation to low
is smaller than the deviation to high.
"""
function build_split_legs(candle::Candle)
    extrema = split_leg_extrema(candle)
    vol_half = candle.volume / 2
    min_leg = (; open = candle.open,
               high = extrema.dir1_high,
               low = extrema.dir1_high,
               close = extrema.dir1_high,
               volume = vol_half)
    dir2_open = extrema.dir1_is_low ? candle.high : candle.low
    max_leg = (; open = dir2_open,
               high = dir2_open,
               low = dir2_open,
               close = candle.close,
               volume = vol_half)
    first_dir = extrema.dir1_is_low ? :dir2 : :dir1
    second_dir = extrema.dir1_is_low ? :dir1 : :dir2
    return (min_leg, max_leg, first_dir, second_dir)
end

# -- legacy trade adaptors ----------------------------------------------------

"""
    adapt_trades(trades) -> Vector{SplitTrade}

Converts legacy trade representations into canonical `SplitTrade`s so the
simulator consumes a single struct regardless of the upstream format.
"""
adapt_trades(trades::AbstractVector{SplitTrade}) = trades

function adapt_trades(trades::AbstractVector{CPPTrade})
    splits = Vector{SplitTrade}(undef, length(trades))
    # Optional split debug logging for first N entries
    do_log = get(ENV, "JULIA_SPLIT_DEBUG", "") != ""
    limit = try
        parse(Int, get(ENV, "JULIA_SPLIT_LIMIT", "100"))
    catch
        100
    end
    for (idx, trade) in enumerate(trades)
        splits[idx] = SplitTrade(
            trade.candle_index,
            trade.leg_dir,
            trade.timestamp,
            trade.open,
            trade.high,
            trade.low,
            trade.close,
            trade.volume,
            trade.orig_volume,
            trade.volume,
            trade.pair,
            trade.is_last,
            trade.orig_high,
            trade.orig_low,
        )
        if do_log && idx <= limit
            # Emit SPLIT for analysis
            println("SPLIT ", JSON3.write(Dict(
                "candle_index" => trade.candle_index,
                "leg_dir" => String(trade.leg_dir),
                "timestamp" => trade.timestamp,
                "pair" => (trade.pair[1], trade.pair[2]),
                "open" => trade.open,
                "high" => trade.high,
                "low" => trade.low,
                "close" => trade.close,
                "volume" => trade.volume,
                "orig_volume" => trade.orig_volume,
                "orig_high" => trade.orig_high,
                "orig_low" => trade.orig_low,
                "is_last" => trade.is_last,
            )))
        end
    end
    sort!(splits, by = s -> s.timestamp)
    return splits
end

end # module
