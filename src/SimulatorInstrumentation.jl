module Instrumentation

using JSON3
using StructTypes

using ...Preprocessing: SplitTrade
using ...DomainTypes: TradePair, LegStage, leg_stage_label, DIR1, DIR2
import ..Trader

export TradeLogger, DebugOptions, default_debug_options, noop_debug_options, with_logger,
       EventContext, event_context, TraderSnapshot, trader_snapshot,
       StepPayload, LegPayload, TweakPayload, record_event!,
       log_step_event, log_leg_event, log_tweak_event, log_preleg_event,
       maybe_step_probe_context, emit_step_probe, trace_step_iter, trace_step_result, trace_dir2_execution

struct TradeLogger
    enabled::Bool
    sink::Union{Nothing,IO}
    buffer::Union{Nothing,Vector{Any}}
end

TradeLogger() = TradeLogger(false, nothing, nothing)
TradeLogger(buffer::Vector{Any}) = TradeLogger(true, nothing, buffer)

struct EventContext
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
end

@inline event_context(config::Int, trade::SplitTrade) =
    EventContext(config, trade.candle_index, trade.timestamp, trade.pair)

struct StepTraceConfig
    enabled::Bool
    target::Symbol
    path::Union{Nothing,String}
end

StepTraceConfig() = StepTraceConfig(false, :stderr, nothing)

struct ProbeConfig
    index::Union{Nothing,Int}
    file::String
end

ProbeConfig() = ProbeConfig(nothing, "")

struct DebugOptions
    logger::TradeLogger
    step_trace::StepTraceConfig
    probe::ProbeConfig
end

DebugOptions(; logger::TradeLogger=TradeLogger(),
              step_trace::StepTraceConfig=StepTraceConfig(),
              probe::ProbeConfig=ProbeConfig()) = DebugOptions(logger, step_trace, probe)

const STEP_TRACE_ENV = "JULIA_STEP_TRACE"
const PROBE_ENV = "JULIA_STEP_PROBE"
const PROBE_FILE_ENV = "JULIA_STEP_PROBE_FILE"
const LOGGER_ENV = "JULIA_TRADE_DEBUG"

@inline function trade_logger_from_env()
    env = get(ENV, LOGGER_ENV, "")
    if isempty(env)
        return TradeLogger()
    else
        return TradeLogger(true, stdout, nothing)
    end
end

@inline function step_trace_config_from_env()
    raw = get(ENV, STEP_TRACE_ENV, "")
    if isempty(raw)
        return StepTraceConfig()
    elseif raw == "stdout"
        return StepTraceConfig(true, :stdout, nothing)
    elseif raw == "stderr"
        return StepTraceConfig(true, :stderr, nothing)
    else
        return StepTraceConfig(true, :file, raw)
    end
end

@inline function probe_config_from_env()
    idx_raw = get(ENV, PROBE_ENV, "")
    idx = isempty(idx_raw) ? nothing : tryparse(Int, idx_raw)
    file = get(ENV, PROBE_FILE_ENV, "")
    return ProbeConfig(idx, file)
end

function default_debug_options(; logger::Union{Nothing,TradeLogger}=nothing)
    log_obj = logger === nothing ? trade_logger_from_env() : logger
    return DebugOptions(log_obj, step_trace_config_from_env(), probe_config_from_env())
end

noop_debug_options() = DebugOptions()

with_logger(debug::DebugOptions, logger::TradeLogger) =
    DebugOptions(logger, debug.step_trace, debug.probe)

abstract type TradeEvent end

struct TraderSnapshot{T}
    reserves::Vector{T}
    target_prices::Vector{T}
    price_oracle::Vector{T}
    last_price::Vector{T}
    xcp_profit::T
    xcp_profit_real::T
    not_adjusted::Bool
    is_light::Bool
end

StructTypes.StructType(::Type{TraderSnapshot}) = StructTypes.Struct()

@inline function trader_snapshot(trader::Trader{T}) where {T}
    return TraderSnapshot{T}(
        copy(trader.curve.x),
        copy(trader.curve.p),
        copy(trader.tweak.price_oracle),
        copy(trader.tweak.last_price),
        trader.profit.xcp_profit,
        trader.profit.xcp_profit_real,
        trader.tweak.not_adjusted,
        trader.tweak.is_light,
    )
end

struct StepEvent{T} <: TradeEvent
    event::String
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
    stage::String
    price_before::T
    limit_low::T
    limit_high::T
    dx::T
    volume_before::T
    ext_vol::T
    gas_fee::T
    pool_n::Int
end

struct PreLegEvent{T} <: TradeEvent
    event::String
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
    stage::String
    price_before::T
    limit_low::T
    limit_high::T
    volume_before::T
    ext_vol::T
    pool_n::Int
    reserves::Vector{T}
    target_prices::Vector{T}
    price_oracle::Vector{T}
    last_price::Vector{T}
end

struct LegEvent{T} <: TradeEvent
    event::String
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
    stage::String
    price_before::T
    price_after::T
    max_price::T
    min_price::T
    last_quote::T
    dx::T
    dy::T
    volume_delta::T
    volume_total::T
    ext_vol::T
    trades::Int
    reserves::Vector{T}
    target_prices::Vector{T}
    price_oracle::Vector{T}
    last_price::Vector{T}
    xcp_profit::T
    xcp_profit_real::T
    not_adjusted::Bool
    is_light::Bool
end

struct TweakEvent{T} <: TradeEvent
    event::String
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
    mode::String
    norm::T
    price_mid::T
    high::T
    low::T
    price_oracle::Vector{T}
    target_prices::Vector{T}
    last_price::Vector{T}
    xcp_profit::T
    xcp_profit_real::T
    allowed_extra_profit::T
    not_adjusted::Bool
    is_light::Bool
    ma_half_time::Int
end

struct StepPayload{T}
    stage::String
    price_before::T
    limit_low::T
    limit_high::T
    dx::T
    volume_before::T
    ext_vol::T
    gas_fee::T
    pool_n::Int
end

struct LegPayload{T}
    stage::String
    price_before::T
    price_after::T
    max_price::T
    min_price::T
    last_quote::T
    dx::T
    dy::T
    volume_delta::T
    volume_total::T
    ext_vol::T
    trades::Int
    snapshot::TraderSnapshot{T}
end

struct TweakPayload{T}
    mode::String
    norm::T
    price_mid::T
    high::T
    low::T
    snapshot::TraderSnapshot{T}
    allowed_extra_profit::T
    ma_half_time::Int
end

StructTypes.StructType(::Type{StepEvent}) = StructTypes.Struct()
StructTypes.StructType(::Type{LegEvent}) = StructTypes.Struct()
StructTypes.StructType(::Type{TweakEvent}) = StructTypes.Struct()
StructTypes.StructType(::Type{PreLegEvent}) = StructTypes.Struct()

struct PreLegLimitEvent{T} <: TradeEvent
    event::String
    config::Int
    candle_index::Int
    timestamp::Int64
    pair::TradePair
    stage::String
    price_before::T
    limit_low::T
    limit_high::T
    volume_before::T
    ext_vol::T
    reason::String
end

StructTypes.StructType(::Type{PreLegLimitEvent}) = StructTypes.Struct()

@inline function emit_event(logger::TradeLogger, payload::TradeEvent)
    logger.enabled || return
    if logger.buffer !== nothing
        push!(logger.buffer, payload)
    end
    if logger.sink !== nothing
        print(logger.sink, "JULIADB ")
        JSON3.write(logger.sink, payload)
        println(logger.sink)
    end
end

@inline function record_event!(logger::TradeLogger, ctx::EventContext, payload::StepPayload)
    event = StepEvent(
        "STEP", ctx.config, ctx.candle_index, ctx.timestamp, ctx.pair, payload.stage,
        payload.price_before, payload.limit_low, payload.limit_high, payload.dx,
        payload.volume_before, payload.ext_vol, payload.gas_fee, payload.pool_n,
    )
    emit_event(logger, event)
end

@inline function record_event!(logger::TradeLogger, ctx::EventContext, stage::String,
                               price_before, limit_low, limit_high, volume_before,
                               ext_vol, pool_n, snap::TraderSnapshot)
    event = PreLegEvent(
        "PRELEG", ctx.config, ctx.candle_index, ctx.timestamp, ctx.pair, stage,
        price_before, limit_low, limit_high, volume_before, ext_vol, pool_n,
        snap.reserves, snap.target_prices, snap.price_oracle, snap.last_price,
    )
    emit_event(logger, event)
end

@inline function log_preleg_limit_event(logger::TradeLogger, config_id::Int, candle_index::Int, trade::SplitTrade,
                                        stage, price_before, limit_low, limit_high,
                                        volume_before, ext_vol, reason::AbstractString)
    ctx = EventContext(config_id, candle_index, trade.timestamp, trade.pair)
    event = PreLegLimitEvent(
        "PRELEG_LIMIT", ctx.config, ctx.candle_index, ctx.timestamp, ctx.pair, stage_label(stage),
        price_before, limit_low, limit_high, volume_before, ext_vol, String(reason),
    )
    emit_event(logger, event)
end

@inline function record_event!(logger::TradeLogger, ctx::EventContext, payload::LegPayload)
    snap = payload.snapshot
    event = LegEvent(
        "LEG", ctx.config, ctx.candle_index, ctx.timestamp, ctx.pair, payload.stage,
        payload.price_before, payload.price_after, payload.max_price, payload.min_price,
        payload.last_quote, payload.dx, payload.dy, payload.volume_delta, payload.volume_total,
        payload.ext_vol, payload.trades, snap.reserves, snap.target_prices, snap.price_oracle,
        snap.last_price, snap.xcp_profit, snap.xcp_profit_real, snap.not_adjusted, snap.is_light,
    )
    emit_event(logger, event)
end

@inline function record_event!(logger::TradeLogger, ctx::EventContext, payload::TweakPayload)
    snap = payload.snapshot
    event = TweakEvent(
        "TWEAK", ctx.config, ctx.candle_index, ctx.timestamp, ctx.pair,
        payload.mode, payload.norm, payload.price_mid, payload.high, payload.low,
        snap.price_oracle, snap.target_prices, snap.last_price,
        snap.xcp_profit, snap.xcp_profit_real, payload.allowed_extra_profit,
        snap.not_adjusted, snap.is_light, payload.ma_half_time,
    )
    emit_event(logger, event)
end

@inline function log_step_event(logger::TradeLogger, config_id::Int, candle_index::Int,
                                trade::SplitTrade, stage,
                                price_before, limit_low, limit_high, dx,
                                volume_before, ext_vol, gas_fee, pool_n)
    ctx = EventContext(config_id, candle_index, trade.timestamp, trade.pair)
    payload = StepPayload(
        stage_label(stage),
        price_before,
        limit_low,
        limit_high,
        dx,
        volume_before,
        ext_vol,
        gas_fee,
        pool_n,
    )
    record_event!(logger, ctx, payload)
end

@inline function log_leg_event(logger::TradeLogger, config_id::Int, candle_index::Int, trade::SplitTrade,
                               stage, price_before, price_after,
                               max_price, min_price, last_quote, dx, dy,
                               volume_delta, volume_total, ext_vol, trades, trader)
    ctx = EventContext(config_id, candle_index, trade.timestamp, trade.pair)
    payload = LegPayload(
        stage_label(stage),
        price_before,
        price_after,
        max_price,
        min_price,
        last_quote,
        dx,
        dy,
        volume_delta,
        volume_total,
        ext_vol,
        trades,
        trader_snapshot(trader),
    )
    record_event!(logger, ctx, payload)
end

@inline function log_tweak_event(logger::TradeLogger, config_id::Int, candle_index::Int,
                                 trade::SplitTrade, midpoint, high, low, trader)
    ctx = EventContext(config_id, candle_index, trade.timestamp, trade.pair)
    payload = TweakPayload(
        String(trader.tweak.last_tweak_mode),
        trader.tweak.last_tweak_norm,
        midpoint,
        high,
        low,
        trader_snapshot(trader),
        trader.tweak.allowed_extra_profit,
        trader.tweak.ma_half_time,
    )
    record_event!(logger, ctx, payload)
end

@inline stage_label(stage::LegStage) = leg_stage_label(stage)
@inline stage_label(stage::Symbol) = uppercase(String(stage))
@inline stage_label(stage::AbstractString) = String(stage)

@inline function log_preleg_event(logger::TradeLogger, config_id::Int, candle_index::Int, trade::SplitTrade,
                                  stage, price_before, limit_low, limit_high,
                                  volume_before, ext_vol, pool_n, trader)
    ctx = EventContext(config_id, candle_index, trade.timestamp, trade.pair)
    snap = trader_snapshot(trader)
    record_event!(logger, ctx, stage_label(stage), price_before, limit_low, limit_high, volume_before, ext_vol, pool_n, snap)
end

@inline function maybe_step_probe_context(debug::DebugOptions, candle_idx::Int, timestamp::Int,
                                        pair::TradePair, stage)
    idx = debug.probe.index
    (idx !== nothing && candle_idx == idx) || return nothing
    stage_name = stage isa LegStage ? lowercase(leg_stage_label(stage)) : lowercase(String(stage))
    return (; candle = candle_idx, timestamp = timestamp, pair = pair, stage = stage_name)
end

function emit_step_probe(debug::DebugOptions, ctx::NamedTuple, trader::Trader, from_asset::Int, to_asset::Int, dx, dy,
                         price, gas, p_min, p_max, vol, ext_vol)
    payload = merge(ctx, (
        from_asset = from_asset - 1,
        to_asset = to_asset - 1,
        dx = dx,
        dy = dy,
        price = price,
        gas = gas,
        p_min = p_min,
        p_max = p_max,
        volume = vol,
        ext_vol = ext_vol,
        balances = copy(trader.curve.x),
        price_scale = copy(trader.curve.p),
        price_oracle = copy(trader.tweak.price_oracle),
        last_price = copy(trader.tweak.last_price),
        dx_seed = trader.profit.dx,
        xcp = trader.profit.xcp,
        xcp_profit = trader.profit.xcp_profit,
        xcp_profit_real = trader.profit.xcp_profit_real,
        mid_fee = trader.fees.mid_fee,
        out_fee = trader.fees.out_fee,
        fee_gamma = trader.fees.fee_gamma,
        adjustment_step = trader.tweak.adjustment_step,
        allowed_extra_profit = trader.tweak.allowed_extra_profit,
        gas_fee = trader.fees.gas_fee,
        ext_fee = trader.fees.ext_fee,
    ))
    cfg = debug.probe
    if isempty(cfg.file)
        print(stderr, "STEPPROBE ")
        JSON3.write(stderr, payload)
        println(stderr)
    else
        open(cfg.file, "a") do io
            print(io, "STEPPROBE ")
            JSON3.write(io, payload)
            println(io)
        end
    end
end

@inline function trace_step_iter(debug::DebugOptions, stage, phase::Symbol, ctx::NamedTuple)
    cfg = debug.step_trace
    cfg.enabled || return
    stage_name = stage isa LegStage ? lowercase(leg_stage_label(stage)) : String(stage)
    write_step_trace(cfg, merge((event = "STEP_ITER", stage = stage_name, phase = String(phase)), ctx))
end

@inline function trace_step_result(debug::DebugOptions, stage, ctx::NamedTuple)
    cfg = debug.step_trace
    cfg.enabled || return
    stage_name = stage isa LegStage ? lowercase(leg_stage_label(stage)) : String(stage)
    write_step_trace(cfg, merge((event = "STEP_RESULT", stage = stage_name), ctx))
end

@inline function trace_dir2_execution(debug::DebugOptions, ctx::NamedTuple)
    cfg = debug.step_trace
    cfg.enabled || return
    write_step_trace(cfg, merge((event = "DIR2_EXEC", stage = "dir2"), ctx))
end

function write_step_trace(cfg::StepTraceConfig, payload)
    io, closer = trace_io(cfg)
    try
        print(io, "STEPTRACE ")
        try
            JSON3.write(io, payload)
        catch err
            # Fallback: avoid JSON error on NaN/Inf by writing a minimal record
            JSON3.write(io, Dict("event" => get(payload, :event, "STEPTRACE"),
                                  "error" => "nonfinite_value"))
        end
        println(io)
    finally
        closer === nothing || closer()
    end
end

@inline function trace_io(cfg::StepTraceConfig)
    if cfg.target == :stdout
        return stdout, nothing
    elseif cfg.target == :stderr
        return stderr, nothing
    else
        io = open(String(cfg.path), "a")
        return io, () -> close(io)
    end
end

end # module
