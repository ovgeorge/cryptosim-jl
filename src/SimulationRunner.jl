module SimulationRunner

using ..SimulatorCore: Trader
using ..SimulatorLogger
using ..SimulatorMath: apply_boost!, price2, price3
using ...Metrics
using ...Preprocessing: SplitTrade
import ..step_for_price
import ..execute_trade!
import ..tweak_price!

const Logger = SimulatorLogger

export SimulationShell, run_split_trades!

mutable struct SimulationShell{T}
    trader::Trader{T}
    metrics::Metrics.MetricAccumulator{T}
    logger::Logger.LoggerAdapter
end

shell_trader(shell::SimulationShell) = shell.trader
shell_metrics(shell::SimulationShell) = shell.metrics
shell_logger(shell::SimulationShell) = shell.logger

@inline asset_index(id::Int) = id + 1

function current_price(trader::Trader{T}, pair::Tuple{Int,Int}) where {T<:AbstractFloat}
    i = asset_index(pair[1])
    j = asset_index(pair[2])
    if trader.curve.n == 3
        return price3(trader.curve, i, j, trader.profit.dx)
    else
        return price2(trader.curve, i, j, trader.profit.dx)
    end
end

mutable struct LegConfig{T}
    stage::Symbol
    p_min::T
    p_max::T
    from_asset::Int
    to_asset::Int
    volume_index::Int
    metric_index::Int
    max_env::T
    min_env::T
    metric_uses_dy::Bool
    trace_dir2::Bool
end

function record_leg_metrics!(shell::SimulationShell{T}, amount::T, price_before::T,
                             price_after::T, denom::T, dt::T) where {T<:AbstractFloat}
    amount <= 0 && return
    price_before == price_after && return
    denom == 0 && return
    volume_delta = amount / denom
    Metrics.push_volume!(shell.metrics, volume_delta)
    shell.trader.metrics_state.volume += volume_delta
    slip_denom = T(2) * abs(price_before - price_after) * denom
    slip_denom == 0 && return
    slip = (amount * (price_before + price_after)) / slip_denom
    if slip > 0 && dt > 0
        Metrics.push_slippage!(shell.metrics, dt, slip)
    end
end

function maybe_execute_leg!(shell::SimulationShell{T}, candle_id::Int, trade::SplitTrade,
                            cfg::LegConfig{T}, price_before::T,
                            vol::T, ext_vol::T, trade_dt::T,
                            last_quote::T, primary_quote::T,
                            ctr::Int) where {T<:AbstractFloat}
    trader = shell.trader
    stage_str = cfg.stage === :dir1 ? "DIR1" : "DIR2"
    lim_low = cfg.stage === :dir1 ? zero(T) : cfg.p_min
    lim_high = cfg.stage === :dir1 ? cfg.p_max : zero(T)
    Logger.log_preleg!(shell.logger, 0, trade.candle_index, trade, stage_str,
                       price_before, lim_low, lim_high, vol, ext_vol, trader.curve.n, trader)
    limit_valid = cfg.stage === :dir1 ?
        (cfg.p_max != zero(T) && cfg.p_max > price_before) :
        (cfg.p_min != zero(T) && cfg.p_min < price_before)
    if !limit_valid
        Logger.log_preleg_limit!(shell.logger, 0, trade.candle_index, trade, stage_str,
                                 price_before, lim_low, lim_high, vol, ext_vol,
                                 cfg.stage === :dir1 ? "price_before >= p_max" : "price_before <= p_min")
        return vol, last_quote, primary_quote, ctr
    end
    probe_ctx = Logger.maybe_step_probe(shell.logger, candle_id, trade.timestamp, trade.pair, cfg.stage)
    step = step_for_price(trader, cfg.p_min, cfg.p_max, trade.pair,
                          vol, ext_vol, shell.logger; stage=cfg.stage, probe=probe_ctx)
    Logger.log_step!(shell.logger, 0, trade.candle_index, trade, stage_str,
                     price_before, lim_low, lim_high, step,
                     vol, ext_vol, trader.fees.gas_fee, trader.curve.n)
    step > 0 || return vol, last_quote, primary_quote, ctr
    dy = execute_trade!(trader, step, cfg.from_asset, cfg.to_asset)
    dy > 0 || return vol, last_quote, primary_quote, ctr
    prev_vol = vol
    metric_amt = cfg.metric_uses_dy ?
        begin
            vol += step * trader.tweak.price_oracle[cfg.volume_index]
            abs(dy)
        end :
        begin
            vol += dy * trader.tweak.price_oracle[cfg.volume_index]
            abs(step)
        end
    last_quote = current_price(trader, trade.pair)
    primary_quote = last_quote
    ctr += 1
    denom_value = trader.curve.x[cfg.metric_index]
    record_leg_metrics!(shell, metric_amt, price_before, last_quote, denom_value, trade_dt)
    Logger.log_leg!(shell.logger, 0, trade.candle_index, trade, stage_str, price_before, last_quote,
                    cfg.max_env, cfg.min_env, last_quote, step, dy, vol - prev_vol,
                    vol, ext_vol, ctr, trader)
    if cfg.trace_dir2
        Logger.trace_dir2!(shell.logger, (
            candle = trade.candle_index,
            pair = trade.pair,
            price_before = price_before,
            price_after = last_quote,
            limit_low = cfg.p_min,
            step = step,
            dy = dy,
            volume_delta = vol - prev_vol,
            volume_total = vol,
            ext_vol = ext_vol,
        ))
    end
    return vol, last_quote, primary_quote, ctr
end

function run_split_trades!(shell::SimulationShell{T}, trades::AbstractVector{SplitTrade}) where {T<:AbstractFloat}
    trader = shell.trader
    lasts = Dict{Tuple{Int,Int},T}()
    start_timestamp = isempty(trades) ? Int64(0) : trades[1].timestamp
    prev_trade_ts = Int64(0)
    for trade in trades
        a, b = trade.pair
        ai = asset_index(a)
        bi = asset_index(b)
        last_quote = get(lasts, trade.pair, trader.tweak.price_oracle[bi] / trader.tweak.price_oracle[ai])
        dt_raw = prev_trade_ts == 0 ? Int64(0) : trade.timestamp - prev_trade_ts
        trade_dt = dt_raw <= 0 ? zero(T) : T(dt_raw)
        vol = zero(T)
        ext_vol = T(trade.volume) * trader.tweak.price_oracle[bi]
        ctr = 0
        price_before = current_price(trader, trade.pair)
        max_price_env = T(trade.high) * (one(T) - trader.fees.ext_fee)
        min_price_env = T(trade.low) * (one(T) + trader.fees.ext_fee)
        high_quote = last_quote
        low_quote = last_quote
        candle_id = trade.candle_index >>> 1

        cfg = LegConfig(:dir1, zero(T), max_price_env, ai, bi, ai, bi, max_price_env, min_price_env, true, false)
        vol, last_quote, high_quote, ctr = maybe_execute_leg!(
            shell, candle_id, trade, cfg, price_before, vol, ext_vol, trade_dt,
            last_quote, high_quote, ctr,
        )
        high_quote = last_quote
        price_before = current_price(trader, trade.pair)

        cfg = LegConfig(:dir2, min_price_env, zero(T), bi, ai, ai, bi, max_price_env, min_price_env, false, true)
        vol, last_quote, low_quote, ctr = maybe_execute_leg!(
            shell, candle_id, trade, cfg, price_before, vol, ext_vol, trade_dt,
            last_quote, low_quote, ctr,
        )
        low_quote = last_quote

        lasts[trade.pair] = last_quote
        if trader.fees.boost_rate > 0 && trade_dt > 0
            factor = apply_boost!(trader.curve, trade_dt, trader.fees.boost_rate)
            trader.profit.xcp_profit_real *= factor
            trader.profit.xcp *= factor
        end
        midpoint = (high_quote + low_quote) / T(2)
        tweak_price!(trader, trade.timestamp, trade.pair, midpoint)
        Logger.log_tweak!(shell.logger, 0, trade.candle_index, trade, midpoint, high_quote, low_quote, trader)
        trader.metrics_state.total_vol += vol
        elapsed = max(Int64(1), trade.timestamp - start_timestamp + 1)
        ARU_x = trader.profit.xcp_profit_real
        ARU_y = (T(86400) * T(365)) / T(elapsed)
        trader.profit.APY = ARU_x^ARU_y - one(T)
        Metrics.set_apy!(shell.metrics, trader.profit.APY)
        prev_trade_ts = trade.timestamp
    end
    return shell
end

end # module
