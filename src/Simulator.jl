module Simulator

try
    using DoubleFloats: Double64
catch
    @eval const Double64 = Float64
end
using JSON3
using Logging: @debug

using ..DataIO: SimulationConfig, CPPTrade
using ..Metrics
using ..Preprocessing: SplitTrade, adapt_trades

include("SimulatorCore.jl")
using .SimulatorCore: Money, CurveState, FeeParams, TweakState, ProfitState,
                       MetricsState, Trader, StepContext, convert_curve_state,
                       promote_step_context, _log_cast
include("SimulatorMath.jl")
using .SimulatorMath
include("SimulatorInstrumentation.jl")
include("SimulatorLogger.jl")
using .SimulatorLogger

const Instr = Instrumentation
const Logger = SimulatorLogger

export Money, CurveState, geometric_mean2, geometric_mean3, reduction_coefficient2,
       reduction_coefficient3, solve_D, solve_x, exchange2!, exchange3!,
       price2, price3, apply_boost!, SimulationState, run_exact_simulation!,
       run_cpp_trade_bundle!, Instrumentation

function SimulatorCore.Trader(config::SimulationConfig, price_vector::AbstractVector{T}) where {T<:AbstractFloat}
    fees = FeeParams(
        T(config.mid_fee),
        T(config.out_fee),
        T(config.fee_gamma),
        T(config.ext_fee),
        T(config.gas_fee),
        T(config.boost_rate) / (T(86400) * T(365)),
    )
    pvec = T.(price_vector)
    curve = CurveState(T(config.A), T(config.gamma), T(config.D), pvec)
    tweak = TweakState(
        copy(pvec),
        copy(pvec),
        T(config.adjustment_step),
        T(config.allowed_extra_profit),
        config.ma_half_time,
        :none,
        zero(T),
        false,
        0,
        0,
        false,
        Int64(0),
    )
    D0 = invariant_D(curve)
    xcp_0 = curve.n == 3 ? get_xcp3(curve) : get_xcp2(curve)
    profit = ProfitState(T(config.D) * T(1e-8), D0, xcp_0, xcp_0, one(T), one(T), zero(T))
    metrics_state = MetricsState(zero(T), zero(T), zero(T), zero(T), zero(T))
    return SimulatorCore.Trader{T}(config, curve, fees, tweak, profit, metrics_state)
end

mutable struct SimulationState{T}
    trader::Trader{T}
    metrics::Metrics.MetricAccumulator{T}
    logger::Logger.LoggerAdapter
end

function SimulationState(config::SimulationConfig, price_vector::AbstractVector{T};
                         logger::Union{Nothing,Instr.TradeLogger}=nothing,
                         debug::Union{Nothing,Instr.DebugOptions}=nothing) where {T<:AbstractFloat}
    trader = Trader(config, price_vector)
    metrics = Metrics.MetricAccumulator(T)
    adapter = Logger.build_logger(logger, debug)
    return SimulationState{T}(trader, metrics, adapter)
end

function step_for_price end
function execute_trade! end
function tweak_price! end

include("SimulationRunner.jl")
using .SimulationRunner
const Runner = SimulationRunner

to_shell(state::SimulationState{T}) where {T<:AbstractFloat} =
    Runner.SimulationShell{T}(state.trader, state.metrics, state.logger)

"""
    CurveState(A, gamma, D, prices)

Creates a pool with `n = length(prices)` coins where each reserve starts at `D / (n * price)`.
"""
function SimulatorCore.CurveState(A::T, gamma::T, D::T, prices::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(prices)
    if n ∉ (2, 3)
        throw(ArgumentError("CurveState currently supports 2 or 3 assets, got $n"))
    end
    p = collect(prices)
    x = similar(p)
    for i in 1:n
        x[i] = D / (n * p[i])
    end
    return CurveState{T}(A, gamma, n, p, x)
end

function update_xcp!(trader::Trader{T}; only_real::Bool=false) where {T<:AbstractFloat}
    new_xcp = trader.curve.n == 3 ? get_xcp3(trader.curve) : get_xcp2(trader.curve)
    ratio = new_xcp / trader.profit.xcp
    trader.profit.xcp_profit_real *= ratio
    if !only_real
        trader.profit.xcp_profit *= ratio
    end
    trader.profit.xcp = new_xcp
    return trader.profit.xcp
end

function execute_trade!(trader::Trader{T}, dx::T, from::Int, to::Int) where {T<:AbstractFloat}
    dy = if trader.curve.n == 3
        exchange3!(trader.curve, dx, from, to;
            mid_fee=trader.fees.mid_fee, out_fee=trader.fees.out_fee,
            fee_gamma=trader.fees.fee_gamma)
    else
        exchange2!(trader.curve, dx, from, to;
            mid_fee=trader.fees.mid_fee, out_fee=trader.fees.out_fee,
            fee_gamma=trader.fees.fee_gamma)
    end
    if dy > 0
        update_xcp!(trader)
    end
    return dy
end

function exchange2!(curve::CurveState{T}, dx::T, i::Int, j::Int;
                    mid_fee::T, out_fee::T, fee_gamma::T, max_price::T=typemax(T)) where {T<:AbstractFloat}
    x_old = copy(curve.x)
    try
        x_new = curve.x[i] + dx
        y = curve_y(curve, x_new, i, j)
        curve.x[i] = x_new
        curve.x[j] = y
        fee_mul = one(T) - fee2(curve, mid_fee, out_fee, fee_gamma)
        dy = x_old[j] - y
        curve.x[j] = x_old[j] - dy * fee_mul
        price = dx / dy
        if price > max_price || dy <= 0
            curve.x .= x_old
            return zero(T)
        end
        return dy
    catch err
        curve.x .= x_old
        @debug "exchange2! aborted" err
        return zero(T)
    end
end

function exchange3!(curve::CurveState{T}, dx::T, i::Int, j::Int;
                    mid_fee::T, out_fee::T, fee_gamma::T, max_price::T=typemax(T)) where {T<:AbstractFloat}
    x_old = copy(curve.x)
    try
        x_new = curve.x[i] + dx
        y = curve_y(curve, x_new, i, j)
        curve.x[i] = x_new
        curve.x[j] = y
        fee_mul = one(T) - fee3(curve, mid_fee, out_fee, fee_gamma)
        dy = x_old[j] - y
        curve.x[j] = x_old[j] - dy * fee_mul
        price = dx / dy
        if price > max_price || dy <= 0
            curve.x .= x_old
            return zero(T)
        end
        return dy
    catch err
        curve.x .= x_old
        @debug "exchange3! aborted" err
        return zero(T)
    end
end

"""
    apply_boost!(curve, dt, rate) -> T

Applies an external boost (donation) by scaling every reserve equally. Returns the
scaling factor for callers that also need to adjust ancillary accumulators.
"""
function apply_boost!(curve::CurveState{T}, dt::T, rate::T) where {T<:AbstractFloat}
    factor = one(T) + dt * rate
    for k in eachindex(curve.x)
        curve.x[k] *= factor
    end
    return factor
end

@inline asset_index(id::Int) = id + 1

@inline function copy_balances(curve::CurveState{T}) where {T}
    return copy(curve.x)
end

@inline function restore_balances!(curve::CurveState{T}, snapshot::Vector{T}) where {T}
    copyto!(curve.x, snapshot)
end

function step_for_price(trader::Trader{Float64}, p_min, p_max, pair::Tuple{Int,Int},
                        vol, ext_vol, logger::Logger.LoggerAdapter; stage::Symbol=:auto,
                        probe::Union{Nothing,NamedTuple}=nothing)
    ctx = StepContext(trader)
    hp_ctx = promote_step_context(ctx, Double64)
    step_hp = _step_for_price(hp_ctx, trader, Double64(p_min), Double64(p_max), pair,
                              Double64(vol), Double64(ext_vol), logger; stage=stage, probe=probe)
    return Float64(step_hp)
end

function step_for_price(trader::Trader{T}, p_min, p_max, pair::Tuple{Int,Int},
                        vol, ext_vol, logger::Logger.LoggerAdapter; stage::Symbol=:auto,
                        probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    ctx = StepContext(trader)
    return _step_for_price(ctx, trader, T(p_min), T(p_max), pair, T(vol), T(ext_vol), logger; stage=stage, probe=probe)
end

@inline function _step_for_price(ctx::StepContext{T}, log_trader, p_min::T, p_max::T, pair::Tuple{Int,Int},
                                 vol::T, ext_vol::T, logger::Logger.LoggerAdapter; stage::Symbol=:auto,
                                 probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    if ctx.curve.n == 3
        return step_for_price_3(ctx, log_trader, p_min, p_max, pair, vol, ext_vol, logger; stage=stage, probe=probe)
    else
        return step_for_price_2(ctx, log_trader, p_min, p_max, pair, vol, ext_vol, logger; stage=stage, probe=probe)
    end
end

function step_for_price_2(ctx::StepContext{T}, log_trader, p_min::T, p_max::T, pair::Tuple{Int,Int},
                          vol::T, ext_vol::T, logger::Logger.LoggerAdapter; stage::Symbol=:auto,
                          probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    curve = ctx.curve
    x0 = copy_balances(curve)
    _dx = zero(T)
    _dy = zero(T)
    from = asset_index(pair[1])
    to = asset_index(pair[2])
    if p_min > 0
        from, to = to, from
    end
    step0 = ctx.dx / curve.p[from]
    step = step0
    gas = ctx.gas_fee / curve.p[from]
    previous_profit = zero(T)
    price = zero(T)

    while true
        dx_prev = _dx
        dy_prev = _dy
        _dx += step
        x = x0[from] + _dx
        y = curve_y(curve, x, from, to)
        curve.x[from] = x
        curve.x[to] = y
        fee_mul = one(T) - fee2(curve, ctx.mid_fee, ctx.out_fee, ctx.fee_gamma)
        _dy = (x0[to] - y) * fee_mul
        curve.x[to] = x0[to] - _dy
        price = from == asset_index(pair[1]) ? _dx / _dy : _dy / _dx
        price_ref = curve.p[to]
        v = vol + _dy * price_ref
        restore_balances!(curve, x0)
        new_profit = from == asset_index(pair[1]) ?
            (_dx / price - _dx / p_max) * p_max :
            (price - p_min) * _dx
        Logger.trace_step_iter!(logger, stage, :grow, (
            pair = pair,
            from_asset = from,
            to_asset = to,
            dx = _dx,
            dy = _dy,
            step = step,
            price = price,
            new_profit = new_profit,
            previous_profit = previous_profit,
            volume = vol,
            trial_volume = v,
            ext_vol = ext_vol,
            p_min = p_min,
            p_max = p_max,
        ))
        if new_profit > previous_profit && v <= ext_vol / 2
            previous_profit = new_profit
        else
            _dx = dx_prev
            _dy = dy_prev
            break
        end
        step += step
    end

    while true
        dx_prev = _dx
        dy_prev = _dy
        if step < 0
            step = -step
        end
        step /= 2
        step < step0 && break
        for _ in 1:2
            step = -step
            trial_dx = _dx + step
            x = x0[from] + trial_dx
            y = curve_y(curve, x, from, to)
            curve.x[from] = x
            curve.x[to] = y
            fee_mul = one(T) - fee2(curve, ctx.mid_fee, ctx.out_fee, ctx.fee_gamma)
            trial_dy = (x0[to] - y) * fee_mul
            curve.x[to] = x0[to] - trial_dy
            price = from == asset_index(pair[1]) ? trial_dx / trial_dy : trial_dy / trial_dx
            price_ref = curve.p[to]
            v = vol + trial_dy * price_ref
            restore_balances!(curve, x0)
            new_profit = from == asset_index(pair[1]) ?
                (trial_dx / price - trial_dx / p_max) * p_max :
                (price - p_min) * trial_dx
            Logger.trace_step_iter!(logger, stage, :shrink, (
                pair = pair,
                from_asset = from,
                to_asset = to,
                dx = trial_dx,
                dy = trial_dy,
                step = step,
                price = price,
                new_profit = new_profit,
                previous_profit = previous_profit,
                volume = vol,
                trial_volume = v,
                ext_vol = ext_vol,
                p_min = p_min,
                p_max = p_max,
            ))
            if new_profit > previous_profit && v <= ext_vol / 2
                previous_profit = new_profit
                _dx = trial_dx
                _dy = trial_dy
                break
            end
        end
    end

    if from == asset_index(pair[1])
        price = (_dx + gas) / _dy
        previous_profit = (_dx / price - _dx / p_max) * p_max
    else
        price = _dy / (_dx + gas)
        previous_profit = (price - p_min) * _dx
    end
    if previous_profit <= 0
        _dx = zero(T)
    end
    Logger.trace_step_result!(logger, stage, (
        pair = pair,
        from_asset = from,
        to_asset = to,
        dx = _dx,
        price = price,
        profit = previous_profit,
        gas = gas,
        p_min = p_min,
        p_max = p_max,
    ))
    if probe !== nothing
        Logger.emit_step_probe!(
            debug,
            probe,
            log_trader,
            from,
            to,
            _log_cast(log_trader, _dx),
            _log_cast(log_trader, _dy),
            _log_cast(log_trader, price),
            _log_cast(log_trader, gas),
            _log_cast(log_trader, p_min),
            _log_cast(log_trader, p_max),
            _log_cast(log_trader, vol),
            _log_cast(log_trader, ext_vol),
        )
    end
    return _dx
end
function step_for_price_3(ctx::StepContext{T}, log_trader, p_min::T, p_max::T, pair::Tuple{Int,Int},
                          vol::T, ext_vol::T, logger::Logger.LoggerAdapter; stage::Symbol=:auto,
                          probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    curve = ctx.curve
    x0 = copy_balances(curve)
    _dx = zero(T)
    _dy = zero(T)
    from = asset_index(pair[1])
    to = asset_index(pair[2])
    if p_min > 0
        from, to = to, from
    end
    step0 = ctx.dx / curve.p[from]
    step = step0
    gas = ctx.gas_fee / curve.p[from]
    previous_profit = zero(T)

    while true
        dx_prev = _dx
        dy_prev = _dy
        _dx += step
        x = x0[from] + _dx
        y = curve_y(curve, x, from, to)
        curve.x[from] = x
        curve.x[to] = y
        fee_mul = one(T) - fee3(curve, ctx.mid_fee, ctx.out_fee, ctx.fee_gamma)
        _dy = (x0[to] - y) * fee_mul
        curve.x[to] = x0[to] - _dy
        price = from == asset_index(pair[1]) ? _dx / _dy : _dy / _dx
        v = vol + _dy * curve.p[to]
        restore_balances!(curve, x0)
        new_profit = from == asset_index(pair[1]) ?
            (_dx / price - _dx / p_max) * p_max :
            (price - p_min) * _dx
        if new_profit > previous_profit && v <= ext_vol / 2
            previous_profit = new_profit
        else
            _dx = dx_prev
            _dy = dy_prev
            break
        end
        step += step
    end

    while true
        dx_prev = _dx
        dy_prev = _dy
        if step < 0
            step = -step
        end
        step /= 2
        step < step0 && break
        for _ in 1:2
            step = -step
            trial_dx = _dx + step
            x = x0[from] + trial_dx
            y = curve_y(curve, x, from, to)
            curve.x[from] = x
            curve.x[to] = y
            fee_mul = one(T) - fee3(curve, ctx.mid_fee, ctx.out_fee, ctx.fee_gamma)
            trial_dy = (x0[to] - y) * fee_mul
            curve.x[to] = x0[to] - trial_dy
            price = from == asset_index(pair[1]) ? trial_dx / trial_dy : trial_dy / trial_dx
            v = vol + trial_dy * curve.p[to]
            restore_balances!(curve, x0)
            new_profit = from == asset_index(pair[1]) ?
                (trial_dx / price - trial_dx / p_max) * p_max :
                (price - p_min) * trial_dx
            if new_profit > previous_profit && v <= ext_vol / 2
                previous_profit = new_profit
                _dx = trial_dx
                _dy = trial_dy
                break
            end
        end
    end

    if from == asset_index(pair[1])
        price = (_dx + gas) / _dy
        previous_profit = (_dx / price - _dx / p_max) * p_max
    else
        price = _dy / (_dx + gas)
        previous_profit = (price - p_min) * _dx
    end
    if previous_profit <= 0
        _dx = zero(T)
    end
    if probe !== nothing
        Logger.emit_step_probe!(
            logger,
            probe,
            log_trader,
            from,
            to,
            _log_cast(log_trader, _dx),
            _log_cast(log_trader, _dy),
            _log_cast(log_trader, price),
            _log_cast(log_trader, gas),
            _log_cast(log_trader, p_min),
            _log_cast(log_trader, p_max),
            _log_cast(log_trader, vol),
            _log_cast(log_trader, ext_vol),
        )
    end
    return _dx
end

function ma_recorder!(trader::Trader{T}, timestamp::Int64, price_vector::Vector{T}) where {T<:AbstractFloat}
    last_ts = trader.tweak.last_timestamp
    dt = last_ts == 0 ? timestamp : timestamp - last_ts
    dt <= 0 && return
    alpha = T(0.5) ^ (T(dt) / T(trader.tweak.ma_half_time))
    for k in 2:length(price_vector)
        trader.tweak.price_oracle[k] = price_vector[k] * (one(T) - alpha) + trader.tweak.price_oracle[k] * alpha
    end
    trader.tweak.last_timestamp = timestamp
end

function tweak_price!(trader::Trader{T}, timestamp::Int64, pair::Tuple{Int,Int}, midpoint::T) where {T<:AbstractFloat}
    if trader.curve.n == 3
        return tweak_price_3!(trader, timestamp, pair, midpoint)
    else
        return tweak_price_2!(trader, timestamp, pair, midpoint)
    end
end

function tweak_price_2!(trader::Trader{T}, timestamp::Int64, pair::Tuple{Int,Int}, midpoint::T) where {T<:AbstractFloat}
    ma_recorder!(trader, timestamp, trader.tweak.last_price)
    a, b = pair
    if b > 0
        trader.tweak.last_price[asset_index(b)] = midpoint * trader.tweak.last_price[asset_index(a)]
    else
        trader.tweak.last_price[asset_index(a)] = trader.tweak.last_price[1] / midpoint
    end
    norm = zero(T)
    for i in 1:trader.curve.n
        delta = trader.tweak.price_oracle[i] / trader.curve.p[i] - one(T)
        norm += delta * delta
    end
    norm = sqrt(norm)
    trader.tweak.last_tweak_norm = norm
    trader.tweak.last_tweak_mode = :none
    adj_step = max(trader.tweak.adjustment_step, norm / T(10))
    if norm <= adj_step
        trader.tweak.is_light = true
        trader.tweak.light_tx += 1
        trader.tweak.last_tweak_mode = :light
        return norm
    end
    if !trader.tweak.not_adjusted && (trader.profit.xcp_profit_real > sqrt(trader.profit.xcp_profit) * (one(T) + trader.tweak.allowed_extra_profit))
        trader.tweak.not_adjusted = true
    end
    if !trader.tweak.not_adjusted
        trader.tweak.light_tx += 1
        trader.tweak.is_light = true
        trader.tweak.last_tweak_mode = :light
        return norm
    end
    trader.tweak.last_tweak_mode = :heavy
    trader.tweak.heavy_tx += 1
    trader.tweak.is_light = false
    p_new = copy(trader.curve.p)
    for i in 1:trader.curve.n
        p_target = trader.curve.p[i]
        p_real = trader.tweak.price_oracle[i]
        p_new[i] = p_target + adj_step * (p_real - p_target) / norm
    end
    old_p = copy(trader.curve.p)
    old_profit = trader.profit.xcp_profit_real
    old_xcp = trader.profit.xcp
    copyto!(trader.curve.p, p_new)
    update_xcp!(trader; only_real=true)
    if trader.profit.xcp_profit_real <= sqrt(trader.profit.xcp_profit)
        copyto!(trader.curve.p, old_p)
        trader.profit.xcp_profit_real = old_profit
        trader.profit.xcp = old_xcp
        trader.tweak.not_adjusted = false
    end
    return norm
end

function tweak_price_3!(trader::Trader{T}, timestamp::Int64, pair::Tuple{Int,Int}, midpoint::T) where {T<:AbstractFloat}
    ma_recorder!(trader, timestamp, trader.tweak.last_price)
    a, b = pair
    if b > 0
        trader.tweak.last_price[asset_index(b)] = midpoint * trader.tweak.last_price[asset_index(a)]
    else
        trader.tweak.last_price[asset_index(a)] = trader.tweak.last_price[1] / midpoint
    end
    norm = zero(T)
    for i in 1:trader.curve.n
        delta = trader.tweak.price_oracle[i] / trader.curve.p[i] - one(T)
        norm += delta * delta
    end
    norm = sqrt(norm)
    trader.tweak.last_tweak_norm = norm
    trader.tweak.last_tweak_mode = :none
    adj_step = max(trader.tweak.adjustment_step, norm / T(10))
    if norm <= adj_step
        trader.tweak.is_light = true
        trader.tweak.light_tx += 1
        trader.tweak.last_tweak_mode = :light
        return norm
    end
    if !trader.tweak.not_adjusted && (trader.profit.xcp_profit_real > sqrt(trader.profit.xcp_profit) * (one(T) + trader.tweak.allowed_extra_profit))
        trader.tweak.not_adjusted = true
    end
    if !trader.tweak.not_adjusted
        trader.tweak.light_tx += 1
        trader.tweak.is_light = true
        trader.tweak.last_tweak_mode = :light
        return norm
    end
    trader.tweak.last_tweak_mode = :heavy
    trader.tweak.heavy_tx += 1
    trader.tweak.is_light = false
    p_new = copy(trader.curve.p)
    for i in 1:trader.curve.n
        p_target = trader.curve.p[i]
        p_real = trader.tweak.price_oracle[i]
        p_new[i] = p_target + adj_step * (p_real - p_target) / norm
    end
    old_p = copy(trader.curve.p)
    old_profit = trader.profit.xcp_profit_real
    old_xcp = trader.profit.xcp
    copyto!(trader.curve.p, p_new)
    update_xcp!(trader; only_real=true)
    if trader.profit.xcp_profit_real <= sqrt(trader.profit.xcp_profit)
        copyto!(trader.curve.p, old_p)
        trader.profit.xcp_profit_real = old_profit
        trader.profit.xcp = old_xcp
        trader.tweak.not_adjusted = false
    end
    return norm
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

function update_price_oracle!(trader::Trader{T}, pair::Tuple{Int,Int}, close_val) where {T}
    a, b = pair
    ai = asset_index(a)
    bi = asset_index(b)
    oracle = trader.tweak.price_oracle
    if a == 0
        if bi <= length(oracle)
            oracle[bi] = T(close_val)
        end
    elseif b == 0
        if ai <= length(oracle) && close_val != 0
            oracle[ai] = T(1 / close_val)
        end
    elseif ai <= length(oracle) && bi <= length(oracle)
        oracle[bi] = oracle[ai] * T(close_val)
    end
end

function update_price_oracle!(trader::Trader{T}, trade::SplitTrade) where {T}
    update_price_oracle!(trader, trade.pair, trade.close)
end

@inline function run_exact_simulation!(state::SimulationState{T}, trades::AbstractVector{SplitTrade}) where {T<:AbstractFloat}
    shell = to_shell(state)
    Runner.run_split_trades!(shell, trades)
    return state
end

@inline function run_split_trades!(state::SimulationState{T}, trades::AbstractVector{SplitTrade}) where {T<:AbstractFloat}
    shell = to_shell(state)
    Runner.run_split_trades!(shell, trades)
    return state
end

function run_cpp_trade_bundle!(state::SimulationState{T}, trades::AbstractVector{CPPTrade}) where {T<:AbstractFloat}
    shell = to_shell(state)
    Runner.run_split_trades!(shell, adapt_trades(trades))
    return state
end

end # module
