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

export Money, CurveState, geometric_mean2, geometric_mean3, reduction_coefficient2,
       reduction_coefficient3, solve_D, solve_x, exchange2!, exchange3!,
       price2, price3, apply_boost!, SimulationState, run_exact_simulation!,
       run_cpp_trade_bundle!, Instrumentation

const Money = Float64

"""
    mutable struct CurveState

Holds the amplification (`A`), gamma, reference prices `p`, and current balances `x`.
"""
mutable struct CurveState{T}
    A::T
    gamma::T
    n::Int
    p::Vector{T}
    x::Vector{T}
end


struct FeeParams{T}
    mid_fee::T
    out_fee::T
    fee_gamma::T
    ext_fee::T
    gas_fee::T
    boost_rate::T
end

mutable struct TweakState{T}
    price_oracle::Vector{T}
    last_price::Vector{T}
    adjustment_step::T
    allowed_extra_profit::T
    ma_half_time::Int
    last_tweak_mode::Symbol
    last_tweak_norm::T
    not_adjusted::Bool
    heavy_tx::Int
    light_tx::Int
    is_light::Bool
    last_timestamp::Int64
end

mutable struct ProfitState{T}
    dx::T
    D0::T
    xcp::T
    xcp_0::T
    xcp_profit::T
    xcp_profit_real::T
    APY::T
end

mutable struct MetricsState{T}
    total_vol::T
    volume::T
    slippage::T
    antislippage::T
    slippage_count::T
end

mutable struct Trader{T}
    config::SimulationConfig
    curve::CurveState{T}
    fees::FeeParams{T}
    tweak::TweakState{T}
    profit::ProfitState{T}
    metrics_state::MetricsState{T}
end

function Trader(config::SimulationConfig, price_vector::AbstractVector{T}) where {T<:AbstractFloat}
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
    return Trader{T}(config, curve, fees, tweak, profit, metrics_state)
end

mutable struct StepContext{T<:AbstractFloat}
    curve::CurveState{T}
    dx::T
    gas_fee::T
    mid_fee::T
    out_fee::T
    fee_gamma::T
end

StepContext(trader::Trader{T}) where {T<:AbstractFloat} =
    StepContext(trader.curve, trader.profit.dx, trader.fees.gas_fee, trader.fees.mid_fee, trader.fees.out_fee, trader.fees.fee_gamma)

function convert_curve_state(curve::CurveState{S}, ::Type{T}) where {S,T<:AbstractFloat}
    CurveState{T}(T(curve.A), T(curve.gamma), curve.n, T.(curve.p), T.(curve.x))
end

function promote_step_context(ctx::StepContext{S}, ::Type{T}) where {S,T<:AbstractFloat}
    StepContext(
        convert_curve_state(ctx.curve, T),
        T(ctx.dx),
        T(ctx.gas_fee),
        T(ctx.mid_fee),
        T(ctx.out_fee),
        T(ctx.fee_gamma),
    )
end

@inline function _log_cast(log_trader, value)
    return eltype(log_trader.curve.x)(value)
end

include("SimulatorInstrumentation.jl")
const Instr = Instrumentation

mutable struct SimulationState{T}
    trader::Trader{T}
    metrics::Metrics.MetricAccumulator{T}
    debug::Instr.DebugOptions
end

function SimulationState(config::SimulationConfig, price_vector::AbstractVector{T};
                         logger::Union{Nothing,Instr.TradeLogger}=nothing,
                         debug::Union{Nothing,Instr.DebugOptions}=nothing) where {T<:AbstractFloat}
    trader = Trader(config, price_vector)
    metrics = Metrics.MetricAccumulator(T)
    debug_opts = compute_debug_options(logger, debug)
    return SimulationState{T}(trader, metrics, debug_opts)
end

function compute_debug_options(logger::Union{Nothing,Instr.TradeLogger},
                               debug::Union{Nothing,Instr.DebugOptions})
    if debug === nothing
        return Instr.default_debug_options(; logger=logger)
    end
    if logger === nothing
        return debug
    end
    return Instr.with_logger(debug, logger)
end


"""
    CurveState(A, gamma, D, prices)

Creates a pool with `n = length(prices)` coins where each reserve starts at `D / (n * price)`.
"""
function CurveState(A::T, gamma::T, D::T, prices::AbstractVector{T}) where {T<:AbstractFloat}
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

# -- Math helpers -------------------------------------------------------------

geometric_mean2(vals::NTuple{2,T}) where {T<:AbstractFloat} = sqrt(vals[1] * vals[2])

function geometric_mean3(vals::NTuple{3,T}) where {T<:AbstractFloat}
    prod = vals[1] * vals[2] * vals[3]
    D = cbrt(prod)
    for _ in 1:255
        prev = D
        D = (2D + prod / (D * D)) / 3
        if abs(D - prev) <= max(T(1e-12), D / T(1e12))
            return D
        end
    end
    error("geometric_mean3 did not converge")
end

function reduction_coefficient2(x::NTuple{2,T}, gamma::T) where {T<:AbstractFloat}
    K = one(T)
    S = x[1] + x[2]
    for xi in x
        K *= 2 * xi / S
    end
    return gamma > 0 ? gamma / (gamma + 1 - K) : K
end

function reduction_coefficient3(x::NTuple{3,T}, gamma::T) where {T<:AbstractFloat}
    K = T(27)
    S = x[1] + x[2] + x[3]
    for xi in x
        K *= xi / S
    end
    return gamma > 0 ? gamma / (gamma + 1 - K) : K
end

@inline function ensure_positive_gamma(gamma)
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
end

function newton_D_2(A::T, gamma::T, xx::NTuple{2,T}, D0::T) where {T<:AbstractFloat}
    ensure_positive_gamma(gamma)
    D = D0
    x = sort([xx[1], xx[2]]; rev=true)
    S = x[1] + x[2]
    NN = T(2)^2
    A_adj = A * NN
    rev_gamma = inv(gamma)
    gamma_1 = one(T) + gamma
    for _ in 1:255
        D_prev = D
        K0 = NN
        for val in x
            K0 = K0 * val / D
        end
        g1k0 = abs(gamma_1 - K0)
        mul1 = D * rev_gamma * g1k0 * rev_gamma * g1k0 / A_adj
        mul2 = 2 * 2 * K0 / g1k0
        neg_fprime = (S + S * mul2) + mul1 * 2 / K0 - mul2 * D
        neg_fprime > 0 || error("neg_fprime must stay positive")
        D = (D * neg_fprime + D * S - D * D) / neg_fprime - D * (mul1 / neg_fprime) * (1 - K0) / K0
        if D < 0
            D = abs(D) / 2
        end
        if abs(D - D_prev) <= max(T(1e-16), D / T(1e14))
            return D
        end
    end
    error("newton_D_2 did not converge")
end

function newton_D_3(A::T, gamma::T, xx::NTuple{3,T}, D0::T) where {T<:AbstractFloat}
    ensure_positive_gamma(gamma)
    D = D0
    x = sort([xx[1], xx[2], xx[3]]; rev=true)
    S = x[1] + x[2] + x[3]
    A_adj = A * T(27)
    rev_gamma = inv(gamma)
    gamma_1 = one(T) + gamma
    for _ in 1:255
        D_prev = D
        K0 = T(27)
        for val in x
            K0 = K0 * val / D
        end
        g1k0 = abs(gamma_1 - K0)
        mul1 = D * rev_gamma * g1k0 * rev_gamma * g1k0 / A_adj
        mul2 = 2 * 3 * K0 / g1k0
        neg_fprime = (S + S * mul2) + mul1 * 3 / K0 - mul2 * D
        neg_fprime > 0 || error("neg_fprime must stay positive")
        D = (D * (neg_fprime + S - D)) / neg_fprime - D * (mul1 / neg_fprime) * (1 - K0) / K0
        if D < 0
            D = abs(D) / 2
        end
        if abs(D - D_prev) <= max(T(1e-16), D / T(1e14))
            return D
        end
    end
    error("newton_D_3 did not converge")
end

function solve_D(A::T, gamma::T, xp::AbstractVector{T}, N::Int) where {T<:AbstractFloat}
    if N == 2
        return newton_D_2(A, gamma, (xp[1], xp[2]), T(2) * geometric_mean2((xp[1], xp[2])))
    elseif N == 3
        return newton_D_3(A, gamma, (xp[1], xp[2], xp[3]), T(3) * geometric_mean3((xp[1], xp[2], xp[3])))
    else
        throw(ArgumentError("unsupported N=$N"))
    end
end

function newton_y(A::T, gamma::T, x::Vector{T}, N::Int, D::T, i::Int) where {T<:AbstractFloat}
    ensure_positive_gamma(gamma)
    y = D / N
    x_sorted = Vector{T}(undef, N - 1)
    idx = 1
    for j in 1:N
        if j == i
            continue
        end
        x_sorted[idx] = x[j]
        idx += 1
    end
    sort!(x_sorted)
    max_x = maximum(x_sorted)
    conv = max(max_x / T(1e14), D / T(1e14), T(1e-16))
    K0_i = one(T)
    S_i = zero(T)
    for val in x_sorted
        y = y * D / (val * N)
        S_i += val
        K0_i *= val * N / D
    end
    NN = T(N)^N
    g2a = gamma * gamma * (A * NN)
    for _ in 1:255
        y_prev = y
        K0 = K0_i * y * N / D
        K0_1 = one(T) - K0
        S = S_i + y
        g1k0 = abs(gamma + K0_1)
        mul1 = D * g1k0 * g1k0 / g2a
        mul2 = one(T) + (K0 + K0) / g1k0
        yfprime = y + mul1 + (S - D) * mul2
        fprime = yfprime / y
        if fprime <= 0 || K0 == 0
            y = y_prev / 2
            continue
        end
        y = ((yfprime + D - S) + mul1 * K0_1 / K0) / fprime
        if y < 0 || fprime < 0
            y = y_prev / 2
        end
        if abs(y - y_prev) <= max(conv, y / T(1e14))
            return y
        end
    end
    error("newton_y did not converge")
end

function newton_y_3(A::T, gamma::T, x::Vector{T}, D::T, i::Int) where {T<:AbstractFloat}
    ensure_positive_gamma(gamma)
    y = D / 3
    x_sorted = Vector{T}(undef, 2)
    if i == 1
        x_sorted .= sort([x[2], x[3]])
    elseif i == 2
        x_sorted .= sort([x[1], x[3]])
    else
        x_sorted .= sort([x[1], x[2]])
    end
    conv = max(maximum(x_sorted) / T(1e14), D / T(1e14), T(1e-16))
    K0_i = one(T)
    S_i = zero(T)
    for val in x_sorted
        y = y * D / (val * 3)
        S_i += val
        K0_i *= val * 3 / D
    end
    g2a = gamma * gamma * (A * T(27))
    for _ in 1:255
        y_prev = y
        K0 = K0_i * y * 3 / D
        K0_1 = one(T) - K0
        S = S_i + y
        g1k0 = abs(gamma + K0_1)
        mul1 = D * g1k0 * g1k0 / g2a
        mul2 = one(T) + (K0 + K0) / g1k0
        yfprime = y + mul1 + (S - D) * mul2
        fprime = yfprime / y
        if fprime <= 0 || K0 == 0
            y = y_prev / 2
            continue
        end
        y = ((yfprime + D - S) + mul1 * K0_1 / K0) / fprime
        if y < 0 || fprime < 0
            y = y_prev / 2
        end
        if abs(y - y_prev) <= max(conv, y / T(1e14))
            return y
        end
    end
    error("newton_y_3 did not converge")
end

function solve_x(A::T, gamma::T, x::AbstractVector{T}, N::Int, D::T, i::Int) where {T<:AbstractFloat}
    if N == 2
        buf = collect(x)
        return newton_y(A, gamma, buf, N, D, i)
    elseif N == 3
        buf = collect(x)
        return newton_y_3(A, gamma, buf, D, i)
    else
        throw(ArgumentError("unsupported N=$N"))
    end
end

# -- Curve helpers ------------------------------------------------------------

function xp(curve::CurveState{T}) where {T}
    return curve.x .* curve.p
end

function invariant_D(curve::CurveState)
    return solve_D(curve.A, curve.gamma, xp(curve), curve.n)
end

function curve_y(curve::CurveState{T}, x_new::T, i::Int, j::Int) where {T<:AbstractFloat}
    xp_vec = xp(curve)
    xp_vec[i] = x_new * curve.p[i]
    yp = solve_x(curve.A, curve.gamma, xp_vec, curve.n, invariant_D(curve), j)
    return yp / curve.p[j]
end

function get_xcp2(curve::CurveState{T}) where {T<:AbstractFloat}
    D = invariant_D(curve)
    X1 = D / (2 * curve.p[1])
    X2 = D / (2 * curve.p[2])
    return geometric_mean2((X1, X2))
end

function get_xcp3(curve::CurveState{T}) where {T<:AbstractFloat}
    D = invariant_D(curve)
    X1 = D / (3 * curve.p[1])
    X2 = D / (3 * curve.p[2])
    X3 = D / (3 * curve.p[3])
    return geometric_mean3((X1, X2, X3))
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

"""
    price2(curve, i, j, dx)

Computes the instantaneous price for a 2-coin pool using the same finite-difference
approach as the legacy simulator (`Trader::price_2`).
"""
function price2(curve::CurveState{T}, i::Int, j::Int, dx::T) where {T<:AbstractFloat}
    dx_raw = dx / curve.p[i]
    curve_res = curve_y(curve, curve.x[i] + dx_raw, i, j)
    return dx_raw / (curve.x[j] - curve_res)
end

"""
    price3(curve, i, j, dx)

3-coin equivalent of [`price2`](@ref).
"""
function price3(curve::CurveState{T}, i::Int, j::Int, dx::T) where {T<:AbstractFloat}
    dx_raw = dx / curve.p[i]
    curve_res = curve_y(curve, curve.x[i] + dx_raw, i, j)
    return dx_raw / (curve.x[j] - curve_res)
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

function fee2(curve::CurveState{T}, mid_fee::T, out_fee::T, fee_gamma::T) where {T<:AbstractFloat}
    xp_vec = xp(curve)
    coeff = reduction_coefficient2((xp_vec[1], xp_vec[2]), fee_gamma)
    return mid_fee * coeff + out_fee * (1 - coeff)
end

function fee3(curve::CurveState{T}, mid_fee::T, out_fee::T, fee_gamma::T) where {T<:AbstractFloat}
    xp_vec = xp(curve)
    coeff = reduction_coefficient3((xp_vec[1], xp_vec[2], xp_vec[3]), fee_gamma)
    return mid_fee * coeff + out_fee * (1 - coeff)
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

@inline function copy_balances(curve::CurveState{T}) where {T}
    return copy(curve.x)
end

@inline function restore_balances!(curve::CurveState{T}, snapshot::Vector{T}) where {T}
    copyto!(curve.x, snapshot)
end

function step_for_price(trader::Trader{Float64}, p_min, p_max, pair::Tuple{Int,Int},
                        vol, ext_vol, debug::Instr.DebugOptions; stage::Symbol=:auto,
                        probe::Union{Nothing,NamedTuple}=nothing)
    ctx = StepContext(trader)
    hp_ctx = promote_step_context(ctx, Double64)
    step_hp = _step_for_price(hp_ctx, trader, Double64(p_min), Double64(p_max), pair,
                              Double64(vol), Double64(ext_vol), debug; stage=stage, probe=probe)
    return Float64(step_hp)
end

function step_for_price(trader::Trader{T}, p_min, p_max, pair::Tuple{Int,Int},
                        vol, ext_vol, debug::Instr.DebugOptions; stage::Symbol=:auto,
                        probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    ctx = StepContext(trader)
    return _step_for_price(ctx, trader, T(p_min), T(p_max), pair, T(vol), T(ext_vol), debug; stage=stage, probe=probe)
end

@inline function _step_for_price(ctx::StepContext{T}, log_trader, p_min::T, p_max::T, pair::Tuple{Int,Int},
                                 vol::T, ext_vol::T, debug::Instr.DebugOptions; stage::Symbol=:auto,
                                 probe::Union{Nothing,NamedTuple}=nothing) where {T<:AbstractFloat}
    if ctx.curve.n == 3
        return step_for_price_3(ctx, log_trader, p_min, p_max, pair, vol, ext_vol, debug; stage=stage, probe=probe)
    else
        return step_for_price_2(ctx, log_trader, p_min, p_max, pair, vol, ext_vol, debug; stage=stage, probe=probe)
    end
end

function step_for_price_2(ctx::StepContext{T}, log_trader, p_min::T, p_max::T, pair::Tuple{Int,Int},
                          vol::T, ext_vol::T, debug::Instr.DebugOptions; stage::Symbol=:auto,
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
        Instr.trace_step_iter(debug, stage, :grow, (
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
            Instr.trace_step_iter(debug, stage, :shrink, (
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
    Instr.trace_step_result(debug, stage, (
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
        Instr.emit_step_probe(
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
                          vol::T, ext_vol::T, debug::Instr.DebugOptions; stage::Symbol=:auto,
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

function record_leg_metrics!(state::SimulationState{T}, amount::T, price_before::T,
                             price_after::T, denom::T, dt::T) where {T<:AbstractFloat}
    amount <= 0 && return
    price_before == price_after && return
    denom == 0 && return
    volume_delta = amount / denom
    Metrics.push_volume!(state.metrics, volume_delta)
    state.trader.metrics_state.volume += volume_delta
    slip_denom = T(2) * abs(price_before - price_after) * denom
    slip_denom == 0 && return
    slip = (amount * (price_before + price_after)) / slip_denom
    if slip > 0 && dt > 0
        Metrics.push_slippage!(state.metrics, dt, slip)
    end
end

# -- high-level simulation loop ----------------------------------------------

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


struct LegConfig{T}
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

@inline function maybe_execute_leg!(state::SimulationState{T}, candle_id::Int, trade::SplitTrade,
                                     cfg::LegConfig{T}, price_before::T,
                                     vol::T, ext_vol::T, trade_dt::T,
                                     last_quote::T, primary_quote::T,
                                     ctr::Int) where {T<:AbstractFloat}
    trader = state.trader
    stage_str = cfg.stage === :dir1 ? "DIR1" : "DIR2"
    lim_low = cfg.stage === :dir1 ? zero(T) : cfg.p_min
    lim_high = cfg.stage === :dir1 ? cfg.p_max : zero(T)
    # Log PRELEG unconditionally (even if limit not valid), so we capture state/reason
    Instr.log_preleg_event(state.debug.logger, 0, trade.candle_index, trade, stage_str,
                           price_before, lim_low, lim_high, vol, ext_vol, trader.curve.n, trader)
    limit_valid = cfg.stage === :dir1 ?
        (cfg.p_max != zero(T) && cfg.p_max > price_before) :
        (cfg.p_min != zero(T) && cfg.p_min < price_before)
    if !limit_valid
        Instr.log_preleg_limit_event(state.debug.logger, 0, trade.candle_index, trade, stage_str,
                                     price_before, lim_low, lim_high, vol, ext_vol,
                                     cfg.stage === :dir1 ? "price_before >= p_max" : "price_before <= p_min")
        return vol, last_quote, primary_quote, ctr
    end
    probe_ctx = Instr.maybe_step_probe_context(state.debug, candle_id, trade.timestamp, trade.pair, cfg.stage)
    step = step_for_price(trader, cfg.p_min, cfg.p_max, trade.pair,
                          vol, ext_vol, state.debug; stage=cfg.stage, probe=probe_ctx)
    Instr.log_step_event(state.debug.logger, 0, trade.candle_index, trade, stage_str,
                         price_before, lim_low, lim_high, step,
                         vol, ext_vol, trader.fees.gas_fee, trader.curve.n)
    step > 0 || return vol, last_quote, primary_quote, ctr
    dy = execute_trade!(trader, step, cfg.from_asset, cfg.to_asset)
    dy > 0 || return vol, last_quote, primary_quote, ctr
    prev_vol = vol
    if cfg.metric_uses_dy
        vol += step * trader.tweak.price_oracle[cfg.volume_index]
        metric_amt = abs(dy)
    else
        vol += dy * trader.tweak.price_oracle[cfg.volume_index]
        metric_amt = abs(step)
    end
    last_quote = current_price(trader, trade.pair)
    primary_quote = last_quote
    ctr += 1
    denom_value = trader.curve.x[cfg.metric_index]
    record_leg_metrics!(state, metric_amt, price_before, last_quote, denom_value, trade_dt)
    Instr.log_leg_event(state.debug.logger, 0, trade.candle_index, trade, stage_str, price_before, last_quote,
                        cfg.max_env, cfg.min_env, last_quote, step, dy, vol - prev_vol,
                        vol, ext_vol, ctr, trader)
    if cfg.trace_dir2
        Instr.trace_dir2_execution(state.debug, (
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

@inline function run_exact_simulation!(state::SimulationState{T}, trades::AbstractVector{SplitTrade}) where {T<:AbstractFloat}
    return run_split_trades!(state, trades)
end

@inline function run_split_trades!(state::SimulationState{T}, trades::AbstractVector{SplitTrade}) where {T<:AbstractFloat}
    trader = state.trader
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
            state, candle_id, trade, cfg, price_before, vol, ext_vol, trade_dt,
            last_quote, high_quote, ctr,
        )
        high_quote = last_quote
        price_before = current_price(trader, trade.pair)

        cfg = LegConfig(:dir2, min_price_env, zero(T), bi, ai, ai, bi, max_price_env, min_price_env, false, true)
        vol, last_quote, low_quote, ctr = maybe_execute_leg!(
            state, candle_id, trade, cfg, price_before, vol, ext_vol, trade_dt,
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
        Instr.log_tweak_event(state.debug.logger, 0, trade.candle_index, trade, midpoint, high_quote, low_quote, trader)
        trader.metrics_state.total_vol += vol
        elapsed = max(Int64(1), trade.timestamp - start_timestamp + 1)
        ARU_x = trader.profit.xcp_profit_real
        ARU_y = (T(86400) * T(365)) / T(elapsed)
        trader.profit.APY = ARU_x^ARU_y - one(T)
        Metrics.set_apy!(state.metrics, trader.profit.APY)
        prev_trade_ts = trade.timestamp
    end
    return state
end

function run_cpp_trade_bundle!(state::SimulationState{T}, trades::AbstractVector{CPPTrade}) where {T<:AbstractFloat}
    return run_split_trades!(state, adapt_trades(trades))
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

end # module
struct FeeParams{T}
    mid_fee::T
    out_fee::T
    fee_gamma::T
    ext_fee::T
    gas_fee::T
    boost_rate::T
end

mutable struct TweakState{T}
    price_oracle::Vector{T}
    last_price::Vector{T}
    adjustment_step::T
    allowed_extra_profit::T
    ma_half_time::Int
    last_tweak_mode::Symbol
    last_tweak_norm::T
    not_adjusted::Bool
    heavy_tx::Int
    light_tx::Int
    is_light::Bool
    last_timestamp::Int64
end

mutable struct ProfitState{T}
    dx::T
    D0::T
    xcp::T
    xcp_0::T
    xcp_profit::T
    xcp_profit_real::T
    APY::T
end

mutable struct MetricsState{T}
    total_vol::T
    volume::T
    slippage::T
    antislippage::T
    slippage_count::T
end
