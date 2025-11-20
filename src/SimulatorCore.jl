module SimulatorCore

using ...DomainTypes: TradePair
using ...DataIO: SimulationConfig

export Money, CurveState, FeeParams, TweakState, ProfitState, MetricsState,
       Trader, StepContext, convert_curve_state, promote_step_context, _log_cast

const Money = Float64

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

mutable struct StepContext{T<:AbstractFloat}
    curve::CurveState{T}
    dx::T
    gas_fee::T
    mid_fee::T
    out_fee::T
    fee_gamma::T
end

StepContext(trader::Trader{T}) where {T} =
    StepContext(trader.curve, trader.profit.dx, trader.fees.gas_fee,
                trader.fees.mid_fee, trader.fees.out_fee, trader.fees.fee_gamma)

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

@inline function _log_cast(log_trader::Trader{S}, value) where {S}
    return eltype(log_trader.curve.x)(value)
end

end # module
