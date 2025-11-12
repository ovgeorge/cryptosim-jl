module Tweaks

export TweakState, TweakResult, apply_tweak!

"""
    mutable struct TweakState

Holds per-candle tweak bookkeeping (EMA, profits, etc.).
"""
mutable struct TweakState{T}
    price_oracle::Vector{T}
    curve_prices::Vector{T}
    last_price::Vector{T}
    allowed_extra_profit::T
    adjustment_step::T
    not_adjusted::Bool
end

"""
    struct TweakResult

Lightweight report describing the tweak decision.
"""
struct TweakResult{T}
    mode::Symbol
    norm::T
end

"""
    apply_tweak!(state, args...) -> TweakResult

Stub that will eventually mirror `tweak_price_{2,3}` from the C++ implementation.
"""
function apply_tweak!(state::TweakState, args...)
    return TweakResult(:none, zero(eltype(state.price_oracle)))
end

end # module
