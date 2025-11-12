module Metrics

export MetricAccumulator, push_volume!, push_slippage!, set_apy!, summarize

"""
    mutable struct MetricAccumulator

Keeps streaming stats such as APY, liquidity density, and cumulative volume.
"""
mutable struct MetricAccumulator{T}
    volume::T
    slippage_sum::T
    antislippage::T
    slippage_weight::T
    apy::T
end

MetricAccumulator(::Type{T}) where {T} = MetricAccumulator{T}(zero(T), zero(T), zero(T), zero(T), zero(T))

"""
    push_volume!(metric, Δvolume)

Accumulates normalized volume.
"""
function push_volume!(acc::MetricAccumulator{T}, Δvolume::T) where {T}
    acc.volume += Δvolume
    return acc
end

"""
    push_slippage!(metric, Δtime, slippage_value)

Mirrors the C++ logic:
  slippage_count += Δtime
  antislippage += Δtime * slippage
  slippage += Δtime / slippage
"""
function push_slippage!(acc::MetricAccumulator{T}, weight::T, slippage_value::T) where {T}
    weight <= 0 && return acc
    acc.slippage_weight += weight
    acc.antislippage += weight * slippage_value
    acc.slippage_sum += weight / slippage_value
    return acc
end

"""
    set_apy!(metric, apy)
"""
set_apy!(acc::MetricAccumulator{T}, apy::T) where {T} = (acc.apy = apy)

"""
    summarize(metric) -> NamedTuple

Produces the final APY / slippage / liquidity density numbers using the same
formulas as the legacy CLI:
    slippage = slippage_sum / slippage_count / 2
    liquidity_density = 2 * antislippage / slippage_count
"""
function summarize(acc::MetricAccumulator{T}) where {T}
    if acc.slippage_weight == 0
        slippage = zero(T)
        liq_density = zero(T)
    else
        slippage = acc.slippage_sum / acc.slippage_weight / T(2)
        liq_density = T(2) * acc.antislippage / acc.slippage_weight
    end
    return (
        volume = acc.volume,
        slippage = slippage,
        liquidity_density = liq_density,
        apy = acc.apy,
    )
end

end # module
