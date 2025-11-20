module SimulatorMath

using ..SimulatorCore: CurveState

export geometric_mean2, geometric_mean3, reduction_coefficient2, reduction_coefficient3,
       ensure_positive_gamma, newton_D_2, newton_D_3, solve_D, newton_y, newton_y_3,
       solve_x, xp, invariant_D, curve_y, get_xcp2, get_xcp3,
       copy_balances, restore_balances!, apply_boost!, price2, price3, fee2, fee3

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

@inline function copy_balances(curve::CurveState{T}) where {T}
    return copy(curve.x)
end

@inline function restore_balances!(curve::CurveState{T}, snapshot::Vector{T}) where {T}
    copyto!(curve.x, snapshot)
end

function apply_boost!(curve::CurveState{T}, dt::T, rate::T) where {T<:AbstractFloat}
    factor = one(T) + dt * rate
    for k in eachindex(curve.x)
        curve.x[k] *= factor
    end
    return factor
end

function price2(curve::CurveState{T}, i::Int, j::Int, dx::T) where {T<:AbstractFloat}
    dx_raw = dx / curve.p[i]
    curve_res = curve_y(curve, curve.x[i] + dx_raw, i, j)
    return dx_raw / (curve.x[j] - curve_res)
end

function price3(curve::CurveState{T}, i::Int, j::Int, dx::T) where {T<:AbstractFloat}
    dx_raw = dx / curve.p[i]
    curve_res = curve_y(curve, curve.x[i] + dx_raw, i, j)
    return dx_raw / (curve.x[j] - curve_res)
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

end # module
