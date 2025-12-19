#!/usr/bin/env julia
"""
Kriging (Gaussian Process) search for best APY over (A, mid_fee=out_fee) for 2-coin stableswap.

Runs:
  1) An initial batch of N points (default: 32) using a simple space-filling design
     (8 log-spaced A × 4 fee points).
  2) Fits a GP to APY and chooses the next batch using Expected Improvement (EI),
     always including the max predicted mean candidate.
  3) Repeats for `--rounds` batches (default: 2).

Outputs JSONL rows (one per simulation) including the observed metrics and (for
kriging-selected points) the GP mean/std and EI used to pick them.
"""

using JSON3
using Logging: @info
using Printf
using LinearAlgebra
using Random
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim
const Sim = CryptoSim.Simulator
const Prep = CryptoSim.Preprocessing
const Loader = CryptoSim.ChunkLoader
const DataIO = CryptoSim.DataIO
const Metrics = CryptoSim.Metrics

Base.@kwdef mutable struct Options
    dataset_path::String = ""
    config_path::String = normpath(joinpath(PROJECT_ROOT, "configs", "ethusdt_chunk_config.json"))
    data_dir::String = DataIO.DEFAULT_DATA_DIR
    output::String = normpath(joinpath(PROJECT_ROOT, "artifacts", "experiments", "stableswap2_kriging", "results.jsonl"))
    surface_out::String = ""
    surface_A_count::Int = 256
    surface_fee_count::Int = 256
    window_days::Int = 365
    window_end::Symbol = :last
    A_center::Float64 = 1_707_629.0
    A_factor::Float64 = 10.0
    fee_min::Float64 = 0.0
    fee_max::Float64 = 0.5
    batch::Int = 32
    rounds::Int = 2
    candidates::Int = 20_000
    hyper_samples::Int = 200
    seed::Int = 0
    jitter::Float64 = 1e-9
end

function usage()
    println("""
    Usage: kriging_best_apy_stableswap2.jl --dataset=PATH [options]
      --dataset=PATH         Candle JSON (.json/.json.gz/.json.xz) for a single 2-coin pair
      --config=PATH          Simulator config template (default: $(Options().config_path))
      --data-dir=PATH        Where config datafiles live (default: $(DataIO.DEFAULT_DATA_DIR))
      --output=PATH          JSONL destination (default: $(Options().output))
      --surface-out=PATH     Write a predicted surface grid JSONL (default: disabled)
      --surface-A-count=N    Surface grid A points (default: 256)
      --surface-fee-count=N  Surface grid fee points (default: 256)
      --window-days=N        Restrict to last/first N days (default: 365; 0 disables)
      --window-end=last|first Anchor the window at dataset end or start (default: last)
      --A-center=VAL         Center of the A search range (default: 1707629)
      --A-factor=VAL         Search A in [A/factor, A*factor] (default: 10)
      --fee-min=VAL          Min mid_fee/out_fee (default: 0)
      --fee-max=VAL          Max mid_fee/out_fee (default: 0.5)
      --batch=N              Simulations per round (default: 32)
      --rounds=N             Rounds to run (default: 2)
      --candidates=N         Candidate pool size per round (default: 20000)
      --hyper-samples=N      Random hyperparameter samples for GP fit (default: 200)
      --seed=N               RNG seed (default: 0)
    """)
end

function parse_args()
    opts = Options()
    for arg in ARGS
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif startswith(arg, "--dataset=")
            opts.dataset_path = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--config=")
            opts.config_path = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--data-dir=")
            opts.data_dir = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--output=")
            opts.output = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--surface-out=")
            opts.surface_out = normpath(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--surface-A-count=")
            opts.surface_A_count = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--surface-fee-count=")
            opts.surface_fee_count = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--window-days=")
            opts.window_days = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--window-end=")
            raw = lowercase(split(arg, '='; limit=2)[2])
            if raw in ("last", "end", "tail")
                opts.window_end = :last
            elseif raw in ("first", "start", "head")
                opts.window_end = :first
            else
                error("Invalid --window-end=$(raw). Expected last or first.")
            end
        elseif startswith(arg, "--A-center=")
            opts.A_center = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--A-factor=")
            opts.A_factor = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--fee-min=")
            opts.fee_min = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--fee-max=")
            opts.fee_max = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--batch=")
            opts.batch = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--rounds=")
            opts.rounds = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--candidates=")
            opts.candidates = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--hyper-samples=")
            opts.hyper_samples = parse(Int, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--seed=")
            opts.seed = parse(Int, split(arg, '='; limit=2)[2])
        else
            error("Unrecognized argument: $(arg)")
        end
    end
    isempty(opts.dataset_path) && error("--dataset=PATH is required")
    isfile(opts.dataset_path) || error("Dataset $(opts.dataset_path) not found")
    isfile(opts.config_path) || error("Config template $(opts.config_path) not found")
    opts.window_days >= 0 || error("window_days must be >= 0")
    opts.A_center > 0 || error("A_center must be > 0")
    opts.A_factor > 1 || error("A_factor must be > 1")
    opts.fee_min >= 0 || error("fee_min must be >= 0")
    opts.fee_max > opts.fee_min || error("fee_max must exceed fee_min")
    opts.batch > 0 || error("batch must be positive")
    opts.rounds > 0 || error("rounds must be positive")
    opts.candidates >= opts.batch || error("candidates must be >= batch")
    opts.hyper_samples > 0 || error("hyper_samples must be positive")
    opts.surface_A_count > 0 || error("surface_A_count must be positive")
    opts.surface_fee_count > 0 || error("surface_fee_count must be positive")
    return opts
end

@inline function override_config(base::DataIO.SimulationConfig, A::Float64, fee::Float64)
    return DataIO.SimulationConfig(
        A,
        base.gamma,
        base.D,
        base.n,
        fee,
        fee,
        base.fee_gamma,
        base.adjustment_step,
        base.allowed_extra_profit,
        base.ma_half_time,
        base.ext_fee,
        base.gas_fee,
        base.boost_rate,
        base.log,
    )
end

# -- grids -------------------------------------------------------------------

logspace(lo::Float64, hi::Float64, count::Int) =
    [10.0 ^ x for x in range(log10(lo), log10(hi); length=count)]

linspace(lo::Float64, hi::Float64, count::Int) =
    collect(range(lo, hi; length=count))

function load_dataset_bundle(opts::Options)
    cfg_file = DataIO.load_config(opts.config_path)
    cfg = cfg_file.configurations[1]
    cfg.n == 2 || error("Only n=2 is supported (got n=$(cfg.n))")

    candles = DataIO.load_candles(opts.dataset_path; pair = (0, 1), data_dir = opts.data_dir)
    isempty(candles) && error("No candles loaded from $(opts.dataset_path)")
    sort!(candles, by = c -> c.timestamp)
    raw_start = candles[1].timestamp
    raw_end = candles[end].timestamp
    start_ts = raw_start
    end_ts = raw_end
    if opts.window_days > 0
        window = Int64(opts.window_days) * Int64(86400)
        if opts.window_end === :first
            start_ts = raw_start
            end_ts = raw_start + window
        else
            end_ts = raw_end
            start_ts = raw_end - window
        end
        candles = filter(c -> start_ts <= c.timestamp <= end_ts, candles)
        isempty(candles) && error("No candles remained after applying the window (start=$(start_ts), end=$(end_ts))")
    end

    cpp_trades = DataIO.CPPTrade[]
    sizehint!(cpp_trades, length(candles) * 2)
    candle_idx = 0
    for candle in candles
        DataIO.append_cpp_trades!(cpp_trades, candle_idx, candle, candle.pair)
        candle_idx += 2
    end
    sort!(cpp_trades, by = tr -> tr.timestamp)
    price_vec = Loader.price_vector_from_trades(cfg.n, cpp_trades)
    split_trades = Prep.adapt_trades(cpp_trades)
    return (; cfg, price_vec, trades = split_trades, candles = length(candles), trade_count = length(split_trades),
            start_ts, end_ts, raw_start, raw_end)
end

function run_batch(bundle, base_cfg::DataIO.SimulationConfig, points; threads::Int)
    n = length(points)
    records = Vector{Any}(undef, n)
    Threads.@threads for i in 1:n
        A, fee, meta = points[i]
        cfg = override_config(base_cfg, A, fee)
        records[i] = try
            state = Sim.SimulationState(cfg, bundle.price_vec)
            Sim.run_exact_simulation!(state, bundle.trades)
            metrics = Metrics.summarize(state.metrics)
            trader = state.trader
            merge(
                (
                    status = "ok",
                    mode = "kriging",
                    threads = threads,
                    A = A,
                    mid_fee = fee,
                    out_fee = fee,
                    metrics = metrics,
                    profit = (
                        xcp_profit_real = trader.profit.xcp_profit_real,
                        apy = trader.profit.APY,
                    ),
                ),
                meta,
            )
        catch err
            merge(
                (
                    status = "error",
                    mode = "kriging",
                    threads = threads,
                    A = A,
                    mid_fee = fee,
                    out_fee = fee,
                    error = string(err),
                ),
                meta,
            )
        end
    end
    return records
end

# -- GP / Kriging ------------------------------------------------------------

@inline function to_features(A::Float64, fee::Float64, logA_min::Float64, logA_max::Float64,
                             fee_min::Float64, fee_max::Float64)
    x1 = (log10(A) - logA_min) / (logA_max - logA_min)
    x2 = (fee - fee_min) / (fee_max - fee_min)
    return (x1, x2)
end

@inline function sqdist(x1, x2, l1, l2)
    d1 = (x1[1] - x2[1]) / l1
    d2 = (x1[2] - x2[2]) / l2
    return d1 * d1 + d2 * d2
end

function kernel_matrix(X::Vector{NTuple{2,Float64}}, σf::Float64, l1::Float64, l2::Float64,
                       σn::Float64, jitter::Float64)
    n = length(X)
    K = Matrix{Float64}(undef, n, n)
    σf2 = σf * σf
    @inbounds for i in 1:n
        K[i, i] = σf2 + σn * σn + jitter
        xi = X[i]
        for j in i+1:n
            d2 = sqdist(xi, X[j], l1, l2)
            v = σf2 * exp(-0.5 * d2)
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

function log_marginal_likelihood(X::Vector{NTuple{2,Float64}}, y::Vector{Float64},
                                σf::Float64, l1::Float64, l2::Float64, σn::Float64,
                                jitter::Float64)
    K = kernel_matrix(X, σf, l1, l2, σn, jitter)
    chol = cholesky(Symmetric(K); check=false)
    # If Cholesky fails, `chol.info` is non-zero.
    chol.info == 0 || return -Inf
    α = chol \ y
    # log |K| = 2 * sum(log(diag(L)))
    logdet = 2.0 * sum(log, diag(chol.L))
    n = length(y)
    return -0.5 * dot(y, α) - 0.5 * logdet - 0.5 * n * log(2π)
end

function fit_gp(X::Vector{NTuple{2,Float64}}, y_raw::Vector{Float64};
                samples::Int, rng::AbstractRNG, jitter::Float64)
    y_mean = mean(y_raw)
    y = y_raw .- y_mean
    y_std = std(y)
    y_std = isfinite(y_std) && y_std > 0 ? y_std : 1.0

    best_ll = -Inf
    best = nothing

    # Random search over reasonable ranges in normalized [0,1] coords.
    # lengthscales: 0.02..0.5 (log-uniform); noise: 1e-6..0.05*y_std; σf: 0.1..5*y_std.
    for _ in 1:samples
        l1 = 10.0 ^ rand(rng, range(log10(0.02), log10(0.5); length=1024))
        l2 = 10.0 ^ rand(rng, range(log10(0.02), log10(0.5); length=1024))
        σf = y_std * (10.0 ^ rand(rng, range(log10(0.1), log10(5.0); length=1024)))
        σn = y_std * (10.0 ^ rand(rng, range(log10(1e-6), log10(5e-2); length=1024)))
        ll = log_marginal_likelihood(X, y, σf, l1, l2, σn, jitter)
        if ll > best_ll
            best_ll = ll
            best = (σf = σf, l1 = l1, l2 = l2, σn = σn, y_mean = y_mean)
        end
    end
    best === nothing && error("GP fit failed (no valid hyperparameters)")
    return best_ll, best
end

function build_gp_posterior(X::Vector{NTuple{2,Float64}}, y_raw::Vector{Float64}, hyp; jitter::Float64)
    y = y_raw .- hyp.y_mean
    K = kernel_matrix(X, hyp.σf, hyp.l1, hyp.l2, hyp.σn, jitter)
    chol = cholesky(Symmetric(K); check=true)
    α = chol \ y
    return (; chol, α)
end

@inline function normal_pdf(z::Float64)
    return exp(-0.5 * z * z) / sqrt(2π)
end

@inline function normal_cdf(z::Float64)
    # A&S 7.1.26 rational approximation (no SpecialFunctions dependency).
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    x = abs(z)
    t = 1.0 / (1.0 + p * x)
    poly = (((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t)
    approx = 1.0 - normal_pdf(x) * poly
    return z >= 0 ? approx : 1.0 - approx
end

function gp_predict_one(X::Vector{NTuple{2,Float64}}, post, hyp, xstar::NTuple{2,Float64})
    n = length(X)
    k = Vector{Float64}(undef, n)
    σf2 = hyp.σf * hyp.σf
    @inbounds for i in 1:n
        d2 = sqdist(X[i], xstar, hyp.l1, hyp.l2)
        k[i] = σf2 * exp(-0.5 * d2)
    end
    μ = dot(k, post.α) + hyp.y_mean
    v = post.chol.L \ k
    σ2 = max(0.0, σf2 - dot(v, v))
    return μ, sqrt(σ2)
end

function gp_mean_one(X::Vector{NTuple{2,Float64}}, post, hyp, xstar::NTuple{2,Float64})
    n = length(X)
    k = Vector{Float64}(undef, n)
    σf2 = hyp.σf * hyp.σf
    @inbounds for i in 1:n
        d2 = sqdist(X[i], xstar, hyp.l1, hyp.l2)
        k[i] = σf2 * exp(-0.5 * d2)
    end
    return dot(k, post.α) + hyp.y_mean
end

function select_next_batch(X::Vector{NTuple{2,Float64}}, y::Vector{Float64}, opts::Options;
                           rng::AbstractRNG)
    ll, hyp = fit_gp(X, y; samples=opts.hyper_samples, rng=rng, jitter=opts.jitter)
    post = build_gp_posterior(X, y, hyp; jitter=opts.jitter)

    y_best = maximum(y)
    ξ = 1e-6

    candidates = Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}(undef, opts.candidates)
    # (A, fee, mean, std, EI)
    logA_min = log10(opts.A_center / opts.A_factor)
    logA_max = log10(opts.A_center * opts.A_factor)

    @inbounds for i in 1:opts.candidates
        u1 = rand(rng)
        u2 = rand(rng)
        logA = logA_min + u1 * (logA_max - logA_min)
        A = 10.0 ^ logA
        fee = opts.fee_min + u2 * (opts.fee_max - opts.fee_min)
        xstar = to_features(A, fee, logA_min, logA_max, opts.fee_min, opts.fee_max)
        μ, σ = gp_predict_one(X, post, hyp, xstar)
        ei = 0.0
        if σ > 0
            z = (μ - y_best - ξ) / σ
            ei = (μ - y_best - ξ) * normal_cdf(z) + σ * normal_pdf(z)
            if !isfinite(ei) || ei < 0
                ei = 0.0
            end
        end
        candidates[i] = (A, fee, μ, σ, ei)
    end

    # Always include the max predicted mean (exploit), then fill by EI (explore).
    point_key(A, fee) = (Int64(round(log10(A) * 1e12)), Int64(round(fee * 1e12)))
    seen = Set{Tuple{Int64,Int64}}()

    best_mean_idx = findmax(c -> c[3], candidates)[2]
    picks = Tuple{Float64,Float64,NamedTuple}[]
    A0, fee0, μ0, σ0, ei0 = candidates[best_mean_idx]
    key0 = point_key(A0, fee0)
    push!(seen, key0)
    push!(picks, (A0, fee0,
                  (kriging = (gp_mean = μ0, gp_std = σ0, ei = ei0),
                   gp = (log_marginal = ll, σf = hyp.σf, l1 = hyp.l1, l2 = hyp.l2, σn = hyp.σn))))

    order = sortperm(1:length(candidates); by = i -> candidates[i][5], rev = true)
    for idx in order
        length(picks) >= opts.batch && break
        A, fee, μ, σ, ei = candidates[idx]
        key = point_key(A, fee)
        key in seen && continue
        push!(seen, key)
        push!(picks, (A, fee, (kriging = (gp_mean = μ, gp_std = σ, ei = ei),)))
    end
    return ll, hyp, picks
end

function dedupe_points!(points, seen::Set{Tuple{Int64,Int64}})
    out = typeof(points)()
    for (A, fee, meta) in points
        key = (Int64(round(log10(A) * 1e12)), Int64(round(fee * 1e12)))
        if !(key in seen)
            push!(seen, key)
            push!(out, (A, fee, meta))
        end
    end
    return out
end

function initial_points(opts::Options)
    logA_min = log10(opts.A_center / opts.A_factor)
    logA_max = log10(opts.A_center * opts.A_factor)
    As = [10.0 ^ (logA_min + (i + 0.5) / 8 * (logA_max - logA_min)) for i in 0:7]
    fees = [opts.fee_min + (j + 0.5) / 4 * (opts.fee_max - opts.fee_min) for j in 0:3]
    pts = Tuple{Float64,Float64,NamedTuple}[]
    for A in As, fee in fees
        push!(pts, (A, fee, (round = 1, strategy = "initial",)))
    end
    length(pts) == 32 || error("internal: expected 32 initial points")
    if opts.batch != 32
        # For non-32 batches, truncate or pad with random points.
        if opts.batch < 32
            return pts[1:opts.batch]
        else
            rng = MersenneTwister(opts.seed == 0 ? 1 : opts.seed)
            while length(pts) < opts.batch
                u1 = rand(rng)
                u2 = rand(rng)
                A = 10.0 ^ (logA_min + u1 * (logA_max - logA_min))
                fee = opts.fee_min + u2 * (opts.fee_max - opts.fee_min)
                push!(pts, (A, fee, (round = 1, strategy = "initial-random",)))
            end
        end
    end
    return pts
end

function dump_surface(out_path::AbstractString, opts::Options, observations)
    ok = filter(r -> r[:status] == "ok", observations)
    isempty(ok) && error("No successful points to build a surface")

    logA_min = log10(opts.A_center / opts.A_factor)
    logA_max = log10(opts.A_center * opts.A_factor)

    X = NTuple{2,Float64}[]
    apy = Float64[]
    vol = Float64[]
    slip = Float64[]
    liq = Float64[]
    for r in ok
        A = Float64(r[:A])
        fee = Float64(r[:mid_fee])
        push!(X, to_features(A, fee, logA_min, logA_max, opts.fee_min, opts.fee_max))
        m = r[:metrics]
        push!(apy, Float64(m[:apy]))
        push!(vol, Float64(m[:volume]))
        push!(slip, Float64(m[:slippage]))
        push!(liq, Float64(m[:liquidity_density]))
    end

    rng = MersenneTwister(opts.seed == 0 ? 1 : opts.seed)
    surface_hyper = min(opts.hyper_samples, 120)
    @info "surface fit" points=length(apy) hyper_samples=surface_hyper

    _, hyp_apy = fit_gp(X, apy; samples=surface_hyper, rng=rng, jitter=opts.jitter)
    post_apy = build_gp_posterior(X, apy, hyp_apy; jitter=opts.jitter)
    _, hyp_vol = fit_gp(X, vol; samples=surface_hyper, rng=rng, jitter=opts.jitter)
    post_vol = build_gp_posterior(X, vol, hyp_vol; jitter=opts.jitter)
    _, hyp_slip = fit_gp(X, slip; samples=surface_hyper, rng=rng, jitter=opts.jitter)
    post_slip = build_gp_posterior(X, slip, hyp_slip; jitter=opts.jitter)
    _, hyp_liq = fit_gp(X, liq; samples=surface_hyper, rng=rng, jitter=opts.jitter)
    post_liq = build_gp_posterior(X, liq, hyp_liq; jitter=opts.jitter)

    As = logspace(opts.A_center / opts.A_factor, opts.A_center * opts.A_factor, opts.surface_A_count)
    fees = linspace(opts.fee_min, opts.fee_max, opts.surface_fee_count)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        for A in As, fee in fees
            xstar = to_features(A, fee, logA_min, logA_max, opts.fee_min, opts.fee_max)
            apy_hat = gp_mean_one(X, post_apy, hyp_apy, xstar)
            vol_hat = gp_mean_one(X, post_vol, hyp_vol, xstar)
            slip_hat = gp_mean_one(X, post_slip, hyp_slip, xstar)
            liq_hat = gp_mean_one(X, post_liq, hyp_liq, xstar)
            record = (
                status = "ok",
                mode = "kriging_surface",
                source = opts.output,
                points = length(ok),
                A = A,
                mid_fee = fee,
                out_fee = fee,
                metrics = (
                    apy = max(0.0, apy_hat),
                    volume = max(0.0, vol_hat),
                    slippage = max(0.0, slip_hat),
                    liquidity_density = max(0.0, liq_hat),
                ),
            )
            JSON3.write(io, record)
            println(io)
        end
    end
    @info "surface written" path=out_path A_points=length(As) fee_points=length(fees)
    return out_path
end

function main()
    opts = parse_args()
    opts.seed != 0 && Random.seed!(opts.seed)
    mkpath(dirname(opts.output))

    bundle = load_dataset_bundle(opts)
    @info "dataset bundle" dataset=opts.dataset_path candles=bundle.candles trades=bundle.trade_count start_ts=bundle.start_ts end_ts=bundle.end_ts threads=Threads.nthreads()

    # Run.
    base_cfg = bundle.cfg
    points_seen = Set{Tuple{Int64,Int64}}()
    observations = Any[]

    open(opts.output, "w") do io
        # Initial batch.
        pts0 = dedupe_points!(initial_points(opts), points_seen)
        @info "round 1" points=length(pts0) strategy="initial"
        recs = run_batch(bundle, base_cfg, pts0; threads=Threads.nthreads())
        for r in recs
            push!(observations, r)
            JSON3.write(io, r); println(io)
        end

        # Subsequent rounds.
        rng = MersenneTwister(opts.seed == 0 ? 1 : opts.seed)
        for round in 2:opts.rounds
            ok = filter(r -> r[:status] == "ok", observations)
            X = NTuple{2,Float64}[]
            y = Float64[]
            logA_min = log10(opts.A_center / opts.A_factor)
            logA_max = log10(opts.A_center * opts.A_factor)
            for r in ok
                A = Float64(r[:A])
                fee = Float64(r[:mid_fee])
                push!(X, to_features(A, fee, logA_min, logA_max, opts.fee_min, opts.fee_max))
                push!(y, Float64(r[:metrics][:apy]))
            end
            length(y) >= 8 || error("Need at least 8 successful points to fit GP (got $(length(y)))")
            ll, hyp, pts = select_next_batch(X, y, opts; rng=rng)
            pts = [(A, fee, merge(meta, (round = round, strategy = "kriging",))) for (A, fee, meta) in pts]
            pts = dedupe_points!(pts, points_seen)
            if length(pts) < opts.batch
                # Rarely, the candidate pool can collide with previously-sampled points after rounding;
                # top up with fresh random draws to preserve the requested batch size.
                logA_min = log10(opts.A_center / opts.A_factor)
                logA_max = log10(opts.A_center * opts.A_factor)
                while length(pts) < opts.batch
                    A = 10.0 ^ (logA_min + rand(rng) * (logA_max - logA_min))
                    fee = opts.fee_min + rand(rng) * (opts.fee_max - opts.fee_min)
                    extra = dedupe_points!([(A, fee, (round = round, strategy = "kriging-topup",))], points_seen)
                    isempty(extra) && continue
                    push!(pts, extra[1])
                end
            end
            @info "round $(round)" points=length(pts) strategy="kriging" log_marginal=ll σf=hyp.σf l1=hyp.l1 l2=hyp.l2 σn=hyp.σn best_apy=maximum(y)
            recs = run_batch(bundle, base_cfg, pts; threads=Threads.nthreads())
            for r in recs
                push!(observations, r)
                JSON3.write(io, r); println(io)
            end
        end
    end

    ok = filter(r -> r[:status] == "ok", observations)
    isempty(ok) && error("No successful simulations")
    best = ok[findmax(r -> Float64(r[:metrics][:apy]), ok)[2]]
    @info "best observed" A=best[:A] fee=best[:mid_fee] apy=best[:metrics][:apy] volume=best[:metrics][:volume] slippage=best[:metrics][:slippage]

    if !isempty(opts.surface_out)
        dump_surface(opts.surface_out, opts, observations)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
