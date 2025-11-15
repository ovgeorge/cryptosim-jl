#!/usr/bin/env julia

"""
Replay a single candle leg to isolate step_for_price_2/3 plateau decisions.

Usage:
  julia --project=. scripts/replay_single_candle.jl --chunk=chunk00973 --stage=DIR1 --timestamp=1621480315

Inputs:
  - Expects per-chunk logs:
      test/fixtures/chunks/<chunk>/julia_log.<chunk>.jsonl
      test/fixtures/chunks/<chunk>/cpp_log.jsonl
  - Uses chunk-config.json (or copied config) to load the base SimulationConfig

Outputs:
  - Prints a compact report comparing:
      * recorded C++ dx/dy vs Julia-computed dx/dy on C++ snapshot
      * recorded Julia dx/dy vs Julia-computed dx/dy on Julia snapshot
  - Useful to confirm whether logic (not just state) explains plateau divergence.
"""

using JSON3

include(joinpath(@__DIR__, "..", "src", "CryptoSim.jl"))
using .CryptoSim

const Sim = CryptoSim.Simulator
const DataIO = CryptoSim.DataIO
const Domain = CryptoSim.DomainTypes

struct Args
    chunk::String
    stage::String
    timestamp::Union{Nothing,Int}
end

function parse_args()
    chunk = ""
    stage = ""
    timestamp::Union{Nothing,Int} = nothing
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--chunk=")
            chunk = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--stage=")
            stage = uppercase(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--timestamp=")
            timestamp = parse(Int, split(arg, '='; limit=2)[2])
        else
            error("Unknown arg: $arg")
        end
        i += 1
    end
    isempty(chunk) && error("--chunk=chunkXXXXX required")
    (stage == "DIR1" || stage == "DIR2") || error("--stage must be DIR1 or DIR2")
    return Args(chunk, stage, timestamp)
end

function find_leg_event(path::AbstractString; stage::String, timestamp::Union{Nothing,Int})
    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            if startswith(s, "CPPDBG ")
                s = s[8:end]
            elseif startswith(s, "JULIADB ")
                s = s[9:end]
            end
            ev = JSON3.read(s)
            if get(ev, "event", "") != "LEG"
                continue
            end
            if timestamp !== nothing && Int(get(ev, "timestamp", -1)) == timestamp
                return ev
            elseif timestamp === nothing && String(get(ev, "stage", "")) == stage
                return ev
            end
        end
    end
    return nothing
end

function load_config_for_chunk(paths::Domain.ChunkPaths)
    if isfile(paths.chunk_config)
        return DataIO.load_config(paths.chunk_config)
    else
        meta = JSON3.read(open(paths.metadata, "r") do io; read(io, String); end)
        copied = haskey(meta, "config") ? String(meta["config"]) : "single-run.json"
        return DataIO.load_config(joinpath(paths.dir, copied))
    end
end

function build_trader_from_snapshot(cfg::DataIO.SimulationConfig, snap)
    n = Int(length(snap["reserves"]))
    # Initialize with config/defaults, then override with snapshot
    trader = Sim.Trader(cfg, ones(Float64, n))
    trader.curve.p .= Float64.(snap["target_prices"])
    trader.curve.x .= Float64.(snap["reserves"])
    trader.tweak.price_oracle .= Float64.(snap["price_oracle"])
    trader.tweak.last_price .= Float64.(snap["last_price"])
    return trader
end

"""
    prestate_from_leg(cfg, leg) -> Dict with keys :x, :p, :oracle, :last

Reconstruct pre-LEG reserves for a 2-coin leg using the post-LEG snapshot and recorded (dx, dy).
Assumes C++/Julia exchange2! semantics:
  x_post[i] = x_old[i] + dx
  x_post[j] = x_old[j] - dy * (1 - fee2(xp_pre))
Where fee2 is evaluated at the pre-fee intermediate state with balances (x_old[i]+dx, y), and y = x_old[j] - dy.
We solve a fixed-point in f = 1 - fee2 using xp = ( (x_old[i]+dx)*p[i], y*p[j] ) and y = x_post[j] - dy*(1 - f).
"""
function prestate_from_leg(cfg::DataIO.SimulationConfig, leg)
    n = length(leg["reserves"])
    n == 2 || error("prestate_from_leg currently supports n=2 (got $n)")
    p = Float64.(leg["target_prices"])
    x_post = Float64.(leg["reserves"])
    dx = Float64(get(leg, "dx", 0.0))
    dy = Float64(get(leg, "dy", 0.0))
    # Determine from/to mapping
    pair = (Int(leg["pair"][1]), Int(leg["pair"][2]))
    stage = String(leg["stage"]) == "DIR2" ? :dir2 : :dir1
    from_idx = stage == :dir2 ? Sim.asset_index(pair[2]) : Sim.asset_index(pair[1])
    to_idx   = stage == :dir2 ? Sim.asset_index(pair[1]) : Sim.asset_index(pair[2])
    # Fixed-point solve for fee multiplier fmul
    mid_fee = Float64(cfg.mid_fee)
    out_fee = Float64(cfg.out_fee)
    fee_gamma = Float64(cfg.fee_gamma)
    # Pre-fee i after adding dx equals post i balance
    xi_prefee = x_post[from_idx]
    function fee_mul_of(f)
        y_prefee = x_post[to_idx] - dy * (1 - f)
        xp1 = xi_prefee * p[from_idx]
        xp2 = y_prefee * p[to_idx]
        coeff = Sim.reduction_coefficient2((xp1, xp2), fee_gamma)
        fee = mid_fee * coeff + out_fee * (1 - coeff)
        return 1 - fee
    end
    # Iterate fixed point
    f = 1 - (mid_fee + out_fee) / 2
    for _ in 1:100
        f_new = fee_mul_of(f)
        if isfinite(f_new) && abs(f_new - f) <= 1e-14
            f = f_new
            break
        end
        f = f_new
    end
    # Reconstruct x_old
    x_old = copy(x_post)
    x_old[from_idx] = x_post[from_idx] - dx
    x_old[to_idx]   = x_post[to_idx] + dy * f
    return (; x = x_old, p = p,
            oracle = Float64.(leg["price_oracle"]),
            last = Float64.(leg["last_price"]))
end

function compute_step(trader::Sim.Trader{T}, ev) where {T}
    # Determine p_min/p_max from stage
    stage = String(ev["stage"]) == "DIR2" ? :dir2 : :dir1
    p_min = stage == :dir2 ? T(get(ev, "limit_low", 0.0)) : zero(T)
    p_max = stage == :dir1 ? T(get(ev, "limit_high", 0.0)) : zero(T)
    pair = (Int(ev["pair"][1]), Int(ev["pair"][2]))
    vol = T(get(ev, "volume_before", 0.0))
    ext_vol = T(get(ev, "ext_vol", 0.0))
    debug = Sim.Instr.noop_debug_options()
    dx = Sim.step_for_price(trader, p_min, p_max, pair, vol, ext_vol, debug; stage=stage)
    # Compute dy for reference using cloned trader
    local_trader = deepcopy(trader)
    from_asset = stage == :dir2 ? Sim.asset_index(pair[2]) : Sim.asset_index(pair[1])
    to_asset = stage == :dir2 ? Sim.asset_index(pair[1]) : Sim.asset_index(pair[2])
    dy = Sim.execute_trade!(local_trader, dx, from_asset, to_asset)
    return dx, dy
end

function main()
    args = parse_args()
    root = normpath(joinpath(@__DIR__, ".."))
    chunk_root = normpath(joinpath(root, "test", "fixtures", "chunks"))
    paths = Domain.ChunkPaths(chunk_root, args.chunk)
    cfg_file = load_config_for_chunk(paths)
    cfg = cfg_file.configurations[1]
    # Find step events in both logs
    jlog = Domain.default_julia_log_path(paths)
    clog = paths.cpp_log
    isfile(jlog) || error("Julia log not found: $(jlog)")
    isfile(clog) || error("C++ log not found: $(clog)")
    # Use LEG events to reconstruct snapshots and inputs
    j_leg = find_leg_event(jlog; stage=args.stage, timestamp=args.timestamp)
    c_leg = find_leg_event(clog; stage=args.stage, timestamp=args.timestamp)
    j_leg === nothing && error("LEG not found in Julia log for given filter")
    c_leg === nothing && error("LEG not found in C++ log for given filter")
    # Reconstruct pre-LEG traders by unwinding fees from LEG payloads
    j_pre = prestate_from_leg(cfg, j_leg)
    c_pre = prestate_from_leg(cfg, c_leg)
    function build_from_pre(cfg, pre)
        t = Sim.Trader(cfg, ones(Float64, length(pre.p)))
        t.curve.p .= pre.p
        t.curve.x .= pre.x
        t.tweak.price_oracle .= pre.oracle
        t.tweak.last_price .= pre.last
        return t
    end
    jt = build_from_pre(cfg, j_pre)
    ct = build_from_pre(cfg, c_pre)
    # Synthesize the minimal STEP inputs from the LEG record
    function step_inputs_from_leg(ev)
        stage = String(ev["stage"]) == "DIR2" ? :dir2 : :dir1
        p_min = stage == :dir2 ? Float64(ev["min_price"]) : 0.0
        p_max = stage == :dir1 ? Float64(ev["max_price"]) : 0.0
        pair = (Int(ev["pair"][1]), Int(ev["pair"][2]))
        vol_before = Float64(ev["volume_total"]) - Float64(ev["volume_delta"]) # before this leg
        ext_vol = Float64(ev["ext_vol"])
        return (; stage, p_min, p_max, pair, vol=vol_before, ext_vol)
    end
    j_in = step_inputs_from_leg(j_leg)
    c_in = step_inputs_from_leg(c_leg)

    # Shim to compute step given synthesized inputs
    function compute_step_from_inputs(trader::Sim.Trader, inp)
        dx = Sim.step_for_price(trader, inp.p_min, inp.p_max, inp.pair, inp.vol, inp.ext_vol, Sim.Instr.noop_debug_options(); stage=inp.stage)
        tmp = deepcopy(trader)
        from_asset = inp.stage == :dir2 ? Sim.asset_index(inp.pair[2]) : Sim.asset_index(inp.pair[1])
        to_asset = inp.stage == :dir2 ? Sim.asset_index(inp.pair[1]) : Sim.asset_index(inp.pair[2])
        dy = Sim.execute_trade!(tmp, dx, from_asset, to_asset)
        return dx, dy
    end

    # Compute dx/dy under Julia logic on both snapshots
    j_dx, j_dy = compute_step_from_inputs(jt, j_in)
    c_dx, c_dy = compute_step_from_inputs(ct, c_in)
    # Report
    ts0 = Int(j_leg["timestamp"])
    println("Single-Candle Replay (", args.chunk, ", stage=", args.stage, ", ts=", ts0, ")")
    println("Recorded (Julia): dx=", Float64(get(j_leg, "dx", 0.0)))
    println("Pre-LEG  (Julia on Julia-pre): dx=", j_dx, " dy=", j_dy)
    println("Recorded (C++  ): dx=", Float64(get(c_leg, "dx", 0.0)))
    println("Pre-LEG  (Julia on C++-pre  ): dx=", c_dx, " dy=", c_dy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
