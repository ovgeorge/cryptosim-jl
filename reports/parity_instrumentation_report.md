# Parity and Instrumentation Report

Date: 2025-11-11

This report documents how the C++ “gold” simulator was reviewed and instrumented, how parity tests were re-run across reference chunks, the log-diff methodology used, where divergences occur, and likely root causes with next steps.

## Gold Source Overview

- Primary file: `gold-standard/cryptopool-simulator/main.cpp`
  - CLI: `simu [trim[#]] [threads=#] [in-json] [out-json]` (main at `:1723`)
  - Data ingestion: memory-mapped scan of Binance-style JSON (`get_data`, `:178–229`)
  - Candle splitting: emits two legs per candle with half-volume and time offsets (`:299–345`)
  - Price vector seed: first USD legs fill `p[0]=1` and remaining entries (`:233–268`)
  - Curve: `newton_D_{2,3}`, `newton_y*`, `price_{2,3}`, `exchange_{2,3}` (`:473–734`, `:903–1266`)
  - Fees: reduction coefficients `reduction_coefficient_{2,3}` blend fees (`:440–458`)
  - Step search: `step_for_price_{2,3}` selects `dx` under gas and external price limits (`:917–1250`)
  - Oracle + tweaks: EMA + guarded price adjustments (`tweak_price_{2,3}` at `:1292–1410`)
  - Metrics: APY, liquidity density, slippage, and volume accumulation (`simulate` body `:1421–1618`)

## Instrumentation

Two instrumented C++ trees exist; both contain instrumentation within their `main.cpp`:

- `cryptopool-simulator-instr/main.cpp` (CPPDBG JSON loggers; includes `log_step_event`, `log_leg_event`, `log_tweak_event`).
  - On this machine, the ASan debug build (`simud`) reported an access error in `get_data` (mapping oversized dataset). Fast path segfaulted. Not used further.
- `instrumented-solver/main.cpp` (also instrumented; stable here)
  - Executed with `CPP_TRADE_DEBUG=1` to emit line-delimited JSON with `CPPDBG` prefix to stdout.

Julia side instrumentation lives under `src/SimulatorInstrumentation.jl` with a compatible schema. Logs are emitted with `JULIADB` prefix.

## How Parity Was Run

1) Baseline metrics parity, all fixture chunks, fully parallel:
   - `bash scripts/run_parity_parallel.sh --output reports/parity_latest.jsonl`
2) Replayed C++ logs with instrumentation for each chunk using `instrumented-solver/simu`:
   - `CPP_TRADE_DEBUG=1 ./simu trim<chunk> ../test/fixtures/chunks/<chunk>/chunk-config.json ../test/fixtures/chunks/<chunk>/results.cpp.replay.json > ../test/fixtures/chunks/<chunk>/cpp_stdout.replay.log 2>&1`
   - Extracted to `test/fixtures/chunks/<chunk>/cpp_log.jsonl` via `grep '^CPPDBG ' ...`
3) Implemented a relaxed log diff tool to align events by `(event, timestamp, pair, stage/mode)`:
   - Added `scripts/diff_chunks_relaxed.jl`
   - Re-ran parity with relaxed diff: `bash scripts/run_parity_parallel.sh --diff-script=scripts/diff_chunks_relaxed.jl --output reports/parity_instr_relaxed.jsonl 00972 00980 00984 01071`

Artifacts:

- Metrics summaries: `reports/parity_latest.jsonl`, `reports/parity_instr_relaxed.jsonl`
- Per-chunk logs: `test/fixtures/chunks/<id>/julia_log.<id>.jsonl`, `test/fixtures/chunks/<id>/cpp_log.jsonl`

## Metrics Parity Summary (4 chunks)

- Relative error averages:
  - Volume: −13.60%
  - Slippage: +0.43%
  - Liquidity density: −0.41%
  - APY: −18.08%

Per-chunk highlights (Julia vs C++):

- 00972
  - vol −15.15%, slip +0.42%, liq −0.40%, apy −17.09%
- 00980
  - vol −12.28%, slip +0.53%, liq −0.50%, apy −16.73%
- 00984
  - vol −13.44%, slip +0.44%, liq −0.42%, apy −18.35%
- 01071
  - vol −13.52%, slip +0.31%, liq −0.30%, apy −20.17%

Slippage and liquidity density are close (≤ ~0.5% rel), while volume and APY diverge materially.

## Log-Level Parity Findings

Initial strict event-key alignment (event + candle_index + stage + timestamp) produced “No overlapping events found” due to different candle_index conventions between C++ and Julia.

Using the relaxed key (event + timestamp + pair + stage/mode) yields many matches and exposes concrete mismatches:

Earliest mismatches per chunk (all `LEG` events):

- 00972 @ ts=1640241065, pair=(0,1), stage=DIR2, field=dy
  - Julia: `dy=92651.01184663177`
  - C++  : `dy=110261.19610395952`
  - Files: `test/fixtures/chunks/00972/julia_log.00972.jsonl:35`, `test/fixtures/chunks/00972/cpp_log.jsonl:67`

- 00980 @ ts=1640241065, pair=(0,1), stage=DIR2, field=dy
  - Julia: `dy=91420.61294233799`
  - C++  : `dy=110234.7867676448`

- 00984 @ ts=1640241065, pair=(0,1), stage=DIR2, field=dy
  - Julia: `dy=85324.99647819996`
  - C++  : `dy=110183.14106990978`

- 01071 @ ts=1640237705, pair=(0,1), stage=DIR2, field=dy
  - Julia: `dy=56218.724091380835`
  - C++  : `dy=34565.13307729422`

Context around first mismatch (00972):

- Prior `STEP` (DIR1) at ts=1640240035 shows a small price drift:
  - Julia `price_before=3952.7501682102998`
  - C++  `price_before=3952.750052419719`
  - Both had `dx=0` (no trade) but the small drift accumulates.

- At mismatch ts=1640241065 (DIR2), state snapshots diverge:
  - Julia `price_before=3956.50548`, `price_oracle=[1.0,3955.0824188841316]`, `last_price=[1.0,3956.50548]`, reserves `[1.5001486e8, 37944.5447]`
  - C++  `price_before=3957.55661`, `price_oracle=[1.0,3955.9525932710258]`, `last_price=[1.0,3957.55661]`, reserves `[1.5002731e8, 37941.4071]`
  - Limits match (`min_price=3950.104676`, `max_price=3947.735324`), but different `dx` decisions lead to different `dy` and volumes.

Event count summaries for 00972 (relaxed unique keys):

- Julia keys: 963; C++ keys: 1790; matched: 835
  - Note: C++ emits more granular STEP/LEG lines (expected given loop structure); relaxed matching focuses on overlapping event keys.

## Likely Sources of Divergence

The first clear quantitative mismatch is the chosen `dx` during `DIR2` legs. Contributing factors that explain the observed pattern:

1) Numeric precision differences
   - C++ operates on `long double` for money; the Julia rewrite uses `Float64` (`const Money = Float64`).
   - The step search doubles and halves `step` around a non-linear objective; even tiny differences in `price_before` (observed at ~1e-4) and EMA can pick different `dx` plateaus, producing large `dy` deltas (10–30% range) later in the run.

2) EMA/oracle timing and mid-price
   - Both implementations update EMA via half-life and tweak once per combined candle using midpoint `(high+low)/2`.
   - The logs show different `price_oracle` and `last_price` at the mismatch timestamp, likely compounding from earlier candles.

3) Step search gating by external volume
   - Both sides gate by `v <= ext_vol / 2` with `v = vol + dy * price_oracle[to]` (DIR2) or `vol + step * price_oracle[from]` (DIR1). Small deltas in oracle produce different admissible step sizes.

4) Fee blending and marginal price
   - The fee-reduction coefficient and finite-difference price computation match the C++ logic. Minor numerical drift can still push decisions across thresholds.

Non-causes verified:

- Data splitting and time offsets match the C++ splitter when using `DataIO.build_cpp_trades` (used by the parity script). The “flat candle” special-case is not used in this path.
- Gas/fee handling matches structurally in both directions (included in final price check and profit test).

## Recommendations / Next Steps

1) Instrument step selection at the mismatched event
   - Use the existing Julia step-probe plumbing to dump the entire step-search trajectory for 00972 @ ts=1640241065 (DIR2). Compare against the single C++ `STEP`/`LEG` decision to confirm which branch differs (profit comparison vs ext_vol gating vs fee computation).
   - The Julia logger already supports `JULIA_STEP_PROBE` and `JULIA_STEP_PROBE_FILE` via `SimulatorInstrumentation`.

2) Test numeric sensitivity
   - Re-run the same chunk using `Double64` (DoubleFloats) for the Money type in Julia to approximate C++ long double and observe if `dx/dy` and metrics converge towards C++.

3) Tighten EMA parity
   - Emit additional fields in both logs (pre/post EMA during `TWEAK`) to assert exact EMA updates and detect any off-by-one or timestamp handling differences.

4) Isolate a single-candle harness
   - Build a deterministic single-candle scenario extracted from 00972 around ts=1640241065 to A/B `step_for_price_2` input/output. This reduces compounding and directly validates the step search.

## Reproduction Notes

- Build and run Julia summaries (emits Julia logs):
  - `julia --project=. scripts/chunk_summary.jl --keep-logs 00972`
- Rebuild and run instrumented C++ (instrumented-solver):
  - `(cd instrumented-solver && CPP_TRADE_DEBUG=1 ./simu trim00972 ../test/fixtures/chunks/00972/chunk-config.json ../test/fixtures/chunks/00972/results.cpp.replay.json > ../test/fixtures/chunks/00972/cpp_stdout.replay.log 2>&1)`
  - `grep '^CPPDBG ' test/fixtures/chunks/00972/cpp_stdout.replay.log | sed 's/^CPPDBG //g' > test/fixtures/chunks/00972/cpp_log.jsonl`
- Relaxed diff against logs:
  - `julia --project=. scripts/diff_chunks_relaxed.jl test/fixtures/chunks/00972/julia_log.00972.jsonl test/fixtures/chunks/00972/cpp_log.jsonl`

## Appendix: Key Files

- Gold source: `gold-standard/cryptopool-simulator/main.cpp`
- C++ instrumentation: `instrumented-solver/main.cpp`
- Julia core: `src/Simulator.jl`, `src/SimulatorInstrumentation.jl`, `src/Preprocessing.jl`, `src/DataIO.jl`
- Diff tools: `scripts/diff_chunks.jl`, `scripts/diff_chunks_relaxed.jl`
- Parity runners: `scripts/chunk_summary.jl`, `scripts/run_parity_parallel.sh`

