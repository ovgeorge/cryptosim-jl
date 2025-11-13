# Chunk07913 Parity Investigation Plan

Goal: explain the massive slippage divergence observed in `chunk07913` (`slippage rel_err ≈ 23.63`, log mismatch at `LEG` candle 64) by auditing every stage where the Julia simulator may deviate from the C++ reference. Previous incidents were caused by logical mismatches rather than floating-point precision, so we explicitly avoid chasing Double64 vs Float64 issues.

## 1. Data intake parity
- **Chunk fixture sanity**: Re-dump `artifacts/chunks_ethusdt-1m_100/data/ethusdt-1m_chunk07913.json` and ensure both simulators point at the same file (`metadata.json` currently shows both gold/instrumented using identical paths; regenerate once true gold binary exists).
- **Julia loader**: Step through `DataIO.load_candles`, `_parse_candles`, `split_leg_extrema`, and `build_cpp_trades` to confirm we reproduce the C++ `get_all` semantics (ordering of dir1/dir2 legs, volume normalization, `is_last` flags). Verify no hidden `trim` flag is being set; `DataIO.build_cpp_trades` should only honor `trim` when explicitly requested.
- **C++ input path**: Review `scripts/generate_chunks.py` + `cryptopool-simulator/main-stableswap.cpp` argument wiring to ensure the chunk config references the same datafile names that the Julia run resolves via `DataIO`. Confirm that the `_tmp.*` binary logs used for parity also come from the same chunk slice (no leftover symlinks).

Status: ✅ Confirmed. `cryptopool-simulator/download/ethusdt-1m_chunk07913.json` is identical to `artifacts/chunks_ethusdt-1m_100/data/ethusdt-1m_chunk07913.json`, and both match the `[791300:791400]` slice of `artifacts/ethusdt-1m.json`. `metadata.json` has no `trim_flag`, and Julia’s `build_cpp_trades` reproduces `cpp_log` split timestamps/leg ordering.

## 2. Preprocessing & split generation
- **Trade splitting**: Inspect `Preprocessing.adapt_trades`, `SplitTrade`, and helpers to confirm we emit the same intermediate trades and leg ordering as the C++ pipeline. Cross-check with `cpp_log.jsonl` for chunk07913: number of `SPLIT`/`PRELEG` events, timestamps, and `leg_dir`. Spot-check for off-by-one candle indices or missing `is_last` flags.
- **Trim flags & filtering**: Ensure nothing reuses `trimXXXX` semantics (a historic source of parity bugs). Search for any `parse_trim` usages in `scripts/chunk_summary.jl` and confirm they are gated behind `metadata["trim_flag"]`; chunk07913 should run without trimming.
- **Volume pruning**: double-check optional logic such as `drop_bottom_by_volume` is disabled during parity (it should be, unless we set `--ignore-bottom-pct`).

## 3. Simulator internals
- **State initialization**: Compare Julia`s `SimulationState`/`Trader` setup (fees, tweak state, `D0`, `xcp`) with the C++ constructor path in `main-stableswap.cpp`. Ensure we copy the initial oracle prices from `price_vector_from_cpp_trades`.
- **Oracle & MA updates**: Walk through `Tweaks.jl` (e.g., `update_oracle!`, `ma_half_time` handling, `adjustment_step`) and align with the corresponding logic in `cryptopool-simulator/main-stableswap.cpp` / `stableswap.cpp`. Since chunk07913 diverges in `price_before`, focus on: `geometric_mean*`, `get_p` helpers, and any `Double64` -> `Float64` casts that might zero-out tweak state.
- **Boost / fee adjustments**: Review `apply_boost!`, `update_fee_gamma`, gas fee handling and compare with the C++ `apply_tweak`/`do_trade` flows. Mismatched toggles (e.g., `log` flag, `heavy_tx` heuristics) could inflate slippage.
- **Exchange functions**: Trace `exchange2!` / `exchange3!`, `solve_D`, `solve_x`, and related invariants to ensure we mirror the C++ solver’s branch conditions (especially early exits, iteration caps, and `price_oracle` usage).

## 4. Instrumentation & log comparison
- **Logger parity**: Confirm `SimulatorInstrumentation.jl` emits the same event keys as the C++ `CPPDBG` macros so that `scripts/diff_chunks.jl` aligns events correctly. For chunk07913, diff the Julia vs C++ `LEG` event around candle 64 to extract the precise reserve/price delta.
- **Diagnostics**: If needed, write a focused script to replay chunk07913 through both simulators step-by-step, dumping intermediate oracle/tweak state to correlate when the paths diverge.

## Execution order
1. Re-run `scripts/chunk_summary.jl chunk07913 --root=artifacts/chunks_ethusdt-1m_100 --keep-logs` to regenerate fresh logs after verifying datasets.
2. Follow sections 1→4 sequentially, checking off each subsystem after confirming parity or filing concrete issues.
3. Document findings per subsystem (data intake, preprocessing, simulator core, instrumentation) to build a clear chain of evidence leading to the slippage discrepancy.
