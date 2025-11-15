# Julia ⇄ C++ Parity Plan (v2)

## 1. Oracle / EMA lifecycle
- Remove every eager call to `update_price_oracle!` inside the trading loop (`run_split_trades!`). `Trader.tweak.price_oracle` should only move inside `ma_recorder!` when `tweak_price!` runs after the candle is complete, mirroring `Trader::ma_recorder` in `main.cpp:1573-1679`.
- Keep using the existing `lasts` dictionary to seed PRELEG limits and midpoint calculations; that already mirrors the C++ `lasts` map, so no new buffers are required.
- Ensure midpoint/high/low capture still works by updating the `last_quote` locals instead of the oracle array when DIR1/DIR2 executes.

## 2. `step_for_price_3` volume gating
- The 3-asset solver should budget against the target curve price (`curve.p[_to]`) exactly like the 2-asset solver and the C++ code (`main.cpp:1247`). Today it multiplies by the EMA (`tweak.price_oracle`) which makes DIR1/DIR2 exit conditions drift. Switch both grow and shrink loops to use `ctx.curve.p[to]` when accumulating `v`.

## 3. Per-trade timing (`trade_dt`)
- C++’s `last_time` is updated every time a `trade_data` entry is processed (i.e., on both halves of the split candle). Julia only advances `prev_trade_ts` when `trade.is_last`, so the first split of every candle sees a zero Δt and under-weights slippage.
- Track two clocks: (a) `prev_trade_ts`, advanced on every split so `record_leg_metrics!` receives the real Δt, and (b) `prev_candle_ts`, advanced only when `trade.is_last` so boost/APY continue to accumulate over the same horizon as before. Clamp negative deltas to zero so we never feed negative weights into the metrics.

## 4. Verification
1. Re-run `scripts/run_parity_parallel.sh --root=artifacts/tricrypto_chunks --jobs=$(nproc) --output reports/tricrypto_chunk_summary.jsonl`.
2. Pick a divergent chunk (e.g., `chunk00010`) and diff PRELEG/LEG logs via `scripts/diff_chunks.jl chunk00010` to confirm the oracle + solver changes closed the gaps.
3. If residual mismatches remain, inspect timestamps/boost logs to ensure the cached `trade_dt` is wired correctly.
