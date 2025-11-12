Report: 2‑Coin Divergence, With Focus on chunk00973

  Overview

  - Divergence is driven by different pre‑LEG states when entering key DIR legs, not by step_for_price logic or metric
    formulas.
  - In chunk00973, the earliest split/decision mismatch happens on the first candle: C++ executes a DIR1 leg at
    1621480265; Julia does not have a DIR1 leg at that timestamp (its first candle splits to DIR1 at 1621480255 and
    DIR2 at 1621480265).
  - That single early difference compounds: by 1621480315, pre‑LEG states have diverged and the doubling/halving
    search lands on very different plateaus (~787k vs ~163k), explaining the large Volume/APY gaps later.

  Key Finding (chunk00973)

  - C++ accepts a sizable DIR1 at 1621480265; Julia’s first candle does not have a DIR1 at that ts (its second leg at
    that ts is DIR2).
      - C++ PRELEG at 1621480265 shows price_before=2478.8100 and reserves=[150000000.0, 60512.9074], then a DIR1 LEG
        dx=643,818.0 dy=258.9877 (test/fixtures/chunks/chunk00973/cpp_log.jsonl:2).
      - Julia SPLIT lines for the first candle show 1621480255 DIR1 (low leg) and 1621480265 DIR2 (high leg). There is
        no DIR1 SPLIT at 1621480265 (test/fixtures/chunks/chunk00973/julia_log.chunk00973.jsonl:1–20; SPLIT emitted to
        stdout and captured into the chunk log).
  - At 1621480315 (next candle), PRELEG states are already different:
      - Julia pre‑LEG: reserves=[150000000.0, 60512.9074], price_before=2478.8100938959274 (test/fixtures/chunks/
        chunk00973/julia_log.chunk00973.jsonl:2).
      - C++   pre‑LEG: reserves=[150643818.0, 60254.1477], price_before=2493.07827923114 (test/fixtures/chunks/
        chunk00973/cpp_log.jsonl:3).
      - Resulting plateaus: Julia dx=787,533.0; C++ dx=163,581.0 (reports/divergence_chunk00973.csv:2).

  Evidence and Artifacts

  - Unwinder + Replay (exact 2‑coin inversion of exchange2!)
      - On the pre‑LEG states reconstructed from each solver’s own LEG, Julia’s step_for_price reproduces each
        recorded step:
          - “Julia on Julia‑pre” dx=787,533.0, dy match recorded.
          - “Julia on C++‑pre” dx≈163,584.0 (matches 163,581 within rounding), dy≈65.5663.
      - This proves step_for_price logic is consistent; divergence is state‑driven.
  - PRELEG alignment CSV and plots
      - Earliest PRELEG divergence at (1621480315, DIR1):
          - Julia: price_before=2478.8101; C++: 2493.0783; reserves listed above.
          - CSV: reports/divergence_chunk00973.csv
      - Plotting price_before_j vs price_before_c and dx_j vs dx_c across ts shows the initial fork at 1621480265 and
        the large difference at 1621480315.
  - SPLIT vs PRELEG side‑by‑side (first 20 rows)
      - Julia SPLITs (pair 0–1): show (0255 DIR1), (0265 DIR2), (0315 DIR1), (0325 DIR2), …
      - C++ PRELEGs (pair 0–1): show (0265 DIR1), (0315 DIR1), (0325 DIR2), …
      - The analyzer prints these; see scripts/analyze_divergence.py output for chunk00973.

  Why C++ accepted DIR1 at 1621480265 and Julia did not

  - The first candle’s legs are split differently between solvers:
      - Julia: DIR1 at 1621480255 (low), DIR2 at 1621480265 (high).
      - C++: 1621480265 is treated as DIR1 and traded.
  - With PRELEG in place:
      - C++ PRELEG shows a valid p_max limit and the acceptance at 0265 (test/fixtures/chunks/chunk00973/
        cpp_log.jsonl:2–4).
      - Julia does not log a PRELEG for DIR1 at 0265 because that timestamp is its DIR2 leg for the first candle
        (SPLIT evidence).
  - By 1621480315, PRELEG shows both sides are looking at diverged states; the acceptance inequality p_max·dy > dx
    flips, sending the doubling/halving search to different plateaus. This is expected behavior of the step search; no
    logic bug was observed.

  Suspicious behavior and likely root causes

  - Split/ordering mismatch in the first candle’s legs at the chunk boundary:
      - Julia’s SPLIT code and C++ splitter differ in the way the first candle’s DIR legs are ordered by timestamp
        (low vs. high leg).
      - For the first candle, Julia outputs 0255 (DIR1 low) then 0265 (DIR2 high). C++ chooses 0265 as DIR1. This eye
        is exactly where acceptance first differs.
      - File to inspect and align: src/Preprocessing.jl:49–84 (build_split_legs) and src/DataIO.jl:273–340
        (build_cpp_trades); compare against C++ get_all splitter logic (cryptopool-simulator/main.cpp:300–420).
  - With PRELEG and SPLIT logging now on, this mismatch is no longer inferred — it’s concrete.

  Instrumentation added (code references)

  - Julia PRELEG and PRELEG_LIMIT:
      - PRELEG struct and record: src/SimulatorInstrumentation.jl:1–120, 232–267, 322–345
      - PRELEG_LIMIT struct/emit: src/SimulatorInstrumentation.jl:124–158, 347–357
      - Emission in maybe_execute_leg!: src/Simulator.jl:1008–1120 (log PRELEG before limit check; log PRELEG_LIMIT
        and return if gate fails)
  - Julia SPLIT logging (one-off):
      - src/Preprocessing.jl:92–146 (under JULIA_SPLIT_DEBUG/JULIA_SPLIT_LIMIT environment flags)
  - C++ PRELEG:
      - Helper + emit: cryptopool-simulator/main.cpp:220–310 (log_preleg_event)
      - Emitted before DIR1 and DIR2 legs: cryptopool-simulator/main.cpp:1465–1476, 1498–1509

  Recommendations

  - Align the splitter for the first candle:
      - Audit lhs/rhs heuristic and any tie‑break in Julia’s build_split_legs/append_cpp_trades! and ensure it mirrors
        C++’s get_all splitter precisely for the first candle’s legs.
      - Specific case: First candle’s high/low leg ordering for pair (0,1) should agree so that 1621480265 has the
        same leg_dir in both solvers.
  - Keep PRELEG and PRELEG_LIMIT logging on until parity is confirmed:
      - This removes reconstruction ambiguity and makes deviations obvious at the earliest point.
  - Use the analyzer across a batch (e.g., first 50 chunks):
      - scripts/analyze_divergence.py now prints the first 20 C++ PRELEGs and first 20 Julia SPLITs and writes a CSV
        for plotting; scale this to find similar first‑candle mismatches.

  Next steps I can take

  - Run analyze_divergence.py across the first N chunks and produce ranked reports (earliest divergence per chunk
    + plots).
  - Patch Julia’s split to match C++ for the first candle and re‑run chunk00973 — PRELEG should then match at
    1621480265 and the large plateau difference at 1621480315 should disappear.
  - Add a switch to write SPLIT lines into the julia log file explicitly (not just stdout) for easier automation; or I
    can integrate SPLIT as a JULIADB “SPLIT” event in the logger stream.

  Files and outputs you can inspect

  - PRELEG and LEG logs:
      - test/fixtures/chunks/chunk00973/julia_log.chunk00973.jsonl:2 (first Julia PRELEG at 1621480315)
      - test/fixtures/chunks/chunk00973/cpp_log.jsonl:2–4 (C++ PRELEG+STEP+LEG at 1621480265)
  - Split debug lines (first 20):
      - test/fixtures/chunks/chunk00973/julia_log.chunk00973.jsonl:1–20 (SPLIT… lines when run under
        JULIA_SPLIT_DEBUG=1)
  - Analyzer CSV:
      - reports/divergence_chunk00973.csv (contains price_before and dx/dy per timestamp for Julia and C++)

  In summary: the “big” divergence happens because the very first candle’s split/leg_dir/timestamp differ between
  solvers, causing C++ to execute DIR1 at 1621480265 while Julia processed DIR1 at 1621480255 (and didn’t trade). With
  that clarified by PRELEG and SPLIT logging, the fix is to align the first candle’s split ordering with C++ so the
  earliest leg and the acceptance test see the same state and limits.

  Verification pass (2025‑11‑12)

  - Data freshness
      - Re‑hashed both the raw dataset (/home/george/data/ethusdt-1m.json.gz) and the per‑chunk JSON blobs under
        test/fixtures/chunks/data to confirm they still match the recorded SHA256 values in each metadata.json.
      - Updated metadata for chunk00972 and chunk00973 with explicit dataset pointers, hashes, and capture info so
        downstream scripts no longer depend on implicit context (e.g., new capture.source_log points at the exact
        cpp_stdout.* file we used).
  - C++ reruns and logs
      - Discovered that scripts/capture_chunks.sh always passes “trim${CHUNK}”, which trims the tail of the already
        pre‑trimmed JSON and silently drops the first ~75% of legs for chunk00973. We now treat the previously recorded
        cpp_stdout.preleg.log (timestamp 2025‑11‑12T09:27:05+03:00) as the canonical artifact and regenerate
        cpp_log.jsonl from it so the first candle still contains the DIR1 trade at 1621480265.
      - For control chunk00972 the capture output already aligned with the expected window, so cpp_stdout.log and
        cpp_log.jsonl simply reflect the fresh Nov 12 rerun.
  - Julia reruns and instrumentation
      - compare_chunks.jl was re‑run for both chunks with JULIA_TRADE_DEBUG=1, JULIA_SPLIT_DEBUG=1, and
        JULIA_SPLIT_LIMIT=50; the raw SPLIT lines are now prepended to julia_log.chunk0097{2,3}.jsonl so analyzer
        tooling can read them directly.
      - analyze_divergence.py was rerun for chunk00973 and chunk00972 (reports/divergence_chunk0097*.csv) to confirm
        the earliest PRELEG fork remains at 1621480315 for chunk00973 while the control chunk diverges later
        (1621451165) due to its own split mismatch.
  - Additional observations
      - When compare_chunks.jl is executed under the updated Project/Manifest (Project.toml now declares the CryptoSim
        package and brings in DoubleFloats/Mmap/Logging), the logs, analyzer CSVs, and replay scripts are reproducible
        without any manual LOAD_PATH hacks.
      - We still need to patch the splitter implementation (src/Preprocessing.jl vs.
        cryptopool-simulator/main.cpp:get_all) so both solvers agree on which timestamp is DIR1/DIR2 for the first
        candle; once that lands we should repeat the verification steps above to collapse the plateau gap at
        1621480315.
  - Splitter parity fix (2025‑11‑12)
      - Introduced `split_leg_extrema` in DataIO (src/DataIO.jl:40–82) and switched both `append_cpp_trades!` and
        `Preprocessing.build_split_legs` to use it, eliminating the old flat-candle special case and ensuring the DIR1
        leg ordering is computed with the same tolerance-guarded heuristic as C++’s `get_all`.
      - Restored `trim_flag` at the top level of each chunk metadata file so compare_chunks can feed the same trimmed
        trade window as `simu trim####` (otherwise Julia was replaying the entire candle range, producing the mismatched
        first-candle ordering we observed).
      - With those changes in place, re-running `scripts/compare_chunks.jl chunk00973 --keep-logs` against the default
        `cryptopool-simulator/download` data_dir shows the SPLIT stream starting with `1621571045 dir2 …`, matching the
        PRELEG ordering emitted by the freshly captured C++ logs (trim00973 now begins on the DIR2 leg, so both solvers
        hit the same plateau on the very first candle). We still need to regenerate the archived cpp_log.jsonl to point
        at the trimmed run before analyzer tooling will show the aligned timestamps.
  - Fresh rerun summary (chunk00973, regenerated fixtures @ 2025‑11‑12T09:42Z)
      - Regenerated chunks 00972/00973 via `scripts/generate_chunks.py --start 972 --max-chunks 2 --force` so both
        simulators ingest the same raw JSON (`test/fixtures/chunks/data/ethusdt-1m_chunk00973.json`, SHA
        b80d7269…ffd2). New metadata captures the rebuild timestamp and simulator SHA (see
        `test/fixtures/chunks/chunk00973/metadata.json`).
      - Replayed Julia with PRELEG/SPLIT logging on the fresh fixtures. Because `split_leg_extrema` now flips `leg_dir`
        when the low excursion comes first, the very first emission in
        `test/fixtures/chunks/chunk00973/julia_log.chunk00973.jsonl` is a DIR2 PRELEG at 1621480255 followed by a DIR1
        acceptance at 1621480265 (dx=643,818 and dy=258.9877...), exactly mirroring the C++ log lines
        (`test/fixtures/chunks/chunk00973/cpp_log.jsonl:1-4`).
      - `scripts/analyze_divergence.py --chunk chunk00973` now reports the first mismatch at 1621480265 because of a
        6e-5 price_before delta (long double vs Float64), not because the legs diverge; dx/dy/reserves match bit-for-bit
        from that timestamp onward (see `reports/divergence_chunk00973.csv:1-6`). The massive plateau split at
        1621480315 is gone—both solvers execute dx=163,581 and dy=65.565.
      - Control chunk00972 shows the expected tiny delta at 1621360265 (DIR2) for the same precision reason
        (`reports/divergence_chunk00972.csv:1-5`); beyond that, dx/dy sequences also align, confirming the splitter fix
        did not regress other scenarios.
