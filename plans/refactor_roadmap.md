# Julia Refactor Roadmap

Goal: Simplify the Julia rewrite for readability and maintainability while preserving parity with the C++ simulator.

## Phase 1 – Structural Audit

### Objectives
1. Catalog every module’s public API and dependency footprint.
2. Surface duplication, dead code, and inconsistent naming/semantics.
3. Establish baseline documentation for current data flows (candle ingestion → trade splits → simulator → metrics/logging).

### Deliverables
- Dependency map (text/table) covering: `DataIO`, `Preprocessing`, `Simulator`, `SimulatorInstrumentation`, `Metrics`, `CLI`, scripts.
- List of duplication hotspots (e.g., multiple tweak states, redundant split logic).
- Notes on dead or unused code paths to remove in later phases.
- High-level data-flow diagram (markdown description) from chunk loading to report emission.

### Tasks
1. **Module Inventory**
   - For each module, list exported types/functions, external dependencies, and primary responsibilities.
   - Capture in a markdown table under this plan.

2. **Dependency Graph**
   - Identify directional dependencies (e.g., `Simulator` uses `DataIO` structs, `Metrics`). Note any cycles.

3. **Duplication Scan**
   - Search for repeated struct definitions, math helpers, and logging helpers; list file/line references.
   - Flag inconsistencies (e.g., `SplitTrade` variations, metrics accumulation code in multiple places).

4. **Dead Code Review**
   - Use `rg`/`julia --project` introspection to find unused exports; confirm via search before marking.

5. **Data Flow Summary**
   - Document the current chunk→simulation→report flow in prose (what script calls what, data artifacts produced).

6. **Report & Next Steps**
   - Compile findings into this markdown plan under new subheadings.
   - Outline specific refactor targets for Phase 2 based on the audit.

---

## Phase 1 Findings

### Module Inventory

| Module | Responsibilities | Key Exports | Depends On | Notes |
| --- | --- | --- | --- | --- |
| `DataIO` (`src/DataIO.jl`) | Load configs/candles, produce CPP-compatible trade bundles, enforce dataset paths | `Candle`, `SimulationConfig`, `ConfigFile`, `CPPTrade`, `load_config`, `load_candles`, `build_cpp_trades`, `initial_price_vector`, `DEFAULT_DATA_DIR`, `split_leg_extrema` | `JSON3`, `StructTypes`, optional `DoubleFloats` | Contains both candle parsing and trade splitting helpers; large single file mixing IO and math heuristics. |
| `Preprocessing` (`src/Preprocessing.jl`) | Convert candles/CPP trades into canonical `SplitTrade`s, preserve C++ ordering | `SplitTrade`, `split_candles`, `adapt_trades` | `DataIO` types, `JSON3` (for debug logging) | Keeps multiple code paths for splits (candles vs. CPP logs); logging branch via env vars. |
| `Simulator` (`src/Simulator.jl`) | Core StableSwap math, Trader state machine, metrics accumulation, instrumentation hooks | `Money`, `CurveState`, `SimulationState`, `run_exact_simulation!`, math helpers, Instrumentation submodule re-export | `DataIO` (`SimulationConfig`, `CPPTrade`), `Preprocessing`, `Metrics`, instrumentation module, `DoubleFloats` | ~1200 lines; interleaves math primitives, tweak/oracle logic, logging calls, and orchestration. |
| `SimulatorInstrumentation` (`src/SimulatorInstrumentation.jl`) | Structured logging (STEP/LEG/PRELEG/TWEAK), debug probes, JSON serialization | `TradeLogger`, `DebugOptions`, `log_*` helpers, `TraderSnapshot` | `JSON3`, `StructTypes`, `Preprocessing.SplitTrade` | API tightly coupled to simulator; events expressed as concrete structs but emitted via `emit_event`. |
| `Metrics` (`src/Metrics.jl`) | Track volume/slippage/liq density/APY | `MetricAccumulator`, `push_volume!`, `push_slippage!`, `set_apy!`, `summarize` | None (base Julia only) | Simple and isolated; invoked directly from simulator loop. |
| `CLI` (`src/CLI.jl`) | Parse legacy `simu` CLI flags (`trim`, `threads`, in/out files) | `CLIOptions`, `parse_cli_args`, `DEFAULT_TRIM` | None | Still modeled after C++ binary; only used by `CryptoSim.run_cli`. |
| `CryptoSim` (`src/CryptoSim.jl`) | Aggregates modules, exports top-level API | `run_cli`, module exports | All modules above | Lightweight wrapper. |
| Scripts (`scripts/*.jl`, `*.sh`, `*.py`) | Chunk generation, summarization, diffing, parity automation | `chunk_summary.jl`, `diff_chunks.jl`, `generate_chunks*.py`, `run_parity_parallel.sh`, `full_parity_report.sh`, `parity_quantiles.jl` | Julia modules, Python stdlib, GNU parallel | Multiple entry points replicate chunk-loading logic (metapaths, config resolution). |

### Dependency Notes

- `Simulator` → `Preprocessing` (for `SplitTrade`), `DataIO` (config structs), `Metrics`, `Instrumentation`.
- `DataIO` is standalone (depends only on external libs) and used by `Preprocessing`, `Simulator`, scripts.
- `Preprocessing` depends on `DataIO`, but `DataIO` does not depend back—acyclic.
- `SimulatorInstrumentation` depends on `Preprocessing` and `Simulator` (imports `Trader`). Creates tight coupling: instrumentation knows about trader internals.
- Scripts (`chunk_summary.jl`, etc.) `include` modules rather than using a package; they reconstruct load paths manually.

### Duplication & Inconsistencies

- **Trade Splitting**: `DataIO.build_cpp_trades` and `Preprocessing.split_candles` share similar logic for timestamp offsets and leg ordering, but implemented separately (one returns `CPPTrade`, the other `SplitTrade`). Opportunity for shared helpers.
- **Chunk Loading**: `scripts/chunk_summary.jl` and `scripts/generate_chunks.py` both resolve metadata/config files independently, leading to repeated file-walking logic.
- **Logging**: C++ and Julia instrumentation structures mirror each other but with slightly different field naming/validation; no shared schema definition.
- **Math Helpers**: Functions like `geometric_mean2/3`, reduction coefficients, price calculations live inside `Simulator.jl` but are used in multiple places; no dedicated math namespace.
- **Env-driven Debugging**: `Preprocessing` and `SimulatorInstrumentation` both read env vars for logging; semantics differ (`JULIA_SPLIT_DEBUG` vs. `JULIA_TRADE_DEBUG`) without central registry.
- **Chunk metadata handling** scattered across scripts/tests; no unified `Chunk` abstraction encapsulating config, data path, price vector, results.

### Dead / Unused Code

- Big clean-up already removed the unused `Tweaks` module. No other completely unused modules found, but:
  - `CLI.parse_cli_args` is only exercised when mimicking the legacy binary; consider whether it’s still required.
  - `Preprocessing`’s JSON logging path (`JULIA_SPLIT_DEBUG`) is developer-only; verify if we can move it behind a structured logger later.

### Data Flow Summary

1. **Chunk Generation** (`scripts/generate_chunks_parallel.py` → `generate_chunks.py`):
   - Reads raw dataset (`data/ethusdt-1m-full.json.gz`), slices candles into chunk JSON files, runs C++ simulator twice to capture `results.json` + `cpp_log.jsonl`, writes metadata.
2. **Chunk Summaries** (`scripts/chunk_summary.jl`):
   - Loads chunk metadata/config, reconstructs CPP trades from raw data via `DataIO`, adapts them to `SplitTrade`s, runs `Simulator.run_exact_simulation!`, compares metrics vs. `results.json`, dumps Julia logs, optionally diffs against C++ logs.
3. **Parity Sweep** (`scripts/run_parity_parallel.sh` / `full_parity_report.sh`):
   - Invokes `chunk_summary.jl` for each chunk (optionally sampled), aggregates JSONL rows into `reports/ethusdt_full_chunk_summary.jsonl`.
4. **Quantile Reporting** (`scripts/parity_quantiles.jl`):
   - Reads JSONL summary, computes metric quantiles, prints table for regression tracking.

Artifacts produced: chunk directories (metadata/config/results/logs), parity summaries (`reports/*.jsonl`), markdown snapshot (`reports/ethusdt_full_parity_summary.md`).

### Next Steps (Feeds Phase 2)

Based on the audit:

1. **Define shared domain types** (`Chunk`, `TradePair`, `LegStage`, `PriceVector`) to remove repeated tuple/Dict handling in scripts/modules.
2. **Extract math/util modules** from `Simulator.jl` so subsequent refactors can reason about solver logic separately from plumbing.
3. **Plan instrumentation abstraction** so `Simulator` doesn’t import instrumentation details directly; align with future logging interfaces.
4. **Centralize chunk loading** into a reusable helper (used by scripts and tests) to eliminate duplicated metadata parsing.

These targets will anchor Phase 2 (“Isolate Domain Types”) and set up later decomposition work.

## Phases

1. **Audit Current Structure**
   - Map module APIs (`DataIO`, `Preprocessing`, `Simulator`, `Metrics`, instrumentation, CLI) and cross-dependencies.
   - Identify duplicated concepts, dead helpers, and inconsistent naming. Log findings for downstream fixes.

2. **Isolate Domain Types**
   - Introduce explicit types/aliases for shared concepts (e.g., `LegStage`, `TradePair`, `PriceVec`).
   - Wrap config/metadata blobs in structs with constructors enforcing invariants.
   - See `plans/refactor_phase2_plan.md` for detailed objectives/tasks.

3. **Functional Decomposition**
   - Split `Simulator.jl` into cohesive submodules: math primitives, trader state, metrics, orchestration loop.
   - Encapsulate logging via lightweight interfaces to avoid instrumentation leakage.
   - Detailed plan: see `plans/refactor_phase3_plan.md`.

4. **Shared Utilities**
   - Extract math helpers into a `MathUtils` module reused across components.
   - Build a `Chunk` abstraction holding metadata, config, price vector, and trades.
   - Detailed plan: see `plans/refactor_phase4_plan.md`.

5. **Improve Trade Pipeline**
   - Standardize on a single `SplitTrade` path for both C++ bundles and Julia-generated splits.
   - Provide iterator-style APIs (e.g., `each_leg(chunk)`) to simplify simulator loops.

6. **Metrics & Reporting**
   - Move metric accumulation into a dedicated module with composable reducers.
   - Redesign parity reporting to produce `ChunkSummary` structs before JSON serialization.

7. **Testing & Validation Hooks**
   - Add unit tests for math helpers, chunk loaders, and instrumentation glue.
   - Capture sample chunks in `test/fixtures` with expected outputs.

8. **Incremental Execution**
   - After each phase, rerun parity (`scripts/full_parity_report.sh`) to guard against regressions.
   - Track progress and findings in this plan file.
