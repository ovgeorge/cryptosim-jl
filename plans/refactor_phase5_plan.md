# Phase 5 – Orchestration Cleanup & Deeper Tests

## Lessons From Recent Attempts
- **Instrumentation coupling:** The run loop, `SimulationState`, and helper logic all reach directly into `SimulatorInstrumentation.jl` constructs (`debug`, probe files, logger wiring). When we tried to move the state into a separate module the compile failed because those helpers were not available.
- **Implicit dependencies on `Trader`:** Multiple helpers mutate the trader and capture intermediate values inside the same file, making it hard to lift functions piecemeal.
- **Test coverage gaps:** Without trim-flag/`ChunkLoader` regression tests we lack a safety net, so the refactor attempts felt risky and we rolled them back.

These issues mean we need an incremental approach with clear seams before cutting over to a `SimulationRunner`.

## Revised Objectives
1. Carve out a minimal logging boundary (a `SimulatorLogger` helper) that owns `compute_debug_options`, adapters, and probe helpers so the rest of the runner can depend on a slim interface.
2. Extract the pure simulation state + loop in stages, first by introducing a light wrapper (`SimulationShell`) inside `Simulator.jl`, then migrating it into `SimulationRunner.jl`.
3. Expand the targeted tests (trimmed chunks, loader options, synthetic run loop) to guard each stage.
4. Keep scripts and tooling aligned with the new modules as soon as the public API changes.

## Strategy & Stages

### Stage 1 – Logging Boundary
- Create `src/SimulatorLogger.jl` (name TBD) that exposes `DebugOptions`, `TradeLogger`, `compute_debug_options`, `trace_step_*`, and helpers currently tied to `Simulator.jl`.
- Update `Simulator.jl` and `SimulatorInstrumentation.jl` to consume this module so `SimulationState` only depends on the logger interface, not concrete instrumentation internals.
- Adjust tests/scripts if they import the moved symbols directly.

### Stage 2 – Simulation Shell
- Introduce a `SimulationShell` struct in `Simulator.jl` that wraps the existing `SimulationState` fields but exposes accessor functions (`shell_logger`, `shell_metrics`, etc.).
- Move the highest-level orchestration (`run_cpp_trade_bundle!`, `run_split_trades!`) to operate on the shell and keep the low-level helpers in place.
- Add unit tests exercising the shell with synthetic data to confirm no regressions before touching module boundaries.

### Stage 3 – SimulationRunner Module
- Create `src/SimulationRunner.jl`, move `SimulationState`, the shell, and the orchestration helpers into it.
- Make `Simulator.jl` a façade that `include`s/`using`s the runner, re-exporting `SimulationState`, `run_exact_simulation!`, etc.
- Update scripts/tests to import through `Simulator` (or directly from `SimulationRunner` when internal use is needed).

### Stage 4 – Test & Script Hardening
- Expand `test/runtests.jl` with:
  - `ChunkLoader` trim flag + `ignore_bottom_pct` coverage.
  - Synthetic solver regression covering step logic and instrumentation toggles.
  - Smoke test for `SimulationRunner` on a mini chunk fixture.
- Run `scripts/chunk_summary.jl` against a few sentinel chunks to confirm the CLI still works, then rerun the 128-chunk parity sample.
- Refresh any documentation/readmes describing the refactor boundaries.

## Verification Checklist
- `julia --project=. test/runtests.jl` passes (with new cases).
- `julia --project=. scripts/chunk_summary.jl --root=artifacts/chunks_ethusdt-1m-full chunk00000` executes without regression.
- Parity sample (`scripts/full_parity_report.sh` equivalent) shows unchanged metrics within tolerances.
