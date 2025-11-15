# Phase 3 – Functional Decomposition & Math Utilities

## Objectives
1. Split solver logic into focused components (math primitives, trader state, orchestration) to improve readability.
2. Extract shared math helpers into a reusable module (`MathUtils.jl`) so both simulator and scripts share consistent implementations.
3. Reduce instrumentation/logging bleed-through by introducing interfaces or adapter structs.

## Deliverables
- New module(s) (e.g., `src/MathUtils.jl`, `src/SimulationCore/`) housing invariant math, pricing helpers, and trader operations.
- `Simulator.jl` slimmed down to orchestration/glue code that composes the extracted pieces.
- Updated instrumentation hooks operating through small interfaces (e.g., `LoggerAdapter`) instead of referencing `Instr` globally.
- Documentation updates in the roadmap summarizing structural changes and new module boundaries.

## Implementation Plan
1. **Introduce Math Utilities**
   - Create `src/MathUtils.jl` encapsulating:
     - Geometric means, invariant solvers (`geometric_mean2/3`, `reduction_coefficient*`, `solve_D`, `solve_x`).
     - Fee helpers (`fee2`, `fee3`) and clamp utilities.
   - Replace direct definitions in `Simulator.jl` with `using .MathUtils`.

2. **Refactor Trader/Curve Logic**
   - Move `CurveState`, `Trader`, and related step/tweak structs into a `SimulatorCore` module (e.g., `src/SimulationCore.jl`).
   - Keep `SimulationState` and `run_exact_simulation!` orchestrating the core module.

3. **Instrumentation Adapter**
   - Define a small interface/protocol (e.g., `struct SimulationLogger`) that wraps the existing instrumentation so the core logic depends on an abstract logger, not the concrete `Instr` module.
   - Provide default adapter that forwards to `Instr`, but allow tests/benchmarks to inject a no-op or custom logger.

4. **Update Imports & Exports**
   - Ensure `CryptoSim` includes/re-exports new modules.
   - Update scripts/tests to use the reorganized module tree.

5. **Documentation**
   - Record new structure in `plans/refactor_roadmap.md`.
   - Briefly explain new modules in `README.md` (optional but recommended).

6. **Validation**
   - Re-run targeted parity check (`scripts/chunk_summary.jl` on a sample chunk) to ensure behavior unchanged.
   - Optionally run a sampled `scripts/full_parity_report.sh` (e.g., `SAMPLE_CHUNKS=32`) to verify no regressions.

## Notes
- Prioritize mechanical moves first (extract math helpers) before attempting deeper re-architecture.
- Keep commits incremental: math extraction → core split → instrumentation adapter.

## Progress

- Added `src/SimulatorMath.jl`, a nested module housing geometric means, invariant solvers (`solve_D`, `solve_x`), fee helpers, curve helpers (`curve_y`, `price2/3`), and boost utilities. `Simulator.jl` now `include`s/`using`s this module, removing the long math block from the main file.
- Verified parity on a representative chunk (`chunk00000`) after the extraction to ensure no behavioral regressions yet.
- Introduced `SimulatorCore.jl` for the core data structures (`CurveState`, `Trader`, fee/tweak/profit/metrics state, `StepContext`) and wired `Simulator.jl` to import/re-export them.
- Added a `LoggerAdapter` wrapper so the main simulation loop no longer references `Instrumentation` directly; all logging goes through adapter helpers.
- Updated `scripts/compare_chunks.jl` and `scripts/replay_single_candle.jl` to rely on `DomainTypes.ChunkPaths`, ensuring consistent path validation/log lookup across tooling.
- Added smoke tests (`test/runtests.jl`) that exercise `DomainTypes.ChunkPaths` and basic simulator initialization to guard the refactor.

## Remaining Work

1. Finalize the `SimulatorCore` move:
   - Remove duplicate struct definitions from `Simulator.jl` so `SimulatorCore` is the canonical source (avoids constructor extension warnings).
   - Ensure `SimulationState`/run loop live in a lightweight orchestration module (or keep them in `Simulator.jl` but importing from `SimulatorCore` cleanly).
2. Update tooling to the new abstractions:
   - `scripts/compare_chunks.jl`, `replay_single_candle.jl`, and similar scripts should reuse `DomainTypes.ChunkPaths` and the logger adapter instead of duplicating path logic/instrumentation hooks.
3. Documentation/test follow-up:
   - Update the roadmap once `SimulatorCore` consolidation is complete.
   - Add small tests (or scripted parity samples) verifying the new modules continue to match C++.
