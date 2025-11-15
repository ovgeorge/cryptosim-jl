# Phase 2 – Isolate Domain Types

## Objectives
1. Introduce explicit types/aliases for shared concepts (e.g., `TradePair`, `LegStage`, `PriceVector`, `ChunkId`).
2. Wrap chunk metadata/config paths in structs to enforce file/dir invariants.

## Deliverables
- New domain module (`src/DomainTypes.jl`) exporting the core types plus constructors with validation.
- `DataIO`, `Preprocessing`, and `Simulator` signatures updated to use the new types where applicable.
- Utility struct (e.g., `ChunkPaths`) centralizing chunk-related file paths to reduce ad-hoc path building in scripts.
- Updated documentation (roadmap) noting adoption progress and pending gaps.

## Implementation Plan
1. **Create Domain Module**
   - Add `src/DomainTypes.jl`:
     - `const TradePair = NTuple{2,Int}`
     - `abstract type LegStage end`; concrete `struct Dir1 <: LegStage` etc., or use `@enum LegStage`.
     - `struct ChunkId` wrapping `String` (validated `startswith("chunk")`).
     - `struct ChunkPaths` with fields (`root`, `metadata`, `config`, `results`, `cpp_log`, `julia_log`, `data_dir`); constructor takes chunk dir and verifies existence.
     - `const PriceVector = Vector{Float64}` (leave flexibility via parametric alias).
   - Provide helper constructors: `ChunkPaths(root::AbstractString, id::AbstractString)` returning validated paths.
   - Export these from `DomainTypes` and re-export via `CryptoSim`.

2. **Adopt Types in DataIO / Preprocessing**
   - Update function signatures to accept/return `TradePair` instead of raw tuples.
   - Ensure `DataIO.build_cpp_trades` returns `CPPTrade` with `TradePair`.

3. **Adopt Types in Simulator / Instrumentation**
   - Update `current_price`, `log_*` helpers, `EventContext` to use `TradePair` type alias.
   - Consider `LegStage` usage in instrumentation (DIR1/DIR2) to ensure consistent symbol usage (`:dir1`/`:dir2`).

4. **Chunk Helper for Scripts**
   - Modify `scripts/chunk_summary.jl` to use `ChunkPaths` for locating metadata/config/logs.
   - Ensure new helper is used anywhere chunk paths are built manually (parity scripts).

5. **Docs & Roadmap**
   - Update `plans/refactor_roadmap.md` to link to this phase plan and record progress.

6. **Validation**
   - Run `julia --project=. scripts/chunk_summary.jl --root=... chunk00000` to ensure type changes don’t break behavior.
   - Optionally run `scripts/full_parity_report.sh` if time permits; otherwise, note in README/plan.

## Progress (current)

- Added `src/DomainTypes.jl` with `TradePair`, `LegStage`, `ChunkId`, `ChunkPaths`, and helpers, re-exported via `CryptoSim`.
- Updated `DataIO`, `Preprocessing`, `Simulator`, and `SimulatorInstrumentation` to consume `TradePair` (reducing raw tuple usage).
- Adopted `LegStage` across the simulator + instrumentation (LegConfig now type-safe, logging/trace helpers convert to canonical labels/symbols).
- Refactored `scripts/chunk_summary.jl` to leverage `ChunkPaths` for path validation/log naming.
- Verified `chunk00000` parity run completes with the new abstractions (`julia --project=. scripts/chunk_summary.jl ... chunk00000`).
