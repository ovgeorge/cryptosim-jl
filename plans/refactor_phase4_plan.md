# Phase 4 – Shared Utilities & Chunk Abstractions

## Objectives
1. Centralize chunk loading/metadata helpers so scripts (`chunk_summary.jl`, `compare_chunks.jl`, `replay_single_candle.jl`, etc.) reuse the same code path.
2. Provide a reusable API for reading expected metrics/results and ensuring data sources (currently duplicated in multiple scripts).
3. Lay groundwork for future scripting/testing by exposing these helpers via the `CryptoSim` namespace.

## Deliverables
- New module (`src/ChunkLoader.jl`) exporting functions such as:
  - `load_chunk(paths::ChunkPaths; data_dir)` returning a struct/NamedTuple with `cfg`, `price_vec`, `trades`, and metadata.
  - `read_expected(paths::ChunkPaths)` returning the CPP metrics.
  - `ensure_data_sources!` lifted out of scripts.
- Scripts updated to consume the new helpers instead of inlined logic.
- Documentation/tests referencing the new shared utilities.

## Implementation Plan
1. **Create ChunkLoader module**
   - Define `struct ChunkData` capturing config, price vector, trades, metadata.
   - Implement `load_chunk`, `read_expected`, and `ensure_data_sources!` in this module.
   - Include the module in `CryptoSim.jl` and re-export it as `CryptoSim.ChunkLoader`.

2. **Refactor scripts**
   - Update `scripts/chunk_summary.jl`, `scripts/compare_chunks.jl`, `scripts/replay_single_candle.jl` to call the shared helpers.
   - Remove redundant functions (`ensure_data_sources!`, ad-hoc JSON reads) from scripts.

3. **Docs & Tests**
   - Add tests in `test/runtests.jl` covering `ChunkLoader.load_chunk` on `chunk00000`.
   - Briefly mention the shared loader in `README.md` (optional).

4. **Validation**
   - Re-run sample parity scripts (`chunk_summary.jl chunk00000`, `compare_chunks.jl chunk00000`) after the refactor.
   - Run `julia --project=. test/runtests.jl`.

## Progress

- Implemented `src/ChunkLoader.jl` (`ChunkData`, `load_chunk`, `read_expected`, `ensure_data_sources!`) and re-exported it via `CryptoSim`.
- `scripts/chunk_summary.jl`, `scripts/compare_chunks.jl`, and `scripts/replay_single_candle.jl` now consume the shared loader + `DomainTypes.ChunkPaths`, eliminating duplicated path/config logic.
- Added a dedicated testset in `test/runtests.jl` to exercise the loader on `chunk00000`, and documented the test command in `README.md`.
