# CryptoSim V2

This repository contains the Julia reimplementation of the Curve CryptoPool simulator alongside a vendored copy of the original C++ solver under `cryptopool-simulator/`.

## Current Parity Baseline

The best-to-date Julia ⇄ C++ parity snapshot (2,148 chunks from `data/ethusdt-1m-full.json.gz`) is captured in `reports/ethusdt_full_parity_summary.md`. Treat those quantiles as the minimum acceptable quality bar—future code changes must not degrade those relative errors without an explicit justification and updated report.

Run `scripts/full_parity_report.sh` for a full “fire-and-forget” sweep: it calls the GNU-parallel runner to resummarize every chunk under `artifacts/chunks_ethusdt-1m-full/`, overwrites `reports/ethusdt_full_chunk_summary.jsonl`, and then invokes `scripts/parity_quantiles.jl` to print the quantile table. Both scripts are argument-free by default; override `CHUNK_ROOT`, `DATA_DIR`, `OUTPUT_PATH`, or `JOBS` via environment variables when needed.

## Data layout

The `data/` entry in this tree is a symbolic link to `/home/george/data`. Treat it as read-only: do not dump generated fixtures, decompressed candles, or any other throwaway artifacts there. Instead use `artifacts/` (ignored by git) for derived data such as decompressed candle feeds, chunk JSON files, and logs.

## Useful scripts

* `scripts/capture_chunks.sh` – capture a few `trimXXXXX` chunks from the C++ simulator (defaults to Curve's 2-coin config).
* `scripts/generate_chunks.py` – batch-produce chunk fixtures from a raw dataset while running both C++ solvers.
* `scripts/generate_chunks_parallel.py` – fan out the sequential generator across all CPU cores so chunk production (and simulator runs) stay fully saturated.
* `scripts/run_parity_parallel.sh` – run `chunk_summary.jl` for every chunk directory (works with `gnu parallel`).

See `reports/` for current divergence notes and diagnostic dumps.

## Tests

Run `julia --project=. test/runtests.jl` to execute lightweight sanity checks covering chunk path validation and simulator initialization. These tests ensure the refactored core modules load correctly before running heavier parity sweeps.

## Parallel chunk generation

Use `scripts/generate_chunks_parallel.py` whenever you need to materialize a large chunk set (for example, 100-candle windows from everything stored under `data/`). The wrapper keeps invoking `scripts/generate_chunks.py` under the hood but splits the work into contiguous ranges and automatically launches one worker per hardware thread—**we always want to use every core, always**. Override `--jobs` only when you explicitly need to throttle a run (such as on a shared box).

Example:

```bash
python scripts/generate_chunks_parallel.py \
  --dataset data/ethusdt-1m.json.gz \
  --config configs/ethusdt_chunk_config.json \
  --output artifacts/chunks_ethusdt-1m_100 \
  --chunk-size 100 \
  --instrumented-sim cryptopool-simulator/simu \
  --chunks-per-job 256 \
  --force
```

The gold solver is optional for now (pass `--gold-sim` once you have the pristine binary) and must never point to the same executable as `--instrumented-sim`.
