# CryptoSim V2

This repository contains the Julia reimplementation of the Curve CryptoPool simulator alongside a vendored copy of the original C++ solver under `cryptopool-simulator/`.

## Data layout

The `data/` entry in this tree is a symbolic link to `/home/george/data`. Treat it as read-only: do not dump generated fixtures, decompressed candles, or any other throwaway artifacts there. Instead use `artifacts/` (ignored by git) for derived data such as decompressed candle feeds, chunk JSON files, and logs.

## Useful scripts

* `scripts/capture_chunks.sh` – capture a few `trimXXXXX` chunks from the C++ simulator (defaults to Curve's 2-coin config).
* `scripts/generate_chunks.py` – batch-produce chunk fixtures from a raw dataset while running both C++ solvers.
* `scripts/run_parity_parallel.sh` – run `chunk_summary.jl` for every chunk directory (works with `gnu parallel`).

See `reports/` for current divergence notes and diagnostic dumps.
