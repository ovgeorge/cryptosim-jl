# CryptoSim V2

Julia reimplementation of Curve’s CryptoPool simulator plus a vendored copy of the original C++ solver in `cryptopool-simulator/`. The repo holds the parity tooling, scripts for producing chunk fixtures, and the reporting pipeline that keeps Julia and C++ in lockstep.

## Architecture at a glance

- **Domain + IO (`DomainTypes`, `DataIO`, `Preprocessing`)** – define shared structs (`ChunkId`, `SplitTrade`, config readers) and split raw candles into deterministic DIR1/DIR2 legs so Julia consumes the same sequence as the C++ binary.
- **Pool core (`SimulatorCore`, `SimulatorMath`)** – contains `Trader`, `CurveState`, fee/tweak/profit state, and the invariant math (`solve_D`, `curve_y`, `exchange{2,3}!`, `price{2,3}`) that mutates pool reserves exactly like the legacy solver.
- **Trading loop (`Simulator`, `SimulationRunner`)** – wires the core math into `step_for_price`, `execute_trade!`, `tweak_price!`, and `run_split_trades!`. Each leg probes profit tolerances, executes the swap, applies boosts, updates moving-average targets, and accumulates slippage/volume/APY metrics.
- **Instrumentation & metrics (`SimulatorInstrumentation`, `SimulatorLogger`, `Metrics`)** – emit structured PRELEG/STEP/LEG/TWEAK logs, optional step traces/probes, and aggregate per-chunk metrics so parity diffs line up with the C++ JSON logs.
- **Tooling (`ChunkLoader`, `ChunkSummary`, `CLI`)** – shared helpers used by the summarizer scripts and tests; everything exposes the `CryptoSim` module so downstream code can simply `using CryptoSim`.

## Data & artifacts

`data/` is a symlink to `/home/george/data` and must stay read-only. Place every generated chunk, decompressed dataset, or log under `artifacts/` (already git-ignored). Scripts assume chunk roots are laid out as `artifacts/chunks_<dataset>/chunkXXXXX`.

## Chunk generation

Materialize fixtures with the parallel wrapper so all CPU cores stay busy:

```bash
python scripts/generate_chunks_parallel.py \
  --dataset data/ethusdt-1m-full.json.gz \
  --config configs/ethusdt_chunk_config.json \
  --output artifacts/chunks_ethusdt-1m-full \
  --chunk-size 2000 \
  --instrumented-sim cryptopool-simulator/simu \
  --chunks-per-job 256 \
  --force
```

That script fans out `generate_chunks.py`, runs both C++ solvers if configured, and refreshes the chunk manifest (dataset name/path/SHA, chunk size, emitted IDs). Use `--jobs` only when you need to throttle; otherwise the wrapper launches `os.cpu_count()` workers automatically.

## Full parity sweep

The canonical parity report lives in `reports/ethusdt_full_parity_summary.md` (currently 2,148 chunks from `data/ethusdt-1m-full.json.gz`). Regenerate it via:

```bash
scripts/full_parity_report.sh
```

The helper prints its config, calls `scripts/run_parity_parallel.sh` to run `scripts/chunk_summary.jl` over every `chunk*` directory with GNU `parallel`, writes a fresh `reports/ethusdt_full_chunk_summary.jsonl`, and finally runs `scripts/parity_quantiles.jl` to emit the quantile table + markdown snapshot. Override `CHUNK_ROOT`, `DATA_DIR`, `OUTPUT_PATH`, `REPORT_MARKDOWN`, or `JOBS` via flags or environment variables, and use `--note`/`--runner` arguments to document provenance.

## Tests

Run `julia --project=. test/runtests.jl` before heavy sweeps. The suite sanity-checks chunk path validation, simulator initialization, and other guard rails so regressions are caught quickly.
