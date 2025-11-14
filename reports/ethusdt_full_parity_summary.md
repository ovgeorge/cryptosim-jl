# ETH/USDT 1m Parity Snapshot

Best-to-date Julia ⇄ C++ parity run using the full Binance ETH/USDT 1‑minute feed.

- **Dataset**: `data/ethusdt-1m-full.json.gz` (sha256 `bd052fd34b5dde524fc06b0be2124b025f14ee37a22ac32d672a677ecbc6b5ef`)
- **Chunk generator**: `scripts/generate_chunks_parallel.py --chunk-size 2000 --instrumented-sim cryptopool-simulator/simu`
- **Chunks**: 2,148 (`artifacts/chunks_ethusdt-1m-full/manifest.json`)
- **Parity runner**: `scripts/run_parity_parallel.sh --root=artifacts/chunks_ethusdt-1m-full --jobs=$(nproc) --output reports/ethusdt_full_chunk_summary.jsonl`
- **Reference metrics**: instrumented C++ simulator (`cryptopool-simulator/simu`)

Relative-error quantiles (absolute value across all 2,148 chunks):

| metric             | q50        | q75        | q90        | q95        | q99        | q99.9      | max (chunk, signed rel)        |
|--------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|--------------------------------|
| volume             | 1.17e-11   | 4.31e-10   | 1.20e-07   | 2.43e-07   | 7.66e-06   | 4.26e-04   | 1.22e-03 (`chunk00697`, −1.22e-03) |
| slippage           | 1.01e-03   | 2.33e-03   | 4.73e-03   | 6.84e-03   | 1.36e-02   | 2.01e-02   | 2.78e-02 (`chunk00064`, +2.78e-02) |
| liquidity_density  | 1.09e-03   | 2.87e-03   | 6.28e-03   | 1.02e-02   | 2.34e-02   | 5.75e-02   | 7.36e-02 (`chunk01589`, +7.36e-02) |
| apy                | 5.21e-10   | 3.06e-08   | 1.28e-07   | 2.32e-07   | 6.77e-07   | 7.12e-05   | 7.89e-04 (`chunk00833`, −7.89e-04) |

> **Note**  
> All chunks currently report `log_ok=false` because the instrumented C++ binary omits `PRELEG` log events. Metrics above still compare cleanly against the instrumented solver; restoring PRELEG emission will clear the log warnings without affecting these values.
