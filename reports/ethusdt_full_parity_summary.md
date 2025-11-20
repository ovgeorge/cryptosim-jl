# Parity Snapshot

- **Dataset**: ethusdt-1m-full (/home/george/data/ethusdt-1m-full.json.gz)
- **Dataset sha256**: bd052fd34b5dde524fc06b0be2124b025f14ee37a22ac32d672a677ecbc6b5ef
- **Chunk root**: /home/george/lab/cryptojl_v2/artifacts/chunks_ethusdt-1m-full
- **Chunks analyzed**: 2148
- **Chunk logs matching C++**: 1499/2148
- **Runner**: scripts/run_parity_parallel.sh --root=/home/george/lab/cryptojl_v2/artifacts/chunks_ethusdt-1m-full --data-dir=/home/george/lab/cryptojl_v2/cryptopool-simulator/download --jobs=32

> **Notes**
> Processed 2148 chunk(s)

| metric | q50.0 | q75.0 | q90.0 | q95.0 | q99.0 | q99.9 | max (chunk, rel) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| volume | 1.174339e-11 | 4.311629e-10 | 1.202649e-07 | 2.430758e-07 | 7.660852e-06 | 4.259261e-04 | 1.215633e-03 (`chunk00697`, rel=-1.215633e-03) |
| slippage | 1.005691e-03 | 2.332639e-03 | 4.728533e-03 | 6.840558e-03 | 1.355521e-02 | 2.007680e-02 | 2.778380e-02 (`chunk00064`, rel=+2.778380e-02) |
| liquidity_density | 1.092303e-03 | 2.870728e-03 | 6.281243e-03 | 1.021652e-02 | 2.337595e-02 | 5.754189e-02 | 7.358361e-02 (`chunk01589`, rel=+7.358361e-02) |
| apy | 5.205843e-10 | 3.060732e-08 | 1.284549e-07 | 2.323505e-07 | 6.765360e-07 | 7.119113e-05 | 7.889354e-04 (`chunk00833`, rel=-7.889354e-04) |

