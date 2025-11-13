# IO Rewrite Plan

## Pain Points
- `scripts/generate_multi_chunks.py` used to hand-roll JSON streaming (`stream_json_array`), so malformed chunks or bracket splits silently poisoned downstream sims.
- `src/DataIO.jl` reimplemented JSON parsing on top of `Mmap`, and valid payloads (e.g., tricrypto chunk00141) tripped the “unterminated array entry” path because `_advance_to_char` couldn’t handle trailing data.
- `load_trade_bundle` / `build_cpp_trades` still clamp feeds to a global `[min_ts, max_ts]` window, so partial tails or mismatched feed lengths skew Julia replay vs. the chunk fixtures we generate.
- Chunk storage relied on unverified symlinks into `cryptopool-simulator/download`, so stale links to deleted artifacts persisted indefinitely.

## Fix Strategy
1. Adopt robust JSON parsing on the generator side (standard `json` or `ijson`) and emit explicit per-feed metadata for every chunk.
2. Switch Julia intake to `JSON3.read`, validating each candle field and normalizing timestamps/float payloads.
3. Encode per-dataset bounds (usable candles, timestamps, hashes) in chunk metadata and teach Julia to respect them.
4. Prune + verify data symlinks before launching simulations so `download/` only contains live files.

## Progress
- Swapped Julia’s candle loader to JSON3 so every chunk (chunk00141 included) parses without the manual scanner.
- Rebuilt `scripts/generate_multi_chunks.py` around Python’s JSON reader, added per-feed metadata + SHA256 hashes, and prune stale `_chunk*.json` symlinks under `cryptopool-simulator/download`. Full tricrypto chunk set regenerated with fresh C++ logs.

## Next Steps
- Document the metadata/hash guarantees so parity/diff tooling can sanity-check chunks automatically.
- Teach Julia’s bundle builders to honor `metadata.window`/`usable_candles` instead of recomputing min/max timestamps.
- Rerun the parity suite after the remaining simulator alignment fixes and archive the updated divergence stats.

If extra system packages or Python deps are required for these steps, let me know so you can install them before I wire up the changes.
