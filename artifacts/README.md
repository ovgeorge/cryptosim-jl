# Artifacts

This folder is intentionally ignored by git. Use it for any derived data such as

- decompressed candle feeds (e.g., `artifacts/tricrypto_raw/…`)
- generated chunk fixtures and comparison logs
- temporary outputs from analysis scripts

Keeping artifacts here prevents accidental writes into the read-only `/home/george/data` symlink and keeps the repository history small.
