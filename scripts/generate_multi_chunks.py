#!/usr/bin/env python3
"""
Generate chunk fixtures for multi-datafile configs (e.g. tricrypto).

For every chunk we:
  * slice each source candle file into sequential windows
  * write the chunk-specific JSON under <output>/data/<name>_chunkXXXXX.json
  * symlink those chunk JSON files into `cryptopool-simulator/download/`
  * emit chunk directories containing chunk-config.json, metadata.json,
    results.json, cpp_stdout.log, and cpp_log.jsonl produced by the C++ simulator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence, Tuple


def save_json(path: Path, obj, *, return_hash: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(obj, separators=(",", ":"))
    with path.open("w", encoding="utf-8") as fh:
        fh.write(serialized)
    if return_hash:
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return None


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            try:
                current = os.readlink(link_path)
                if current == rel:
                    return
            except OSError:
                pass
            link_path.unlink()
        else:
            raise RuntimeError(
                f"{link_path} exists and is not a symlink; "
                "refusing to overwrite to keep download/ clean"
            )
    os.symlink(rel, link_path)


def run_simulator(sim_bin: Path, config: Path, result: Path, stdout_log: Path) -> None:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CPP_TRADE_DEBUG": "1", "CPP_KEEP_TMP": "0"})
    with stdout_log.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(sim_bin), str(config), str(result)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(sim_bin.parent),
        )


def extract_cppdbg(stdout_log: Path, dest: Path) -> None:
    with stdout_log.open("r", encoding="utf-8", errors="replace") as src, dest.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            if line.startswith("CPPDBG "):
                out.write(line[7:] if line.startswith("CPPDBG ") else line)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tricrypto_chunk_config.json"),
        help="Path to the base config that references the source candle files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tricrypto_chunks"),
        help="Directory where chunk dirs + data should be written",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("cryptopool-simulator/download"),
        help="Directory containing the source candle JSON (and where chunk JSON symlinks are created)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Candles per chunk for each source dataset",
    )
    parser.add_argument("--start", type=int, default=0, help="First chunk index to emit")
    parser.add_argument("--max-chunks", type=int, default=None, help="Total chunks to emit")
    parser.add_argument(
        "--sim-bin",
        type=Path,
        default=Path("cryptopool-simulator/simu"),
        help="Path to the C++ simulator binary",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing chunk directories / chunk data",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip running the C++ simulator (useful for dry runs)",
    )
    return parser.parse_args()


def resolve_dataset_path(name: str, data_dir: Path) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    filename = name if name.endswith(".json") else f"{name}.json"
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"dataset '{name}' not found (looked for {path})")
    return path.resolve()


def load_dataset_rows(path: Path) -> List[Sequence[float]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON array")
    rows: List[Sequence[float]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, list):
            raise ValueError(f"{path} row {idx} is not an array")
        if len(row) != 6:
            raise ValueError(f"{path} row {idx} must contain 6 fields (got {len(row)})")
        rows.append(row)
    return rows


def build_dataset_metadata(name: str, path: Path, rows: List[Sequence[float]], usable: int):
    total = len(rows)
    dropped = total - usable
    first_ts = rows[0][0] if rows else None
    last_ts = rows[usable - 1][0] if usable > 0 else None
    return {
        "name": name,
        "path": str(path),
        "total_candles": total,
        "usable_candles": usable,
        "dropped_tail": max(0, dropped),
        "first_timestamp": first_ts,
        "last_usable_timestamp": last_ts,
    }


def prune_stale_chunk_links(download_dir: Path) -> int:
    removed = 0
    for entry in download_dir.glob("*_chunk*.json"):
        if not entry.is_symlink():
            continue
        target = entry.resolve(strict=False)
        if not target.exists():
            entry.unlink()
            removed += 1
    if removed:
        print(f"[info] removed {removed} stale chunk symlinks from {download_dir}")
    return removed


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    template_cfg = deepcopy(config["configuration"][0])
    debug_flag = int(config.get("debug", 0))
    datafile_names = [str(entry) for entry in config.get("datafile", [])]
    if not datafile_names:
        raise ValueError("config does not list any datafiles")
    datasets = [
        (name, resolve_dataset_path(name, args.data_dir.resolve()))
        for name in datafile_names
    ]

    out_root = args.output.resolve()
    data_root = out_root / "data"
    download_dir = args.data_dir.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_chunk_links(download_dir)
    sim_bin = args.sim_bin.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    chunk_size = args.chunk_size
    start_idx = max(0, args.start)
    max_chunks = args.max_chunks if args.max_chunks is None or args.max_chunks > 0 else None

    dataset_rows: List[List[Sequence[float]]] = []
    for name, path in datasets:
        rows = load_dataset_rows(path)
        if not rows:
            raise RuntimeError(f"{path} contains no candles")
        dataset_rows.append(rows)

    usable_len = min(len(rows) for rows in dataset_rows)
    if usable_len == 0:
        raise RuntimeError("datasets share no overlapping candles")
    if any(len(rows) != usable_len for rows in dataset_rows):
        for (name, path), rows in zip(datasets, dataset_rows):
            extra = len(rows) - usable_len
            if extra > 0:
                print(
                    f"[warn] trimming {extra} candles from tail of {name} ({path}) "
                    f"to match shortest dataset"
                )

    total_chunks = math.ceil(usable_len / chunk_size)
    if total_chunks == 0 or start_idx >= total_chunks:
        raise RuntimeError(
            f"start index {start_idx} is beyond available chunk windows ({total_chunks})"
        )

    dataset_meta = [
        build_dataset_metadata(name, path, rows, usable_len)
        for (name, path), rows in zip(datasets, dataset_rows)
    ]

    emitted = 0
    for chunk_id in range(start_idx, total_chunks):
        if max_chunks is not None and emitted >= max_chunks:
            break
        window_start = chunk_id * chunk_size
        window_end = min(window_start + chunk_size, usable_len)
        chunk_name = f"chunk{chunk_id:05d}"
        chunk_dir = out_root / chunk_name
        if chunk_dir.exists():
            if args.force:
                shutil.rmtree(chunk_dir)
            else:
                raise RuntimeError(f"{chunk_dir} already exists (use --force)")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_dataset_entries: List[Tuple[str, Path, int, int, int, str]] = []
        datafile_names: List[str] = []
        for (dataset_name, _), rows in zip(datasets, dataset_rows):
            payload = rows[window_start:window_end]
            chunk_base = f"{dataset_name}_chunk{chunk_id:05d}"
            chunk_path = data_root / f"{chunk_base}.json"
            chunk_hash = save_json(chunk_path, payload, return_hash=True)
            chunk_dataset_entries.append(
                (
                    chunk_base,
                    chunk_path,
                    len(payload),
                    payload[0][0] if payload else None,
                    payload[-1][0] if payload else None,
                    chunk_hash,
                )
            )
            link_path = download_dir / f"{chunk_base}.json"
            ensure_symlink(chunk_path, link_path)
            datafile_names.append(chunk_base)

        chunk_cfg_entry = deepcopy(template_cfg)
        chunk_cfg = {
            "configuration": [chunk_cfg_entry],
            "datafile": datafile_names,
            "debug": debug_flag,
        }
        chunk_cfg_path = chunk_dir / "chunk-config.json"
        save_json(chunk_cfg_path, chunk_cfg)

        metadata = {
            "chunk": f"{chunk_id:05d}",
            "chunk_size": chunk_size,
            "window": {
                "start_index": window_start,
                "end_index": window_end,
                "length": window_end - window_start,
            },
            "datafiles": [
                {
                    "name": entry,
                    "path": str(path),
                    "candles": count,
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts,
                    "sha256": chunk_hash,
                }
                for (entry, path, count, first_ts, last_ts, chunk_hash) in chunk_dataset_entries
            ],
            "source_datasets": dataset_meta,
        }
        save_json(chunk_dir / "metadata.json", metadata)

        if not args.skip_sim:
            stdout_log = chunk_dir / "cpp_stdout.log"
            result_path = chunk_dir / "results.json"
            run_simulator(sim_bin, chunk_cfg_path, result_path, stdout_log)
            extract_cppdbg(stdout_log, chunk_dir / "cpp_log.jsonl")

        emitted += 1

    if emitted == 0:
        raise RuntimeError("no chunks were generated; adjust start/max or dataset lengths")


if __name__ == "__main__":
    main()
