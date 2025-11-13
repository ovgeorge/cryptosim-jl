#!/usr/bin/env python3
"""
Parallel chunk generator that fans out `generate_chunks.py` across all CPU cores.

This wrapper keeps the chunking logic identical to the sequential script but
splits the work into contiguous ranges and spawns as many workers as the machine
has hardware threads. We always want to keep every core busy; override the job
count only when you absolutely must throttle the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SEQUENTIAL_SCRIPT = THIS_DIR / "generate_chunks.py"

_seq_spec = importlib.util.spec_from_file_location("_sequential_chunk_gen", SEQUENTIAL_SCRIPT)
if _seq_spec is None or _seq_spec.loader is None:
    raise RuntimeError(f"Unable to load {SEQUENTIAL_SCRIPT}")
_seq_module = importlib.util.module_from_spec(_seq_spec)
_seq_spec.loader.exec_module(_seq_module)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ethusdt-1m.json.gz"),
        help="Path to the canonical candle dataset (.json or .json.gz)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ethusdt_chunk_config.json"),
        help="Template config used for every chunk",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/chunks_ethusdt-1m"),
        help="Directory that will receive chunk dirs and data/",
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="Candles per chunk")
    parser.add_argument("--start", type=int, default=0, help="First chunk index to emit")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Total number of chunks to emit (None = drain dataset)",
    )
    parser.add_argument(
        "--gold-sim",
        type=Path,
        default=None,
        help="Path to the pristine C++ binary (omit until the gold solver is available)",
    )
    parser.add_argument(
        "--instrumented-sim",
        type=Path,
        default=Path("cryptopool-simulator/simu"),
        help="Path to the instrumented C++ binary",
    )
    parser.add_argument(
        "--chunks-per-job",
        type=int,
        default=256,
        help="How many consecutive chunks each worker should process",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Override the worker count (defaults to all CPU cores; avoid changing this)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk directories/data if present",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Only write chunk data/config/metadata; do not run simulators",
    )
    return parser.parse_args()


def count_total_candles(dataset: Path) -> int:
    total = 0
    for _ in _seq_module.stream_json_array(dataset):
        total += 1
    return total


def chunk_ranges(start: int, end: int, span: int) -> Iterable[Tuple[int, int]]:
    current = start
    while current < end:
        yield current, min(span, end - current)
        current += span


def build_command(args, start: int, count: int) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(SEQUENTIAL_SCRIPT),
        "--dataset",
        str(args.dataset),
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--chunk-size",
        str(args.chunk_size),
        "--start",
        str(start),
        "--max-chunks",
        str(count),
        "--instrumented-sim",
        str(args.instrumented_sim),
    ]
    if args.gold_sim:
        cmd.extend(["--gold-sim", str(args.gold_sim)])
    if args.force:
        cmd.append("--force")
    if args.skip_sim:
        cmd.append("--skip-sim")
    return cmd


def run_job(job_id: int, start: int, count: int, args) -> None:
    cmd = build_command(args, start, count)
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            print(f"[parallel][job {job_id}][{start}:{start + count}) {line}")
    if proc.returncode != 0:
        err_block = proc.stderr.strip() if proc.stderr else ""
        raise RuntimeError(
            f"Job {job_id} failed for range [{start}, {start + count}):\n{err_block}"
        )


def list_chunk_ids(output_dir: Path) -> List[str]:
    chunk_ids: List[str] = []
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("chunk"):
            continue
        suffix = name[5:]
        if suffix.isdigit():
            chunk_ids.append(suffix)
    return sorted(chunk_ids, key=lambda item: int(item))


def rebuild_manifest(
    output_dir: Path,
    chunk_size: int,
    dataset_path: Path,
    total_candles: int,
    start_index: int,
) -> None:
    dataset_sha = _seq_module.sha256_file(dataset_path)
    dataset_name = _seq_module.infer_dataset_name(dataset_path)
    dataset_rel = _seq_module.relative_path(dataset_path, PROJECT_ROOT)
    total_chunks = math.ceil(total_candles / chunk_size) if chunk_size > 0 else 0
    chunk_ids = list_chunk_ids(output_dir)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "dataset": {
            "path": dataset_rel,
            "sha256": dataset_sha,
            "name": dataset_name,
            "total_candles": total_candles,
        },
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "generated": chunk_ids,
        "start_index": start_index,
        "end_index": chunk_ids[-1] if chunk_ids else None,
        "timestamp": now,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.chunks_per_job <= 0:
        raise ValueError("--chunks-per-job must be positive")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    output_dir = args.output.resolve()
    instrumented_sim = args.instrumented_sim.resolve()
    if not instrumented_sim.exists():
        raise FileNotFoundError(f"Instrumented simulator not found: {instrumented_sim}")
    gold_sim = args.gold_sim.resolve() if args.gold_sim else None
    if gold_sim and not gold_sim.exists():
        raise FileNotFoundError(f"Gold simulator not found: {gold_sim}")

    args.dataset = dataset_path
    args.config = config_path
    args.output = output_dir
    args.instrumented_sim = instrumented_sim
    args.gold_sim = gold_sim
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    total_candles = count_total_candles(dataset_path)
    total_chunks = math.ceil(total_candles / args.chunk_size)
    start = max(0, args.start)
    if start >= total_chunks:
        raise ValueError(f"Start index {start} exceeds total chunks {total_chunks}")
    remaining = total_chunks - start
    target = remaining if args.max_chunks is None else min(max(args.max_chunks, 0), remaining)
    if target == 0:
        print("[parallel] nothing to do (target chunk count is zero)")
        return
    end = start + target

    ranges = list(chunk_ranges(start, end, args.chunks_per_job))
    jobs = args.jobs or os.cpu_count() or 1
    print(
        f"[parallel] dataset={dataset_path} "
        f"chunks={total_chunks} chunk_size={args.chunk_size} "
        f"start={start} target={target} jobs={jobs}",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for job_id, (range_start, count) in enumerate(ranges):
            futures.append(executor.submit(run_job, job_id, range_start, count, args))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    rebuild_manifest(output_dir, args.chunk_size, dataset_path, total_candles, start)
    print(
        f"[parallel] completed generation into {output_dir} "
        f"(chunk_size={args.chunk_size}, total_candles={total_candles})"
    )


if __name__ == "__main__":
    main()
