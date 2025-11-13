#!/usr/bin/env python3
"""
Generate StableSwap chunk fixtures from a canonical candle dataset.

For each chunk we:
  * write the sliced candle JSON under `<output>/data/<dataset>_chunkXXXXX.json`
  * emit `chunk-config.json` + `metadata.json` inside `<output>/chunkXXXXX/`
  * run both the gold-standard (unmodified) C++ simulator and our instrumented fork
    to capture `results.{gold,instrumented}.json` and CPP log files.

Chunks are indexed sequentially (chunk00000, chunk00001, …) and each chunk contains
`chunk_size` candles (default 2000). The generator always keeps the raw candle data
in `data/ethusdt-1m.json.gz` (or whichever dataset you pass in) and the derived
chunk JSON lives under the project root so the simulators can bind-mount or symlink
to it as needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterator, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def open_text(path: Path):
    path = Path(path)
    if ".gz" in path.suffixes:
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def stream_json_array(path: Path) -> Iterator[list]:
    decoder = json.JSONDecoder()
    with open_text(path) as fh:
        buffer = ""
        opened = False
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                head = buffer[0]
                if not opened:
                    if head != "[":
                        raise ValueError("JSON payload must start with '['")
                    opened = True
                    buffer = buffer[1:]
                    continue
                if head == ",":
                    buffer = buffer[1:]
                    continue
                if head == "]":
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    # Need more data, read another chunk
                    break
                yield obj
                buffer = buffer[idx:]
        buffer = buffer.lstrip()
        if buffer and buffer[0] != "]":
            raise ValueError("JSON payload did not terminate properly")


def ensure_symlink(target: Path, link_path: Path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        current = None
        try:
            current = os.readlink(link_path)
        except OSError:
            pass
        if current == rel:
            return
        link_path.unlink()
    os.symlink(rel, link_path)


def run_command(cmd, *, cwd: Path, env=None, logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with logfile.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed (see {logfile})")


def extract_cppdbg(stdout_log: Path, dest: Path):
    with stdout_log.open("r", encoding="utf-8", errors="replace") as src, dest.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            if line.startswith("CPPDBG "):
                out.write(line[7:] if line.startswith("CPPDBG ") else line)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def copy_json(src: Path, dest: Path):
    data = load_json(src)
    save_json(dest, data)


def build_chunk_config(template_cfg, datafile_name: str, debug_flag: int):
    cfg = {
        "configuration": [deepcopy(template_cfg)],
        "datafile": [datafile_name],
        "debug": debug_flag,
    }
    return cfg


def infer_dataset_name(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[:-8]
    if name.endswith(".json"):
        return name[:-5]
    return name


def run_simu(binary: Path, config_path: Path, result_path: Path, log_path: Path, extra_env=None):
    run_command(
        [str(binary), str(config_path), str(result_path)],
        cwd=binary.parent,
        env=extra_env,
        logfile=log_path,
    )


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
    parser.add_argument("--output", type=Path, default=Path("chunks_ethusdt-1m"))
    parser.add_argument("--chunk-size", type=int, default=2000, help="Candles per chunk")
    parser.add_argument("--start", type=int, default=0, help="First chunk index to materialize")
    parser.add_argument("--max-chunks", type=int, default=100, help="Number of chunks to emit (None = all)")
    parser.add_argument(
        "--gold-sim",
        type=Path,
        default=None,
        help="Path to the pristine C++ binary (omit if unavailable)",
    )
    parser.add_argument(
        "--instrumented-sim",
        type=Path,
        default=Path("instrumented-solver/simu"),
        help="Path to the instrumented C++ binary",
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


def read_config_template(path: Path):
    raw = load_json(path)
    if not raw.get("configuration"):
        raise ValueError(f"{path} does not contain any configurations")
    template = deepcopy(raw["configuration"][0])
    debug_flag = int(raw.get("debug", 0))
    return template, debug_flag


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    out_root = args.output.resolve()
    data_root = out_root / "data"
    chunks_root = out_root

    project_root = Path(__file__).resolve().parents[1]
    gold_repo = args.gold_sim.resolve().parent if args.gold_sim else None
    instrumented_repo = args.instrumented_sim.resolve().parent

    dataset_name = infer_dataset_name(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    template_cfg, debug_flag = read_config_template(config_path)

    def safe_git_commit(path: Path) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path, text=True
            ).strip()
        except Exception:
            return "unknown"

    gold_commit = safe_git_commit(gold_repo) if gold_repo else None
    instrumented_commit = safe_git_commit(instrumented_repo)
    project_commit = safe_git_commit(project_root)

    gold_sha = sha256_file(args.gold_sim.resolve()) if args.gold_sim else None
    instrumented_sha = sha256_file(args.instrumented_sim.resolve())

    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    start = max(0, args.start)
    max_chunks = args.max_chunks if args.max_chunks is None or args.max_chunks > 0 else None
    end = start + max_chunks if max_chunks is not None else None

    generated = []
    total_candles = 0
    active_chunk_idx = None
    active_chunk_data: List[list] = []
    chunk_size = args.chunk_size

    now = dt.datetime.now(dt.timezone.utc).isoformat()

    def finalize_chunk(chunk_idx: int, candles: List[list]):
        chunk_id = f"{chunk_idx:05d}"
        chunk_name = f"{dataset_name}_chunk{chunk_id}"
        chunk_dir = chunks_root / f"chunk{chunk_id}"
        if chunk_dir.exists():
            if args.force:
                shutil.rmtree(chunk_dir)
            else:
                raise RuntimeError(f"{chunk_dir} already exists; use --force to overwrite")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_data_path = data_root / f"{chunk_name}.json"
        payload = json.dumps(candles, separators=(",", ":")).encode("utf-8")
        chunk_data_path.write_bytes(payload)
        chunk_data_sha = sha256_bytes(payload)

        cfg = build_chunk_config(template_cfg, chunk_name, debug_flag)
        chunk_cfg_path = chunk_dir / "chunk-config.json"
        save_json(chunk_cfg_path, cfg)

        binaries = {
            "instrumented": {
                "path": str(args.instrumented_sim),
                "commit": instrumented_commit,
                "sha256": instrumented_sha,
            },
        }
        if args.gold_sim:
            binaries["gold"] = {
                "path": str(args.gold_sim),
                "commit": gold_commit,
                "sha256": gold_sha,
            }

        metadata = {
            "chunk": chunk_id,
            "chunk_index": chunk_idx,
            "chunk_size": chunk_size,
            "candle_range": {
                "start": chunk_idx * chunk_size,
                "count": len(candles),
            },
            "dataset": {
                "path": relative_path(dataset_path, project_root),
                "sha256": dataset_sha,
                "name": dataset_name,
            },
            "datafile": chunk_name,
            "data_path": relative_path(chunk_data_path, project_root),
            "data_sha256": chunk_data_sha,
            "config_template": relative_path(config_path, project_root),
            "generator": {
                "script": relative_path(Path(__file__), project_root),
                "timestamp": now,
                "project_commit": project_commit,
            },
            "binaries": binaries,
        }
        metadata_path = chunk_dir / "metadata.json"
        save_json(metadata_path, metadata)

        for repo in filter(None, (gold_repo, instrumented_repo)):
            link = repo / "download" / f"{chunk_name}.json"
            ensure_symlink(chunk_data_path, link)

        if not args.skip_sim:
            gold_result = None
            if args.gold_sim:
                gold_result = chunk_dir / "results.gold.json"
                gold_log = chunk_dir / "cpp_stdout.gold.log"
                run_simu(args.gold_sim.resolve(), chunk_cfg_path, gold_result, gold_log)

            instrumented_result = chunk_dir / "results.instrumented.json"
            inst_log = chunk_dir / "cpp_stdout.instrumented.log"
            run_simu(
                args.instrumented_sim.resolve(),
                chunk_cfg_path,
                instrumented_result,
                inst_log,
            )

            trace_result = chunk_dir / "results.instrumented.trace.json"
            trace_log = chunk_dir / "cpp_stdout.instrumented.trace.log"
            run_simu(
                args.instrumented_sim.resolve(),
                chunk_cfg_path,
                trace_result,
                trace_log,
                extra_env={"CPP_TRADE_DEBUG": "1", "CPP_KEEP_TMP": "0"},
            )
            extract_cppdbg(trace_log, chunk_dir / "cpp_log.jsonl")
            try:
                trace_result.unlink()
            except FileNotFoundError:
                pass
            copy_source = gold_result if gold_result else instrumented_result
            copy_json(copy_source, chunk_dir / "results.json")

        generated.append(chunk_id)

    for candle in stream_json_array(dataset_path):
        chunk_idx = total_candles // chunk_size
        within_range = chunk_idx >= start and (end is None or chunk_idx < end)
        if within_range:
            if active_chunk_idx != chunk_idx:
                active_chunk_idx = chunk_idx
                active_chunk_data = []
            active_chunk_data.append(candle)
            if len(active_chunk_data) == chunk_size:
                finalize_chunk(chunk_idx, active_chunk_data)
                active_chunk_idx = None
                active_chunk_data = []
        total_candles += 1

    if active_chunk_idx is not None and active_chunk_data:
        finalize_chunk(active_chunk_idx, active_chunk_data)

    total_chunks = (total_candles + chunk_size - 1) // chunk_size if chunk_size > 0 else 0
    manifest = {
        "dataset": {
            "path": relative_path(dataset_path, project_root),
            "sha256": dataset_sha,
            "name": dataset_name,
            "total_candles": total_candles,
        },
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "generated": generated,
        "start_index": start,
        "end_index": generated[-1] if generated else None,
        "timestamp": now,
    }
    save_json(out_root / "manifest.json", manifest)
    print(
        f"Wrote {len(generated)} chunk(s) "
        f"(chunk_size={chunk_size}, total_candles={total_candles}, total_chunks={total_chunks})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[generate_chunks] error: {exc}", file=sys.stderr)
        sys.exit(1)
