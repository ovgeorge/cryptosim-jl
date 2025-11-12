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
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Iterator, List, Tuple


def stream_json_array(path: Path) -> Iterator[list]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
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
                        raise ValueError(f"{path} must start with '['")
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
                    break
                yield obj
                buffer = buffer[idx:]
        buffer = buffer.lstrip()
        if buffer and buffer[0] != "]":
            raise ValueError(f"{path} JSON array did not terminate properly")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        try:
            current = os.readlink(link_path)
            if current == rel:
                return
        except OSError:
            pass
        link_path.unlink()
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
    sim_bin = args.sim_bin.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    chunk_size = args.chunk_size
    start_idx = max(0, args.start)
    max_chunks = args.max_chunks if args.max_chunks is None or args.max_chunks > 0 else None

    streams = [stream_json_array(path) for _, path in datasets]
    buffers = [[] for _ in datasets]
    eof = [False for _ in datasets]
    chunk_idx = 0
    emitted = 0

    while True:
        for j, stream in enumerate(streams):
            while len(buffers[j]) < chunk_size and not eof[j]:
                try:
                    buffers[j].append(next(stream))
                except StopIteration:
                    eof[j] = True
                    break
        if all(eof) and all(len(buf) == 0 for buf in buffers):
            break
        if any(len(buf) == 0 for buf in buffers) and not all(eof):
            raise RuntimeError("datasets have mismatched lengths; cannot chunk evenly")

        if chunk_idx >= start_idx:
            chunk_id = chunk_idx
            chunk_name = f"chunk{chunk_id:05d}"
            chunk_dir = out_root / chunk_name
            if chunk_dir.exists():
                if args.force:
                    shutil.rmtree(chunk_dir)
                else:
                    raise RuntimeError(f"{chunk_dir} already exists (use --force)")
            chunk_dir.mkdir(parents=True, exist_ok=True)

            chunk_dataset_entries: List[Tuple[str, Path]] = []
            for (dataset_name, _), payload in zip(datasets, buffers):
                payload = list(payload)
                chunk_base = f"{dataset_name}_chunk{chunk_id:05d}"
                chunk_path = data_root / f"{chunk_base}.json"
                save_json(chunk_path, payload)
                chunk_dataset_entries.append((chunk_base, chunk_path))
                link_path = download_dir / f"{chunk_base}.json"
                ensure_symlink(chunk_path, link_path)

            chunk_cfg_entry = deepcopy(template_cfg)
            chunk_cfg = {
                "configuration": [chunk_cfg_entry],
                "datafile": [entry for entry, _ in chunk_dataset_entries],
                "debug": debug_flag,
            }
            save_json(chunk_dir / "chunk-config.json", chunk_cfg)

            metadata = {
                "chunk": f"{chunk_id:05d}",
                "chunk_size": chunk_size,
                "datafiles": [
                    {"name": entry, "path": str(path)} for entry, path in chunk_dataset_entries
                ],
                "source_datasets": [
                    {"name": name, "path": str(path)} for name, path in datasets
                ],
            }
            save_json(chunk_dir / "metadata.json", metadata)

            if not args.skip_sim:
                stdout_log = chunk_dir / "cpp_stdout.log"
                result_path = chunk_dir / "results.json"
                run_simulator(sim_bin, chunk_dir / "chunk-config.json", result_path, stdout_log)
                extract_cppdbg(stdout_log, chunk_dir / "cpp_log.jsonl")

            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                break

        for buf in buffers:
            buf.clear()
        chunk_idx += 1

        if max_chunks is not None and emitted >= max_chunks:
            break

    if emitted == 0:
        raise RuntimeError("no chunks were generated; check dataset size/start/max settings")


if __name__ == "__main__":
    main()
