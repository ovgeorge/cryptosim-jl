#!/usr/bin/env python3
"""
Build synchronized USDT/ETH/CRV chunk fixtures:
  * align the overlapping minute window of CRV/USDT and ETH/USDT candles
  * derive an ETH/CRV feed from the ratio CRV_USDT / ETH_USDT
  * materialize chunk directories with three datafiles (0-1, 0-2, 1-2 pairs)
  * run both gold and instrumented C++ simulators to capture metrics + logs
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

getcontext().prec = 28


def load_json_rows(path: Path) -> List[list]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def rows_by_timestamp(rows: Sequence[list]) -> Dict[int, list]:
    return {int(row[0]): row for row in rows}


def format_decimal(val: Decimal, places: int = 12) -> str:
    quant = Decimal(1).scaleb(-places)
    return format(val.quantize(quant), "f")


def derive_ratio_row(ts: int, crv_row: list, eth_row: list) -> list:
    ratio = lambda a, b: Decimal(a) / Decimal(b) if Decimal(b) != 0 else Decimal(0)
    open_ = format_decimal(ratio(crv_row[1], eth_row[1]))
    high = format_decimal(ratio(crv_row[2], eth_row[3]))  # optimistic bound
    low = format_decimal(ratio(crv_row[3], eth_row[2]))   # pessimistic bound
    close = format_decimal(ratio(crv_row[4], eth_row[4]))
    volume = crv_row[5]
    close_time = int(crv_row[6]) if int(crv_row[6]) else ts + 59999
    template_tail = [
        str(close_time),
        "0.00000000",
        "0",
        "0.00000000",
        "0.00000000",
        "0",
    ]
    return [ts, open_, high, low, close, volume, *template_tail]


def build_intersection(crv_rows: List[list], eth_rows: List[list]) -> List[Tuple[int, list, list, list]]:
    eth_map = rows_by_timestamp(eth_rows)
    combined: List[Tuple[int, list, list, list]] = []
    for row in crv_rows:
        ts = int(row[0])
        if ts in eth_map:
            eth_row = eth_map[ts]
            ratio_row = derive_ratio_row(ts, row, eth_row)
            combined.append((ts, eth_row, row, ratio_row))
    combined.sort(key=lambda item: item[0])
    return combined


def chunk_ranges(total: int, chunk_size: int, start: int, max_chunks: int | None):
    first = start
    last_idx = total // chunk_size
    if max_chunks is not None:
        last_idx = min(last_idx, first + max_chunks)
    for idx in range(first, last_idx):
        begin = idx * chunk_size
        end = begin + chunk_size
        yield idx, begin, end


def run_command(cmd: Sequence[str], cwd: Path, env=None, logfile: Path | None = None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    stdout = subprocess.PIPE if logfile is None else None
    stderr = subprocess.STDOUT if logfile is not None else None
    with (open(logfile, "w", encoding="utf-8") if logfile is not None else subprocess.DEVNULL) as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=log if logfile is not None else stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed (exit {proc.returncode})")


def extract_cppdbg(stdout_log: Path, dest: Path):
    with stdout_log.open("r", encoding="utf-8", errors="replace") as src, dest.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            if line.startswith("CPPDBG "):
                out.write(line[7:])


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def ensure_symlink(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dest.parent)
    if dest.exists() or dest.is_symlink():
        try:
            if os.readlink(dest) == rel:
                return
        except OSError:
            pass
        dest.unlink()
    os.symlink(rel, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crv", type=Path, default=Path("data/crvusdt-1m.json.gz"))
    parser.add_argument("--eth", type=Path, default=Path("data/ethusdt-1m.json.gz"))
    parser.add_argument("--config", type=Path, default=Path("configs/crvusdt_chunk_config.json"))
    parser.add_argument("--output", type=Path, default=Path("chunks_usd_eth_crv-1m"))
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=100)
    parser.add_argument("--gold-sim", type=Path, default=Path("gold-standard/cryptopool-simulator/simu"))
    parser.add_argument("--instrumented-sim", type=Path, default=Path("cryptopool-simulator/simu"))
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    crv_rows = load_json_rows(args.crv.resolve())
    eth_rows = load_json_rows(args.eth.resolve())
    combined = build_intersection(crv_rows, eth_rows)
    if not combined:
        print("No overlapping timestamps between CRV and ETH datasets", file=sys.stderr)
        sys.exit(1)

    out_root = args.output.resolve()
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    config_template = json.loads(Path(args.config).read_text())
    template_cfg = config_template["configuration"][0]
    debug_flag = int(config_template.get("debug", 0))

    now = dt.datetime.now(dt.timezone.utc).isoformat()

    generated = []
    for chunk_idx, start, end in chunk_ranges(len(combined), args.chunk_size, args.start, args.max_chunks):
        slice_rows = combined[start:end]
        if len(slice_rows) < args.chunk_size:
            break
        chunk_id = f"{chunk_idx:05d}"
        chunk_dir = out_root / f"chunk{chunk_id}"
        if chunk_dir.exists() and not args.force:
            raise RuntimeError(f"{chunk_dir} exists (use --force to overwrite)")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        dataset_prefix = f"usdethcrv_chunk{chunk_id}"
        ds_names = [
            f"{dataset_prefix}_usdt_eth",
            f"{dataset_prefix}_usdt_crv",
            f"{dataset_prefix}_eth_crv",
        ]
        datasets = ([], [], [])
        for ts, eth_row, crv_row, ratio_row in slice_rows:
            datasets[0].append(eth_row)
            datasets[1].append(crv_row)
            datasets[2].append(ratio_row)
        for ds_name, rows in zip(ds_names, datasets):
            write_json(data_root / f"{ds_name}.json", rows)
            for repo in (args.gold_sim.resolve().parent, args.instrumented_sim.resolve().parent):
                ensure_symlink(data_root / f"{ds_name}.json", repo / "download" / f"{ds_name}.json")

        chunk_cfg = {
            "configuration": [template_cfg],
            "datafile": ds_names,
            "debug": debug_flag,
        }
        chunk_cfg_path = chunk_dir / "chunk-config.json"
        write_json(chunk_cfg_path, chunk_cfg)

        metadata = {
            "chunk": chunk_id,
            "chunk_index": chunk_idx,
            "chunk_size": args.chunk_size,
            "datasets": {
                "crv": str(args.crv.resolve()),
                "eth": str(args.eth.resolve()),
            },
            "datafiles": ds_names,
            "generator": {
                "script": os.path.relpath(Path(__file__).resolve(), out_root.parent),
                "timestamp": now,
            },
        }
        write_json(chunk_dir / "metadata.json", metadata)

        if not args.skip_sim:
            gold_result = chunk_dir / "results.gold.json"
            gold_log = chunk_dir / "cpp_stdout.gold.log"
            run_command([str(args.gold_sim.resolve()), str(chunk_cfg_path), str(gold_result)], args.gold_sim.resolve().parent, logfile=gold_log)

            inst_result = chunk_dir / "results.instrumented.json"
            inst_log = chunk_dir / "cpp_stdout.instrumented.log"
            run_command(
                [str(args.instrumented_sim.resolve()), str(chunk_cfg_path), str(inst_result)],
                args.instrumented_sim.resolve().parent,
                logfile=inst_log,
            )

            trace_result = chunk_dir / "results.instrumented.trace.json"
            trace_log = chunk_dir / "cpp_stdout.instrumented.trace.log"
            run_command(
                [str(args.instrumented_sim.resolve()), str(chunk_cfg_path), str(trace_result)],
                args.instrumented_sim.resolve().parent,
                env={"CPP_TRADE_DEBUG": "1", "CPP_KEEP_TMP": "0"},
                logfile=trace_log,
            )
            extract_cppdbg(trace_log, chunk_dir / "cpp_log.jsonl")
            try:
                trace_result.unlink()
            except FileNotFoundError:
                pass
            write_json(chunk_dir / "results.json", json.loads(gold_result.read_text()))
        generated.append(chunk_id)

    manifest = {
        "total_rows": len(combined),
        "chunk_size": args.chunk_size,
        "generated": generated,
        "timestamp": now,
    }
    write_json(out_root / "manifest.json", manifest)
    print(f"Generated {len(generated)} chunk(s) into {out_root}")


if __name__ == "__main__":
    main()
