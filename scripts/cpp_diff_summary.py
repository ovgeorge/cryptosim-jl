#!/usr/bin/env python3
"""
Summarize metric deltas between two C++ simulator outputs stored inside chunk folders.

Example:
    python scripts/cpp_diff_summary.py \
        --root chunks_ethusdt-1m \
        --gold-file results.json \
        --inst-file results.instrumented.json \
        --output reports/cpp_diff_eth_default.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, List


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def load_metric(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data["configuration"][0]["Result"]
    return {
        "volume": float(result["volume"]),
        "slippage": float(result["slippage"]),
        "liquidity_density": float(result.get("liq_density", result.get("liquidity_density", 0.0))),
        "apy": float(result["APY"]),
    }


def summarize(root: Path, gold_file: str, inst_file: str) -> Dict:
    stats = {name: {"abs": [], "signed": [], "chunks": []} for name in ("volume", "slippage", "liquidity_density", "apy")}
    chunks = []
    for chunk_dir in sorted(root.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk"):
            continue
        gold_path = chunk_dir / gold_file
        inst_path = chunk_dir / inst_file
        if not (gold_path.exists() and inst_path.exists()):
            continue
        gold = load_metric(gold_path)
        inst = load_metric(inst_path)
        chunks.append(chunk_dir.name)
        for metric in stats.keys():
            g = gold[metric]
            i = inst[metric]
            rel = 0.0 if g == 0 else (i - g) / g
            stats[metric]["abs"].append(abs(rel))
            stats[metric]["signed"].append(rel)
            stats[metric]["chunks"].append((abs(rel), rel, chunk_dir.name))
    summary = {"total_chunks": len(chunks), "metrics": {}, "missing": []}
    for metric, data in stats.items():
        if not data["abs"]:
            continue
        worst = max(data["chunks"], key=lambda x: x[0])
        summary["metrics"][metric] = {
            "count": len(data["abs"]),
            "mean_abs_rel": sum(data["abs"]) / len(data["abs"]),
            "median_abs_rel": median(data["abs"]),
            "p90_abs_rel": quantile(data["abs"], 0.90),
            "p99_abs_rel": quantile(data["abs"], 0.99),
            "max_abs_rel": worst[0],
            "worst_chunk": worst[2],
            "worst_rel": worst[1],
            "mean_signed_rel": sum(data["signed"]) / len(data["signed"]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--inst-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.root.resolve(), args.gold_file, args.inst_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
