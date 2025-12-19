#!/usr/bin/env python3

"""
Render simple heatmap "colormaps" from grid_search_stableswap2.jl JSONL output.

This script intentionally avoids heavyweight plotting deps (matplotlib/numpy) and uses
only Pillow (PIL), which is already present in this environment.

Example:
  python3 scripts/plot_stableswap2_grid_colormaps.py \
    --input artifacts/experiments/stableswap2_grid/btcusdc_full_combined.jsonl \
    --out-dir artifacts/experiments/stableswap2_grid/colormaps
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def fmt_num(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return format(x, ".6g")


def viridis_rgb(t: float) -> Tuple[int, int, int]:
    # A small set of viridis-like stops (RGB), linearly interpolated.
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            if t1 == t0:
                return c1
            u = (t - t0) / (t1 - t0)
            r = round(c0[0] + u * (c1[0] - c0[0]))
            g = round(c0[1] + u * (c1[1] - c0[1]))
            b = round(c0[2] + u * (c1[2] - c0[2]))
            return int(r), int(g), int(b)
    return stops[-1][1]


def pick_tick_indices(n: int, max_ticks: int = 8) -> List[int]:
    if n <= 0:
        return []
    if n <= max_ticks:
        return list(range(n))
    # Evenly spaced, always include endpoints.
    idx = {0, n - 1}
    if max_ticks <= 2:
        return sorted(idx)
    for k in range(1, max_ticks - 1):
        idx.add(int(round(k * (n - 1) / (max_ticks - 1))))
    return sorted(idx)


@dataclass(frozen=True)
class Grid:
    As: List[float]
    fees: List[float]
    values: Dict[str, List[List[float]]]  # metric -> [iA][ifee]


def _metrics_obj(row: dict) -> dict:
    # grid_search_stableswap2.jl uses `metrics`, but keep compatibility with C++ `Result`.
    m = row.get("metrics")
    if isinstance(m, dict):
        return m
    m = row.get("Result")
    if isinstance(m, dict):
        return m
    return {}


def load_grid(path: str, metrics: List[str]) -> Grid:
    rows: List[Tuple[float, float, dict]] = []
    As_set = set()
    fees_set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            A = float(row["A"])
            fee = float(row["mid_fee"])
            As_set.add(A)
            fees_set.add(fee)
            m = _metrics_obj(row)
            rows.append((A, fee, m))

    As = sorted(As_set)
    fees = sorted(fees_set)
    idxA = {a: i for i, a in enumerate(As)}
    idxF = {f: j for j, f in enumerate(fees)}

    values: Dict[str, List[List[float]]] = {
        metric: [[math.nan for _ in fees] for _ in As] for metric in metrics
    }
    for A, fee, m in rows:
        i = idxA[A]
        j = idxF[fee]
        for metric in metrics:
            v = m.get(metric)
            try:
                values[metric][i][j] = float(v)
            except Exception:
                values[metric][i][j] = math.nan

    return Grid(As=As, fees=fees, values=values)


def _finite_minmax(vals: Iterable[float]) -> Tuple[float, float]:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return (math.nan, math.nan)
    return (min(finite), max(finite))


def render_metric_png(
    *,
    grid: Grid,
    metric: str,
    output_path: str,
    title: str,
    cell_px: int = 56,
    font_px: int = 13,
    color_scale: str = "linear",
) -> None:
    As = grid.As
    fees = grid.fees
    Z = grid.values[metric]

    nA = len(As)
    nF = len(fees)

    margin_left = 110
    margin_top = 50
    margin_bottom = 90
    margin_right = 30
    colorbar_w = 26
    colorbar_pad = 18

    grid_w = nF * cell_px
    grid_h = nA * cell_px
    w = margin_left + grid_w + colorbar_pad + colorbar_w + margin_right
    h = margin_top + grid_h + margin_bottom

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Pillow's default font is small; scale the layout with font_px via spacing instead.
    del font_px

    # Collect min/max for normalization.
    flat = [v for row in Z for v in row]
    vmin, vmax = _finite_minmax(flat)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    def transform(v: float) -> float:
        if not math.isfinite(v):
            return math.nan
        if color_scale == "log10":
            return math.log10(v) if v > 0 else math.nan
        return v

    tmin = transform(vmin)
    tmax = transform(vmax)
    if not math.isfinite(tmin) or not math.isfinite(tmax) or tmin == tmax:
        tmin, tmax = 0.0, 1.0

    def color_for(v: float) -> Tuple[int, int, int]:
        tv = transform(v)
        if not math.isfinite(tv):
            return (210, 210, 210)
        t = (tv - tmin) / (tmax - tmin)
        return viridis_rgb(t)

    # Title.
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    # Heatmap cells (A increases upwards; SVG-style plotting).
    x0 = margin_left
    y0 = margin_top
    for iA, A in enumerate(As):
        # invert vertical index so smallest A is at bottom
        y = y0 + (nA - 1 - iA) * cell_px
        for jF, fee in enumerate(fees):
            x = x0 + jF * cell_px
            v = Z[iA][jF]
            draw.rectangle(
                [x, y, x + cell_px - 1, y + cell_px - 1],
                fill=color_for(v),
                outline=(240, 240, 240),
            )

    # Axes labels.
    draw.text((margin_left + grid_w / 2 - 60, h - margin_bottom + 55), "mid_fee (= out_fee)", fill=(0, 0, 0), font=font)
    draw.text((10, margin_top + grid_h / 2 - 5), "A", fill=(0, 0, 0), font=font)

    # X ticks.
    x_ticks = set(pick_tick_indices(len(fees)))
    for jF, fee in enumerate(fees):
        if jF not in x_ticks:
            continue
        x = x0 + jF * cell_px + cell_px / 2
        y = y0 + grid_h + 8
        draw.line([(x, y0 + grid_h), (x, y0 + grid_h + 6)], fill=(0, 0, 0))
        label = fmt_num(fee)
        tw = draw.textlength(label, font=font)
        draw.text((x - tw / 2, y0 + grid_h + 10), label, fill=(0, 0, 0), font=font)

    # Y ticks.
    y_ticks = set(pick_tick_indices(len(As)))
    for iA, A in enumerate(As):
        if iA not in y_ticks:
            continue
        y = y0 + (nA - 1 - iA) * cell_px + cell_px / 2
        draw.line([(x0 - 6, y), (x0, y)], fill=(0, 0, 0))
        label = fmt_num(A)
        tw = draw.textlength(label, font=font)
        draw.text((x0 - 10 - tw, y - 6), label, fill=(0, 0, 0), font=font)

    # Colorbar.
    cb_x = margin_left + grid_w + colorbar_pad
    cb_y = margin_top
    steps = max(1, grid_h)
    for k in range(steps):
        t = 1.0 - k / (steps - 1) if steps > 1 else 0.5
        draw.line([(cb_x, cb_y + k), (cb_x + colorbar_w, cb_y + k)], fill=viridis_rgb(t))
    draw.rectangle([cb_x, cb_y, cb_x + colorbar_w, cb_y + grid_h], outline=(0, 0, 0))

    # Colorbar labels.
    draw.text((cb_x + colorbar_w + 6, cb_y - 6), fmt_num(vmax), fill=(0, 0, 0), font=font)
    draw.text((cb_x + colorbar_w + 6, cb_y + grid_h - 6), fmt_num(vmin), fill=(0, 0, 0), font=font)
    draw.text((cb_x, cb_y + grid_h + 10), metric, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="PNG")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL grid file (from grid_search_stableswap2.jl)")
    p.add_argument("--out-dir", default="", help="Output directory (default: alongside input)")
    p.add_argument("--prefix", default="", help="Output filename prefix (default: input basename)")
    p.add_argument(
        "--metrics",
        default="apy,slippage,liquidity_density,volume",
        help="Comma-separated metrics to render (default: apy,slippage,liquidity_density,volume)",
    )
    p.add_argument("--cell", type=int, default=56, help="Cell size in pixels (default: 56)")
    p.add_argument(
        "--color-scale",
        choices=["linear", "log10"],
        default="linear",
        help="Color scaling (default: linear)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    grid = load_grid(args.input, metrics)

    out_dir = args.out_dir or os.path.dirname(args.input)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    for metric in metrics:
        out_path = os.path.join(out_dir, f"{prefix}_{metric}.png")
        render_metric_png(
            grid=grid,
            metric=metric,
            output_path=out_path,
            title=f"{prefix}: {metric}",
            cell_px=args.cell,
            color_scale=args.color_scale,
        )


if __name__ == "__main__":
    main()
