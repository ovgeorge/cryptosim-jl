#!/usr/bin/env python3
import argparse
import json
import pathlib


def parse_probe_line(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STEPPROBE "):
                line = line[len("STEPPROBE ") :]
            return json.loads(line)
    raise RuntimeError(f"No STEPPROBE payloads found in {path}")


def fmt(val: float) -> str:
    return f"{val:.17e}"


def load_step_event(log_path: pathlib.Path, candle: int, stage: str):
    stage = stage.upper()
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            if data.get("event") == "STEP" and data.get("candle_index") == candle and data.get("stage", "").upper() == stage:
                return data
    raise RuntimeError(f"STEP event not found in {log_path} for candle {candle} stage {stage}")


def load_leg_event(log_path: pathlib.Path, candle: int, stage: str):
    stage = stage.upper()
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            if data.get("event") == "LEG" and data.get("candle_index") == candle and data.get("stage", "").upper() == stage:
                return data
    raise RuntimeError(f"LEG event not found in {log_path} for candle {candle} stage {stage}")


def to_snapshot(probe, cfg, cpp_step, cpp_leg, julia_leg):
    trader_cfg = cfg["configuration"][0]
    pair = [probe["pair"][0] + 1, probe["pair"][1] + 1]
    description = f"Probe dump candle {probe['candle']:05d} stage {probe['stage'].upper()}"
    snapshot = {
        "description": description,
        "pair": pair,
        "stage": probe["stage"],
        "token_balances": [fmt(v) for v in probe["balances"]],
        "price_scale": [fmt(v) for v in probe["price_scale"]],
        "price_oracle": [fmt(v) for v in probe["price_oracle"]],
        "last_price": [fmt(v) for v in probe["last_price"]],
        "xcp": fmt(probe["xcp"]),
        "xcp_profit": fmt(probe["xcp_profit"]),
        "xcp_profit_real": fmt(probe["xcp_profit_real"]),
        "dx": fmt(probe["dx_seed"]),
        "mid_fee": fmt(trader_cfg["mid_fee"]),
        "out_fee": fmt(trader_cfg["out_fee"]),
        "fee_gamma": fmt(trader_cfg["fee_gamma"]),
        "adjustment_step": fmt(trader_cfg["adjustment_step"]),
        "allowed_extra_profit": fmt(trader_cfg["allowed_extra_profit"]),
        "gas_fee": fmt(trader_cfg["gas_fee"]),
        "ext_fee": fmt(trader_cfg["ext_fee"]),
        "p_min": fmt(probe["p_min"]),
        "p_max": fmt(probe["p_max"]),
        "volume": fmt(probe["volume"]),
        "ext_vol": fmt(probe["ext_vol"]),
        "step_cpp": fmt(cpp_step["dx"]),
        "step_julia": fmt(probe["dx"]),
        "gross_cpp": fmt(cpp_leg["dy"]),
        "gross_julia": fmt(julia_leg["dy"]),
    }
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Convert a STEPPROBE payload into a Newton snapshot JSON.")
    parser.add_argument("--probe", required=True, type=pathlib.Path, help="Path to STEPPROBE JSONL file")
    parser.add_argument("--chunk", required=True, type=pathlib.Path, help="Chunk directory (contains logs/config)")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output snapshot JSON path")
    args = parser.parse_args()

    probe = parse_probe_line(args.probe)
    cfg = json.loads((args.chunk / "chunk-config.json").read_text())
    stage = probe["stage"]
    candle = probe["candle"]
    cpp_step = load_step_event(args.chunk / "cpp_log.jsonl", candle, stage)
    cpp_leg = load_leg_event(args.chunk / "cpp_log.jsonl", candle, stage)
    julia_leg = load_leg_event(args.chunk / f"julia_log.{args.chunk.name}.jsonl", candle, stage)

    snapshot = to_snapshot(probe, cfg, cpp_step, cpp_leg, julia_leg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote snapshot to {args.out}")


if __name__ == "__main__":
    main()
