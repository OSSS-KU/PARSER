#!/usr/bin/env python
"""Aggregate per-method pipeline timing into one method x sparsity table.

Reads the Level-2 timing sidecars emitted by the pipeline:
  - ``<model>/blocks/calibration_timing.json``    (save_blocks.py) — calibration
    wall time keyed directly by compression method name.
  - ``<model>/<tag>/compressed/phase_timing.json`` (compress_blocks.py) — one per
    method x sparsity: prepare / backbone / compress_only.

Output: one row per (method, sparsity) with calibration / prepare / backbone /
compress_only / wall seconds. Calibration is shared across a method's sparsity
sweep, so the same calibration value repeats on each of that method's rows.

Usage:
  python aggregate_timing.py artifacts/Qwen_Qwen1.5-MoE-A2.7B
  python aggregate_timing.py artifacts            # scans every model subdir
  python aggregate_timing.py artifacts/ModelA artifacts/ModelB --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: could not read {path}: {exc}", file=sys.stderr)
        return None


def _is_model_dir(path: Path) -> bool:
    return (path / "blocks").is_dir()


def _discover_model_dirs(roots: List[Path]) -> List[Path]:
    models: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"warn: path does not exist: {root}", file=sys.stderr)
            continue
        if _is_model_dir(root):
            models.append(root)
        else:
            for child in sorted(root.iterdir()):
                if child.is_dir() and _is_model_dir(child):
                    models.append(child)
    return models


def _calibration_by_method(model_dir: Path) -> Dict[str, float]:
    """Return {method_name: calibration_seconds} from calibration_timing.json."""
    calib = _load_json(model_dir / "blocks" / "calibration_timing.json") or {}
    out: Dict[str, float] = {}
    methods = calib.get("methods", {})
    if isinstance(methods, dict):
        for method, entry in methods.items():
            if isinstance(entry, dict) and entry.get("calibration_seconds") is not None:
                out[method] = float(entry["calibration_seconds"])
    return out


def _collect_rows(model_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for phase_path in sorted(model_dir.glob("*/compressed/phase_timing.json")):
        timing = _load_json(phase_path)
        if timing is None:
            continue
        method = str(timing.get("method", "?"))
        sparsity = timing.get("sparsity")
        derived = timing.get("derived", {})
        phases = timing.get("phases", {})

        def _phase(name: str) -> float:
            entry = phases.get(name)
            return float(entry.get("seconds", 0.0)) if isinstance(entry, dict) else 0.0

        prepare = float(derived.get("prepare_seconds", _phase("prepare")))
        backbone = float(derived.get("backbone_seconds", _phase("backbone")))
        compress_only = float(derived.get("compress_only_seconds", 0.0))
        wall = float(derived.get("wall_seconds", prepare + backbone + compress_only))
        rows.append(
            {
                "method": method,
                "sparsity": sparsity,
                "prepare": prepare,
                "backbone": backbone,
                "compress_only": compress_only,
                "wall": wall,
                "tag": phase_path.parent.parent.name,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="model dir(s) with blocks/, or a parent like artifacts/")
    parser.add_argument("--csv", default=None, help="optional path to write a flat CSV")
    args = parser.parse_args()

    model_dirs = _discover_model_dirs([Path(p) for p in args.paths])
    if not model_dirs:
        print("no model directories found (need a dir containing blocks/)", file=sys.stderr)
        sys.exit(1)

    csv_rows: List[dict] = []
    hdr = (
        f"  {'sparsity':>8s} {'calibration':>13s} {'prepare':>11s} "
        f"{'backbone':>12s} {'compress':>12s} {'wall':>13s}"
    )

    for model_dir in model_dirs:
        calib_by_method = _calibration_by_method(model_dir)
        rows = _collect_rows(model_dir)
        print(f"\n=== {model_dir} ===")
        if not rows:
            print("  (no compressed/phase_timing.json found yet)")
            continue

        by_method: Dict[str, List[dict]] = {}
        for r in rows:
            by_method.setdefault(r["method"], []).append(r)

        for method in sorted(by_method):
            mrows = sorted(
                by_method[method],
                key=lambda r: (r["sparsity"] is None, r["sparsity"]),
            )
            calib_s = calib_by_method.get(method)
            calib_str = "-" if calib_s is None else f"{calib_s:.3f}"
            print(f"\nmethod: {method}   (calibration measured this run: {calib_str} s)")
            print(hdr)
            sweep_compress = 0.0
            for r in mrows:
                sp = "  -   " if r["sparsity"] is None else f"{float(r['sparsity']):.3f}"
                cs = "-" if calib_s is None else f"{calib_s:13.3f}"
                print(
                    f"  {sp:>8s} {cs:>13s} {r['prepare']:11.3f} "
                    f"{r['backbone']:12.3f} {r['compress_only']:12.3f} {r['wall']:13.3f}"
                )
                sweep_compress += r["wall"]
                csv_rows.append(
                    {
                        "model": model_dir.name,
                        "method": method,
                        "sparsity": r["sparsity"],
                        "calibration_s": "" if calib_s is None else round(calib_s, 3),
                        "prepare_s": round(r["prepare"], 3),
                        "backbone_s": round(r["backbone"], 3),
                        "compress_only_s": round(r["compress_only"], 3),
                        "wall_s": round(r["wall"], 3),
                        "tag": r["tag"],
                    }
                )
            grand = sweep_compress + (calib_s or 0.0)
            print(f"  {'-' * 72}")
            print(f"  sweep compression total : {sweep_compress:13.3f} s")
            print(
                f"  method grand total      : {grand:13.3f} s  "
                f"(calibration once + every sparsity's compression)"
                + ("" if calib_s is not None else "   [calibration time missing]")
            )

    if args.csv and csv_rows:
        fields = [
            "model", "method", "sparsity", "calibration_s",
            "prepare_s", "backbone_s", "compress_only_s", "wall_s", "tag",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
