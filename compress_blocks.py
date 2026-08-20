from __future__ import annotations

import argparse
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from moe_pipeline.transformers_compat import install_cache_compat_shim
install_cache_compat_shim()
try:
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()
except Exception:
    pass

from moe_pipeline.methods import build_compressor
from moe_pipeline.methods.base import CompressionSpec
from moe_pipeline.moe_introspection import find_moe_blocks
from moe_pipeline.phase_timing import global_timer
from moe_pipeline.serialization_utils import (
    install_parser_forward_alias,
    install_modulelist_forward_unimplemented_alias,
    sanitize_loaded_block,
)
from moe_pipeline.sparse_utils import count_nonzero_tensor, is_sparse_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress saved transformer blocks.")
    parser.add_argument("--blocks-dir", required=True, help="Directory containing manifest.json")
    parser.add_argument(
        "--output-dir",
        default="auto",
        help="Directory to store compressed blocks (or 'auto')",
    )
    parser.add_argument(
        "--method",
        default="parser-l-sp",
        help=(
            "Compression method: parser-l-sp (layer scope / global pooling) or "
            "parser-e-sp (expert scope / local selection); or 'original' for a no-op passthrough."
        ),
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--backbone", default="mean", choices=["mean", "barycenter"])
    parser.add_argument(
        "--importance",
        default="formula",
        choices=[
            "formula",
            "wanda",
        ],
        help="Importance metric for parser methods.",
    )
    parser.add_argument(
        "--parser-routing-score",
        default="on",
        choices=["on", "off", "1", "0", "true", "false", "yes", "no"],
        help="Whether parser methods multiply importance by pre-top-k router softmax^2 stats.",
    )
    parser.add_argument("--min-units", type=int, default=4)
    parser.add_argument("--layers", default="all", help="all | last:k | start:end | comma list")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration", default=None, help="Path to activation_stats.pt")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip compression if output layer files already exist",
    )
    return parser.parse_args()


def parse_layer_spec(spec: str, total_layers: int) -> List[int]:
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(range(total_layers))
    if spec.startswith("last:"):
        k = int(spec.split(":", 1)[1])
        start = max(total_layers - k, 0)
        return list(range(start, total_layers))
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_layers
        return list(range(start, min(end, total_layers)))
    indices = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    return sorted(set(i for i in indices if 0 <= i < total_layers))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_block(path: Path, device: str):
    install_modulelist_forward_unimplemented_alias()
    try:
        block = torch.load(path, map_location=device, weights_only=False)
    except AttributeError as exc:
        if "parser_forward" not in str(exc):
            raise
        install_parser_forward_alias()
        block = torch.load(path, map_location=device, weights_only=False)
    except ModuleNotFoundError as exc:
        if not str(exc).startswith("No module named 'transformers_modules"):
            raise
        try:
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        except Exception:
            pass
        block = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        block = torch.load(path, map_location=device)
    if isinstance(block, torch.nn.Module):
        sanitize_loaded_block(block)
    return block


def _count_unique_parameters(
    module: torch.nn.Module,
    *,
    seen: set[int],
    include_nonzero: bool,
) -> Tuple[int, Optional[int]]:
    total = 0
    nonzero = 0 if include_nonzero else None
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.numel())
        if include_nonzero:
            nonzero += count_nonzero_tensor(param)
    for _, buffer in module.named_buffers():
        if (
            not torch.is_tensor(buffer)
            or not is_sparse_tensor(buffer)
            or not (buffer.is_floating_point() or buffer.is_complex())
        ):
            continue
        bid = id(buffer)
        if bid in seen:
            continue
        seen.add(bid)
        total += int(buffer.numel())
        if include_nonzero:
            nonzero += count_nonzero_tensor(buffer)
    return total, nonzero


def _count_moe_parameters(
    block: torch.nn.Module,
    *,
    include_nonzero: bool,
) -> Tuple[int, Optional[int], int]:
    seen: set[int] = set()
    total = 0
    nonzero = 0 if include_nonzero else None
    moe_blocks = 0
    for _, moe_block in find_moe_blocks(block):
        moe_blocks += 1
        part_total, part_nonzero = _count_unique_parameters(
            moe_block,
            seen=seen,
            include_nonzero=include_nonzero,
        )
        total += part_total
        if include_nonzero:
            nonzero += int(part_nonzero or 0)

        for module in moe_block.modules():
            shared = getattr(module, "_shared_backbone", None)
            if not isinstance(shared, torch.nn.Module):
                continue
            shared_total, shared_nonzero = _count_unique_parameters(
                shared,
                seen=seen,
                include_nonzero=include_nonzero,
            )
            total += shared_total
            if include_nonzero:
                nonzero += int(shared_nonzero or 0)

    return total, nonzero, moe_blocks


def _use_nonzero_param_metric(method: str, residual_mode: str) -> bool:
    method = (method or "").strip().lower()
    if method.startswith("parser"):
        return residual_mode == "unstructured"
    return False


def main() -> None:
    setup_logging()
    log = logging.getLogger("compress_blocks")
    args = parse_args()
    method = (args.method or "").strip().lower()
    if not method:
        raise ValueError("Method must be non-empty.")
    residual_mode = "structured"
    residual_scope = "expert"
    importance = (args.importance or "formula").strip().lower()
    parser_routing_score = str(args.parser_routing_score).strip().lower() in {
        "on",
        "1",
        "true",
        "yes",
    }
    if method.startswith("parser"):
        if method == "parser-l-sp":
            residual_scope = "layer"
            residual_mode = "structured"
        elif method == "parser-e-sp":
            residual_scope = "expert"
            residual_mode = "structured"
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Supported: 'parser-l-sp' "
                "(layer scope / global pooling) or 'parser-e-sp' (expert scope / local selection)."
            )
    elif importance != "formula":
        log.warning("importance=%s is only used by parser; falling back to formula", importance)
        importance = "formula"
    log.info(
        "start blocks_dir=%s output_dir=%s method=%s sparsity=%.4f importance=%s "
        "parser_routing_score=%s device=%s",
        args.blocks_dir,
        args.output_dir,
        method,
        args.sparsity,
        importance,
        parser_routing_score,
        args.device,
    )
    blocks_dir = Path(args.blocks_dir)
    original_dir = blocks_dir / "original"
    output_dir = Path(args.output_dir)

    manifest_path = blocks_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        total_layers = int(manifest.get("num_layers", 0))
        log.info("loaded manifest: total_layers=%d", total_layers)
    else:
        total_layers = len(list(original_dir.glob("layer_*.pt")))
        log.info("manifest missing: inferred total_layers=%d", total_layers)

    layer_indices = parse_layer_spec(args.layers, total_layers)
    log.info("layers selected: %d", len(layer_indices))

    spec = CompressionSpec(
        method=method,
        sparsity=args.sparsity,
        residual_mode=residual_mode,
        residual_scope=residual_scope,
        backbone=args.backbone,
        importance=importance,
        parser_routing_score=parser_routing_score,
        min_units=args.min_units,
    )
    log.info("compression spec=%s", spec)
    if args.output_dir == "auto":
        from moe_pipeline.output_paths import default_compressed_dir

        output_dir = default_compressed_dir(
            blocks_dir,
            method=method,
            sparsity=args.sparsity,
            residual_mode=residual_mode,
            residual_scope=residual_scope,
            backbone=args.backbone,
            importance=importance,
            parser_routing_score=parser_routing_score,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("output_dir=%s", output_dir)

    if method == "original":
        log.info("method=original: skipping compression, copying original blocks to output_dir")
        copied = 0
        skipped = 0
        for idx in layer_indices:
            src_path = original_dir / f"layer_{idx}.pt"
            if not src_path.exists():
                log.warning("missing original block layer_%d.pt at %s, skipping", idx, src_path)
                continue
            dst_path = output_dir / f"layer_{idx}.pt"
            if args.skip_existing and dst_path.exists():
                skipped += 1
                continue
            shutil.copy2(src_path, dst_path)
            copied += 1
        log.info(
            "method=original: copied=%d skipped(existing)=%d total_selected=%d",
            copied,
            skipped,
            len(layer_indices),
        )
        original_manifest = {
            "method": "original",
            "sparsity": 0.0,
            "residual_mode": "na",
            "residual_scope": "na",
            "backbone": args.backbone,
            "importance": "na",
            "min_units": args.min_units,
            "layers": layer_indices,
            "calibration": None,
            "output_dir": str(output_dir),
            "metrics": {
                "moe_param_count_rule": "noop",
                "moe_param_count_effective_mode": "noop",
                "original_moe_params": 0,
                "compressed_moe_params": 0,
                "compressed_moe_nonzero_params": 0,
                "compressed_moe_effective_params": 0,
                "moe_param_reduction_pct": 0.0,
                "moe_blocks": 0,
                "per_layer_moe_params": [],
            },
        }
        with (output_dir / "compression_manifest.json").open("w") as f:
            json.dump(original_manifest, f, indent=2)
        log.info("method=original: wrote no-op manifest at %s", output_dir / "compression_manifest.json")
        return

    if args.skip_existing and layer_indices:
        expected = set(layer_indices)
        existing = {
            idx for idx in expected if (output_dir / f"layer_{idx}.pt").exists()
        }
        if existing == expected:
            log.info(
                "all %d selected layers already compressed; reusing files and recomputing metrics",
                len(layer_indices),
            )

    activation_stats = None
    calibration_path = None
    if method.startswith("parser"):
        calibration_path = Path(args.calibration) if args.calibration else (blocks_dir / "activation_stats.pt")
        if not calibration_path.exists():
            raise FileNotFoundError(
                "Activation stats not found. Provide --calibration or run save_blocks.py with --calibrate."
            )
        log.info("loading activation stats from %s", calibration_path)
        activation_stats = torch.load(calibration_path, map_location="cpu")

    compressor = build_compressor(
        spec,
        calibration_stats=activation_stats,
    )

    use_nonzero_metric = _use_nonzero_param_metric(
        method=method,
        residual_mode=residual_mode,
    )
    metric_rule = "nonzero_if_unstructured_else_numel"
    effective_mode = "nonzero" if use_nonzero_metric else "numel"
    total_original_moe_params = 0
    total_compressed_moe_params = 0
    total_compressed_moe_nonzero_params = 0
    total_compressed_moe_effective_params = 0
    total_moe_blocks = 0
    per_layer_moe_metrics: Dict[str, Dict[str, int]] = {}

    def _accumulate_param_metrics(
        *,
        layer_idx: int,
        original_block: torch.nn.Module,
        compressed_block: torch.nn.Module,
    ) -> None:
        nonlocal total_original_moe_params
        nonlocal total_compressed_moe_params
        nonlocal total_compressed_moe_nonzero_params
        nonlocal total_compressed_moe_effective_params
        nonlocal total_moe_blocks

        original_moe_params, _, _ = _count_moe_parameters(
            original_block,
            include_nonzero=False,
        )
        compressed_moe_params, compressed_moe_nonzero_params_raw, moe_blocks = _count_moe_parameters(
            compressed_block,
            include_nonzero=use_nonzero_metric,
        )
        if use_nonzero_metric:
            compressed_moe_nonzero_params = int(compressed_moe_nonzero_params_raw or 0)
        else:
            compressed_moe_nonzero_params = int(compressed_moe_params)
        effective_params = (
            compressed_moe_nonzero_params
            if use_nonzero_metric
            else compressed_moe_params
        )

        total_original_moe_params += original_moe_params
        total_compressed_moe_params += compressed_moe_params
        total_compressed_moe_nonzero_params += compressed_moe_nonzero_params
        total_compressed_moe_effective_params += effective_params
        total_moe_blocks += moe_blocks
        per_layer_moe_metrics[str(layer_idx)] = {
            "original_moe_params": int(original_moe_params),
            "compressed_moe_params": int(compressed_moe_params),
            "compressed_moe_nonzero_params": int(compressed_moe_nonzero_params),
            "compressed_moe_effective_params": int(effective_params),
            "moe_blocks": int(moe_blocks),
        }

    timer = global_timer()

    _t0 = time.perf_counter()
    compressor.prepare(
        total_layers, layer_indices,
        blocks_dir=blocks_dir, device=args.device,
    )
    timer.add("prepare", time.perf_counter() - _t0)

    total_blocks = len(layer_indices)
    _compress_loop_t0 = time.perf_counter()
    for pos, idx in enumerate(layer_indices, start=1):
        block_path = original_dir / f"layer_{idx}.pt"
        if not block_path.exists():
            log.warning("missing block %s, skipping", block_path)
            continue
        if args.skip_existing:
            output_path = output_dir / f"layer_{idx}.pt"
            if output_path.exists():
                log.info(
                    "skipping existing compressed block %d/%d (layer_%d.pt)",
                    pos,
                    total_blocks,
                    idx,
                )
                original_block_for_metrics = _load_block(block_path, "cpu")
                compressed_block_for_metrics = _load_block(output_path, "cpu")
                _accumulate_param_metrics(
                    layer_idx=idx,
                    original_block=original_block_for_metrics,
                    compressed_block=compressed_block_for_metrics,
                )
                del original_block_for_metrics, compressed_block_for_metrics
                continue
        log.info("compressing block %d/%d (layer_%d.pt)", pos, total_blocks, idx)
        block = _load_block(block_path, args.device)
        block = block.to(args.device)
        compressed = compressor.compress_block(block, idx)
        compressed = compressed.to("cpu")
        _accumulate_param_metrics(
            layer_idx=idx,
            original_block=block,
            compressed_block=compressed,
        )
        torch.save(compressed, output_dir / f"layer_{idx}.pt")
        del block, compressed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    timer.add("compress_loop", time.perf_counter() - _compress_loop_t0)

    reduction_pct = 0.0
    if total_original_moe_params > 0:
        reduction_pct = (1.0 - (total_compressed_moe_effective_params / total_original_moe_params)) * 100.0

    log.info(
        "moe params original=%d compressed=%d compressed_nonzero=%d effective(%s)=%d reduction=%.2f%%",
        total_original_moe_params,
        total_compressed_moe_params,
        total_compressed_moe_nonzero_params,
        effective_mode,
        total_compressed_moe_effective_params,
        reduction_pct,
    )

    compression_manifest = {
        "method": method,
        "sparsity": args.sparsity,
        "residual_mode": residual_mode,
        "residual_scope": residual_scope,
        "backbone": args.backbone,
        "importance": importance,
        "min_units": args.min_units,
        "layers": layer_indices,
        "calibration": str(calibration_path) if calibration_path else None,
        "output_dir": str(output_dir),
        "metrics": {
            "moe_param_count_rule": metric_rule,
            "moe_param_count_effective_mode": effective_mode,
            "original_moe_params": int(total_original_moe_params),
            "compressed_moe_params": int(total_compressed_moe_params),
            "compressed_moe_nonzero_params": int(total_compressed_moe_nonzero_params),
            "compressed_moe_effective_params": int(total_compressed_moe_effective_params),
            "moe_param_reduction_pct": float(reduction_pct),
            "moe_blocks": int(total_moe_blocks),
            "per_layer_moe_params": per_layer_moe_metrics,
        },
    }
    with (output_dir / "compression_manifest.json").open("w") as f:
        json.dump(compression_manifest, f, indent=2)

    timing = timer.as_dict()
    prepare_s = timing["phases"].get("prepare", {}).get("seconds", 0.0)
    backbone_s = timing["phases"].get("backbone", {}).get("seconds", 0.0)
    loop_s = timing["phases"].get("compress_loop", {}).get("seconds", 0.0)
    timing["derived"] = {
        "prepare_seconds": round(prepare_s, 4),
        "backbone_seconds": round(backbone_s, 4),
        "compress_only_seconds": round(loop_s - backbone_s, 4),
        "wall_seconds": round(prepare_s + loop_s, 4),
    }
    timing["total_seconds"] = timing["derived"]["wall_seconds"]
    timing["method"] = method
    timing["sparsity"] = args.sparsity
    with (output_dir / "phase_timing.json").open("w") as f:
        json.dump(timing, f, indent=2)
    log.info(
        "phase timing: prepare=%.1fs backbone=%.1fs compress_only=%.1fs",
        timing["phases"].get("prepare", {}).get("seconds", 0.0),
        backbone_s,
        loop_s - backbone_s,
    )

    log.info("compressed %d blocks to %s", len(layer_indices), output_dir)


if __name__ == "__main__":
    main()
