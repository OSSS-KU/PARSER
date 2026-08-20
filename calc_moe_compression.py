import argparse
import json
import os
import sys
from pathlib import Path

import torch

from moe_pipeline.moe_introspection import find_moe_blocks
from moe_pipeline.output_paths import default_blocks_dir
from moe_pipeline.serialization_utils import (
    install_modulelist_forward_unimplemented_alias,
    sanitize_loaded_block,
)
from moe_pipeline.sparse_utils import count_nonzero_tensor, is_sparse_tensor


def _ensure_transformers_modules_on_path(modules_cache: str | None) -> bool:
    cache_path = None
    if modules_cache:
        cache_path = Path(modules_cache)
    else:
        try:
            from transformers.utils.hub import TRANSFORMERS_MODULES_CACHE
        except Exception:
            return False
        cache_path = Path(TRANSFORMERS_MODULES_CACHE)
    if not cache_path or not cache_path.exists():
        return False
    cache_str = str(cache_path)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)
    return True


def _install_parser_forward() -> None:
    """Install ``parser_forward`` on all known expert classes in sys.modules.

    Parser-compressed blocks pickle a reference to ``expert.parser_forward``.
    If the class hasn't been patched yet, ``torch.load`` will fail with
    ``AttributeError``.  This mirrors the pattern used by
    ``install_modulelist_forward_unimplemented_alias``.
    """
    try:
        from moe_pipeline.methods.parser import parser_forward as pf
        from moe_pipeline.methods.parser import parser_moe_forward as pmf
    except ImportError:
        return
    from moe_pipeline.moe_introspection import (
        DEEPSEEK_EXPERT_CLASS_NAMES,
        MOE_BLOCK_CLASS_NAMES,
        OLMOE_EXPERT_CLASS_NAMES,
        QWEN_EXPERT_CLASS_NAMES,
    )

    all_names = (
        QWEN_EXPERT_CLASS_NAMES
        | DEEPSEEK_EXPERT_CLASS_NAMES
        | OLMOE_EXPERT_CLASS_NAMES
    )
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for name in all_names:
            try:
                cls = getattr(mod, name, None)
                if cls is not None and isinstance(cls, type) and not hasattr(cls, "parser_forward"):
                    setattr(cls, "parser_forward", pf)
            except Exception:
                continue
        for name in MOE_BLOCK_CLASS_NAMES:
            try:
                cls = getattr(mod, name, None)
                if cls is not None and isinstance(cls, type) and not hasattr(cls, "parser_moe_forward"):
                    setattr(cls, "parser_moe_forward", pmf)
            except Exception:
                continue


def load_block(path: Path) -> torch.nn.Module:
    install_modulelist_forward_unimplemented_alias()
    try:
        block = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("transformers_modules"):
            raise SystemExit(
                "Missing transformers dynamic modules. Add the modules cache to PYTHONPATH "
                "or pass --modules-cache (typically ~/.cache/huggingface/modules). "
                "If the cache is empty, run AutoConfig.from_pretrained(model_id, trust_remote_code=True) "
                "once to populate it."
            ) from exc
        raise
    except AttributeError as exc:
        if "parser_forward" in str(exc):
            _install_parser_forward()
            block = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise
    except TypeError:
        block = torch.load(path, map_location="cpu")
    if isinstance(block, torch.nn.Module):
        sanitize_loaded_block(block)
    return block


def count_moe_params(block: torch.nn.Module) -> tuple[int, int]:
    """Count **non-zero** MoE parameters in *block*.

    Using ``count_nonzero`` instead of ``numel`` gives accurate counts for
    both structured pruning (smaller tensors) and unstructured pruning
    (same-shaped tensors with zeroed-out entries).
    """
    seen: set[int] = set()
    total = 0
    moe_blocks = 0
    for _, moe in find_moe_blocks(block):
        moe_blocks += 1
        for p in moe.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += count_nonzero_tensor(p)
        for _, buffer in moe.named_buffers():
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
            total += count_nonzero_tensor(buffer)
        for module in moe.modules():
            if hasattr(module, "_shared_backbone"):
                backbone = getattr(module, "_shared_backbone")
                if isinstance(backbone, torch.nn.Module):
                    for p in backbone.parameters():
                        pid = id(p)
                        if pid in seen:
                            continue
                        seen.add(pid)
                        total += count_nonzero_tensor(p)
    return total, moe_blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MoE-only compression ratio for compressed blocks."
    )
    parser.add_argument("--blocks-dir", default=os.getenv("BLOCKS_DIR"))
    parser.add_argument("--compressed-dir", default=os.getenv("COMPRESSED_DIR"))
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--comp-tag", default=os.getenv("COMP_TAG"))
    parser.add_argument("--modules-cache", default=os.getenv("TRANSFORMERS_MODULES_CACHE"))
    parser.add_argument(
        "--mode",
        choices=["subset", "overall", "both"],
        default="both",
        help="subset=compressed layers only, overall=all layers, both=print both",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    blocks_dir = args.blocks_dir
    if not blocks_dir:
        if not args.model_id:
            raise SystemExit(
                "Need --blocks-dir or --model-id (or env BLOCKS_DIR/MODEL_ID)."
            )
        blocks_dir = str(default_blocks_dir(args.model_id))

    compressed_dir = args.compressed_dir
    if not compressed_dir:
        if not args.comp_tag:
            raise SystemExit(
                "Need --compressed-dir or --comp-tag (or env COMPRESSED_DIR/COMP_TAG)."
            )
        compressed_dir = str(Path(blocks_dir).parent / args.comp_tag / "compressed")

    return Path(blocks_dir), Path(compressed_dir)


def sum_layers(
    layer_list: list[int],
    orig_dir: Path,
    compressed_dir: Path,
    use_compressed: bool,
) -> tuple[int, int]:
    total = 0
    moe_blocks = 0
    for idx in layer_list:
        orig_path = orig_dir / f"layer_{idx}.pt"
        if not orig_path.exists():
            continue
        path = orig_path
        if use_compressed:
            comp_path = compressed_dir / f"layer_{idx}.pt"
            if comp_path.exists():
                path = comp_path
        params, blocks = count_moe_params(load_block(path))
        total += params
        moe_blocks += blocks
    return total, moe_blocks


def main() -> None:
    args = _parse_args()
    _ensure_transformers_modules_on_path(args.modules_cache)
    blocks_dir, compressed_dir = _resolve_paths(args)

    orig_dir = blocks_dir / "original"
    if not orig_dir.exists():
        raise SystemExit(f"original blocks not found: {orig_dir}")

    all_layers = sorted(int(p.stem.split("_")[1]) for p in orig_dir.glob("layer_*.pt"))

    cm_path = compressed_dir / "compression_manifest.json"
    subset_layers = None
    if cm_path.exists():
        cm = json.loads(cm_path.read_text())
        subset_layers = cm.get("layers")

    if not subset_layers:
        subset_layers = all_layers

    if args.mode in {"subset", "both"}:
        orig_subset, orig_subset_blocks = sum_layers(
            subset_layers, orig_dir, compressed_dir, False
        )
        if orig_subset_blocks == 0:
            raise SystemExit("No MoE blocks found in subset layers.")
        comp_subset, _ = sum_layers(subset_layers, orig_dir, compressed_dir, True)
        reduction = (1 - comp_subset / orig_subset) * 100 if orig_subset else 0.0
        print("=== subset (compressed layers only) ===")
        print(f"orig_moe_params={orig_subset:,}")
        print(f"comp_moe_params={comp_subset:,}")
        print(f"reduction={reduction:.2f}%")

    if args.mode in {"overall", "both"}:
        orig_all, orig_all_blocks = sum_layers(
            all_layers, orig_dir, compressed_dir, False
        )
        if orig_all_blocks == 0:
            raise SystemExit("No MoE blocks found in overall layers.")
        comp_all, _ = sum_layers(all_layers, orig_dir, compressed_dir, True)
        reduction = (1 - comp_all / orig_all) * 100 if orig_all else 0.0
        print("=== overall (all moe layers) ===")
        print(f"orig_moe_params={orig_all:,}")
        print(f"comp_moe_params={comp_all:,}")
        print(f"reduction={reduction:.2f}%")


if __name__ == "__main__":
    main()
