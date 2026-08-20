from __future__ import annotations

import argparse
import json
import logging
import gc
import types
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_pipeline.transformers_compat import install_cache_compat_shim
install_cache_compat_shim()
try:
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()
except Exception:
    pass

from moe_pipeline.eval.lm_eval import run_lm_eval
from moe_pipeline.eval.vram import reset_vram_stats
from moe_pipeline.methods.parser import ensure_parser_forward
from moe_pipeline.model_adapter import HfModelAdapter
from moe_pipeline.serialization_utils import (
    install_parser_forward_alias,
    install_modulelist_forward_unimplemented_alias,
    sanitize_loaded_block,
)
from moe_pipeline.sparse_utils import is_sparse_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model with saved blocks.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--blocks-dir", required=True, help="Directory containing original blocks")
    parser.add_argument("--compressed-dir", required=False, help="Directory with compressed blocks")
    parser.add_argument(
        "--output-dir",
        default="auto",
        help="Directory to store eval results (or 'auto')",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default=None,
        help="HF device map (e.g. auto) to shard across GPUs",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="HF attention backend to pass to from_pretrained.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tasks", nargs="+", default=["wikitext"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", default="all", help="all | last:k | start:end | comma list")
    return parser.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return None


def resolve_hf_torch_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    resolved = resolve_dtype(dtype)
    if resolved is None:
        raise ValueError(f"Unsupported dtype for HF loader: {dtype}")
    return resolved


def resolve_device_map(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "no"}:
        return None
    return text


def _infer_module_device(module: torch.nn.Module, fallback: str) -> torch.device:
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        device = torch.device(execution_device)
        if device.type != "meta":
            return device
    for param in module.parameters(recurse=True):
        if param.device.type != "meta":
            return param.device
    for buf in module.buffers(recurse=True):
        if buf.device.type != "meta":
            return buf.device
    fallback_device = torch.device(fallback)
    if fallback_device.type == "meta":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return fallback_device


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        if value.device.type == "meta":
            return value
        if device.type == "meta":
            return value
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _attach_local_device_align_hook(module: torch.nn.Module, device: torch.device) -> None:
    if getattr(module, "_local_device_align_hook", False):
        module._local_device_align_device = device
        return

    original_forward = module.forward

    def _wrapped_forward(self, *args, **kwargs):
        moved_args = tuple(_move_to_device(arg, self._local_device_align_device) for arg in args)
        moved_kwargs = {
            key: _move_to_device(value, self._local_device_align_device)
            for key, value in kwargs.items()
        }
        return self._local_device_align_original_forward(*moved_args, **moved_kwargs)

    module._local_device_align_original_forward = original_forward
    module._local_device_align_device = device
    module.forward = types.MethodType(_wrapped_forward, module)
    module._local_device_align_hook = True


def _remove_local_device_align_hooks(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if not getattr(submodule, "_local_device_align_hook", False):
            continue
        original_forward = getattr(submodule, "_local_device_align_original_forward", None)
        if original_forward is not None:
            submodule.forward = original_forward
        for attr in (
            "_local_device_align_original_forward",
            "_local_device_align_device",
            "_local_device_align_hook",
        ):
            try:
                delattr(submodule, attr)
            except AttributeError:
                pass


def _model_has_sparse_tensors(module: torch.nn.Module) -> bool:
    for param in module.parameters(recurse=True):
        if is_sparse_tensor(param):
            return True
    for buf in module.buffers(recurse=True):
        if is_sparse_tensor(buf):
            return True
    return False


def _redispatch_model(model: torch.nn.Module, log: logging.Logger) -> None:
    """Re-dispatch model after block replacement.

    When blocks have structural changes (e.g. expert pruning removes entire
    expert modules), the original hf_device_map keys no longer match the
    current module tree.  In that case we infer a fresh device map from the
    modified model so that accelerate can dispatch correctly.
    """
    try:
        from accelerate import dispatch_model, infer_auto_device_map  # type: ignore
        from accelerate.utils import get_max_memory  # type: ignore
    except Exception as exc:
        log.warning("device_map set but accelerate is unavailable: %s", exc)
        return

    old_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(old_device_map, dict):
        log.warning("device_map set but model has no hf_device_map; skipping redispatch")
        return

    current_named = {name for name, _ in model.named_modules() if name}
    old_keys = set(old_device_map.keys())
    stale_keys = old_keys - current_named
    if not stale_keys:
        log.info(
            "block structure is unchanged; skipping accelerate redispatch and "
            "reusing per-block local device hooks"
        )
        return

    _remove_local_device_align_hooks(model)

    if stale_keys:
        log.info(
            "old device_map has %d stale keys (structure changed); "
            "inferring new device_map. stale sample: %s",
            len(stale_keys),
            list(stale_keys)[:5],
        )
        max_memory = get_max_memory()
        no_split = getattr(model, "_no_split_modules", None) or []
        new_device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
        )
        unique_devices = sorted(set(str(d) for d in new_device_map.values()))
        cpu_entries = [k for k, v in new_device_map.items() if str(v) == "cpu"]
        log.info(
            "new device_map: %d entries, devices=%s, cpu_entries=%d",
            len(new_device_map),
            unique_devices,
            len(cpu_entries),
        )
        if cpu_entries:
            log.warning(
                "new device_map includes CPU entries (first 10): %s",
                cpu_entries[:10],
            )
        dispatch_model(model, device_map=new_device_map)
        log.info("re-dispatched model with NEW inferred device_map after block replacement")
    else:
        dispatch_model(model, device_map=old_device_map)
        log.info("re-dispatched model with original hf_device_map after block replacement")


def _log_current_vram(log: logging.Logger, note: str) -> None:
    if not torch.cuda.is_available():
        return
    total_alloc = 0.0
    total_reserved = 0.0
    per_device = []
    for idx in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        except RuntimeError as exc:
            log.warning("skip vram log for cuda:%d due to runtime error: %s", idx, exc)
            continue
        total_alloc += allocated
        total_reserved += reserved
        per_device.append((idx, allocated, reserved))
    log.info(
        "%s vram allocated=%.2f GB reserved=%.2f GB",
        note,
        total_alloc,
        total_reserved,
    )
    for idx, allocated, reserved in per_device:
        log.info(
            "%s vram cuda:%d allocated=%.2f GB reserved=%.2f GB",
            note,
            idx,
            allocated,
            reserved,
        )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _align_block_attn_implementation(block: torch.nn.Module, impl: str) -> int:
    """Force every sub-config in a loaded block to use `impl` for attention.

    Saved blocks pickle their own copy of the model config, which bakes in the
    ``_attn_implementation`` used at save time. Injecting such a block into a
    host model loaded with a different backend leaves the injected layers using
    the stale backend. Reconcile to the host implementation.
    """
    if not impl:
        return 0
    seen: set[int] = set()
    count = 0
    for sub in block.modules():
        cfg = getattr(sub, "config", None)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        if hasattr(cfg, "_attn_implementation"):
            try:
                cfg._attn_implementation = impl
                count += 1
            except Exception:
                pass
    return count


def main() -> None:
    setup_logging()
    log = logging.getLogger("evaluate_blocks")
    args = parse_args()
    log.info(
        "start model=%s blocks_dir=%s compressed_dir=%s output_dir=%s device=%s device_map=%s",
        args.model,
        args.blocks_dir,
        args.compressed_dir,
        args.output_dir,
        args.device,
        args.device_map,
    )
    if args.output_dir == "auto":
        from moe_pipeline.output_paths import default_eval_dir

        output_dir = default_eval_dir(
            compressed_dir=Path(args.compressed_dir) if args.compressed_dir else None,
            blocks_dir=Path(args.blocks_dir),
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("resolved output_dir=%s", output_dir)

    torch_dtype = resolve_hf_torch_dtype(args.dtype)
    device_map = resolve_device_map(args.device_map)
    log.info("loading model %s (dtype=%s)", args.model, args.dtype)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    log.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_map is None:
        model.to(args.device)
        input_device = args.device
        log.info("model loaded to %s", args.device)
    else:
        input_device = str(_infer_module_device(model, "cuda:0" if torch.cuda.is_available() else "cpu"))
        log.info("model loaded with device_map=%s (input_device=%s)", device_map, input_device)
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            cpu_layers = [name for name, dev in hf_device_map.items() if str(dev) == "cpu"]
            log.info(
                "hf_device_map devices=%d cpu_layers=%d",
                len(set(hf_device_map.values())),
                len(cpu_layers),
            )
            if cpu_layers:
                preview = ", ".join(cpu_layers[:10])
                log.warning("hf_device_map includes CPU layers (first 10): %s", preview)
        else:
            log.info("hf_device_map not available on model")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_current_vram(log, "model load")

    patched = ensure_parser_forward(model)
    if patched:
        log.info("registered parser_forward for %d MoE expert classes", patched)

    adapter = HfModelAdapter(model)
    blocks_dir = Path(args.blocks_dir) / "original"
    compressed_dir = Path(args.compressed_dir) if args.compressed_dir else None

    total_blocks = adapter.block_count()
    layer_indices = parse_layer_spec(args.layers, total_blocks)
    total_selected = len(layer_indices)
    log.info("loading blocks total=%d selected=%d", total_blocks, total_selected)
    if not layer_indices:
        log.warning("no layers selected; leaving model blocks untouched")
    for pos, idx in enumerate(layer_indices, start=1):
        compressed_path = (
            compressed_dir / f"layer_{idx}.pt" if compressed_dir else None
        )
        if compressed_path and compressed_path.exists():
            block = _load_block(compressed_path, "cpu")
            source = "compressed"
        else:
            original_path = blocks_dir / f"layer_{idx}.pt"
            if original_path.exists():
                block = _load_block(original_path, "cpu")
                source = "original"
            else:
                log.warning("missing block layer_%d.pt, skipping", idx)
                continue
        host_attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if host_attn_impl:
            n_aligned = _align_block_attn_implementation(block, host_attn_impl)
            if n_aligned:
                log.info(
                    "aligned block layer_%d.pt attn_implementation -> %s (%d sub-config(s))",
                    idx, host_attn_impl, n_aligned,
                )
        log.info("setting block %d/%d (layer_%d.pt) from %s", pos, total_selected, idx, source)
        if device_map is None:
            target_device = torch.device(args.device)
            old_block = None
        else:
            current_block = adapter.blocks()[idx]
            target_device = _infer_module_device(current_block, input_device)
            old_block = current_block
        block = block.to(target_device)
        if device_map is not None:
            _attach_local_device_align_hook(block, target_device)
        adapter.set_block(idx, block)

        del block
        if old_block is not None:
            del old_block
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if device_map is not None:
        _redispatch_model(model, log)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log_current_vram(log, "post block replace")

    if torch.cuda.is_available():
        reset_vram_stats()
        log.info("reset peak VRAM stats before evaluation")

    log.info(
        "running lm-eval tasks=%s batch_size=%d num_fewshot=%d limit=%s",
        args.tasks,
        args.batch_size,
        args.num_fewshot,
        args.limit,
    )
    eval_device = input_device if device_map is not None else args.device
    lm_results, effective_batch = run_lm_eval(
        model,
        tokenizer,
        tasks=args.tasks,
        device=eval_device,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        token_counter=None,
    )
    hf_device_map = getattr(model, "hf_device_map", None)
    hf_device_map_serialized = None
    hf_device_map_devices = None
    hf_device_map_cpu_layers = None
    if isinstance(hf_device_map, dict):
        hf_device_map_serialized = {name: str(dev) for name, dev in hf_device_map.items()}
        hf_device_map_devices = sorted({str(dev) for dev in hf_device_map.values()})
        hf_device_map_cpu_layers = [name for name, dev in hf_device_map_serialized.items() if dev == "cpu"]
    results = {
        "tasks": args.tasks,
        "effective_batch_size": effective_batch,
        "config": {
            "model": args.model,
            "blocks_dir": str(args.blocks_dir),
            "compressed_dir": str(args.compressed_dir) if args.compressed_dir else None,
            "output_dir": str(output_dir),
            "dtype": args.dtype,
            "device": args.device,
            "device_map": device_map,
            "cache_dir": args.cache_dir,
            "tasks": list(args.tasks),
            "batch_size": args.batch_size,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "layers": args.layers,
            "resolved_layers": list(layer_indices),
            "total_blocks": total_blocks,
            "input_device": input_device,
            "parser_forward_patched": int(patched),
            "hf_device_map": hf_device_map_serialized,
            "hf_device_map_devices": hf_device_map_devices,
            "hf_device_map_cpu_layers": hf_device_map_cpu_layers,
        },
        "lm_eval": {},
    }
    for task, payload in lm_results.items():
        task_payload = {
            "primary_metric": payload.primary_metric,
            "primary_value": payload.primary_value,
            "metrics": payload.metrics,
            "eval_time_sec": payload.eval_time_sec,
            "fewshot": payload.fewshot,
            "vram_total_allocated_mb": payload.vram_total_allocated_mb,
            "vram_total_reserved_mb": payload.vram_total_reserved_mb,
            "vram_per_device_allocated_mb": payload.vram_per_device_allocated_mb,
            "vram_per_device_reserved_mb": payload.vram_per_device_reserved_mb,
            "vram_concurrent_peak_allocated_mb": payload.vram_concurrent_peak_allocated_mb,
            "vram_concurrent_peak_reserved_mb": payload.vram_concurrent_peak_reserved_mb,
            "vram_per_device_allocated_at_concurrent_peak_mb": (
                payload.vram_per_device_allocated_at_concurrent_peak_mb
            ),
            "vram_per_device_reserved_at_concurrent_peak_mb": (
                payload.vram_per_device_reserved_at_concurrent_peak_mb
            ),
            "vram_concurrent_peak_sample_count": payload.vram_concurrent_peak_sample_count,
            "vram_concurrent_peak_sample_interval_sec": (
                payload.vram_concurrent_peak_sample_interval_sec
            ),
            "vram_model_size_mb": payload.vram_model_size_mb,
            "vram_per_device_model_size_mb": payload.vram_per_device_model_size_mb,
        }
        if payload.tokens is not None:
            task_payload["tokens"] = payload.tokens
            task_payload["tokens_per_sec"] = payload.tokens_per_sec
        results["lm_eval"][task] = task_payload

    with (output_dir / "eval_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    for task, payload in lm_results.items():
        if payload.tokens_per_sec is not None:
            log.info(
                "%s %s=%.4f time=%.2fs tps=%.2f tokens=%d",
                task,
                payload.primary_metric,
                payload.primary_value,
                payload.eval_time_sec,
                payload.tokens_per_sec,
                payload.tokens,
            )
        else:
            log.info(
                "%s %s=%.4f time=%.2fs",
                task,
                payload.primary_metric,
                payload.primary_value,
                payload.eval_time_sec,
            )
        if payload.vram_total_allocated_mb is not None:
            log.info(
                "%s vram sum_per_device_peaks allocated=%.2f MB, reserved=%.2f MB",
                task,
                payload.vram_total_allocated_mb,
                payload.vram_total_reserved_mb or 0.0,
            )
            if payload.vram_per_device_allocated_mb and payload.vram_per_device_reserved_mb:
                for device, allocated in payload.vram_per_device_allocated_mb.items():
                    reserved = payload.vram_per_device_reserved_mb.get(device, 0.0)
                    log.info(
                        "%s vram %s allocated=%.2f MB, reserved=%.2f MB",
                        task,
                        device,
                        allocated,
                        reserved,
                    )
        if payload.vram_concurrent_peak_allocated_mb is not None:
            log.info(
                "%s vram concurrent_peak allocated=%.2f MB, reserved=%.2f MB "
                "(samples=%s interval=%.3fs)",
                task,
                payload.vram_concurrent_peak_allocated_mb,
                payload.vram_concurrent_peak_reserved_mb or 0.0,
                payload.vram_concurrent_peak_sample_count,
                payload.vram_concurrent_peak_sample_interval_sec or 0.0,
            )
            if (
                payload.vram_per_device_allocated_at_concurrent_peak_mb
                and payload.vram_per_device_reserved_at_concurrent_peak_mb
            ):
                for device, allocated in payload.vram_per_device_allocated_at_concurrent_peak_mb.items():
                    reserved = payload.vram_per_device_reserved_at_concurrent_peak_mb.get(device, 0.0)
                    log.info(
                        "%s vram concurrent_peak %s allocated=%.2f MB, reserved=%.2f MB",
                        task,
                        device,
                        allocated,
                        reserved,
                    )
        if payload.vram_model_size_mb is not None:
            log.info("%s vram model_size=%.2f MB", task, payload.vram_model_size_mb)
            if payload.vram_per_device_model_size_mb:
                for device, model_size in payload.vram_per_device_model_size_mb.items():
                    log.info(
                        "%s vram model_size %s=%.2f MB",
                        task,
                        device,
                        model_size,
                    )


if __name__ == "__main__":
    main()
