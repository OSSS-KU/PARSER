from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_pipeline.transformers_compat import install_cache_compat_shim
install_cache_compat_shim()

from moe_pipeline.calibration.parser_activation import collect_activation_stats
from moe_pipeline.calibration.parser_activation import ROUTING_WEIGHT_STATS_VERSION
from moe_pipeline.calibration.parser_activation import normalize_exclude_domains
from moe_pipeline.model_adapter import HfModelAdapter, describe_model
from moe_pipeline.serialization_utils import remove_runtime_forward_wrappers
from moe_pipeline.phase_timing import global_timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save transformer blocks to disk.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument(
        "--output-dir",
        default="auto",
        help="Directory to store blocks (or 'auto')",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
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
    parser.add_argument("--calibrate", action="store_true", help="Collect activation stats for Parser.")
    parser.add_argument(
        "--calibration-dataset",
        default="wikitext",
        help="Calibration dataset name(s), comma-separated for multiple (config/text field inferred).",
    )
    parser.add_argument(
        "--calibration-split",
        default="train",
        help="Calibration split(s), comma-separated to align with datasets.",
    )
    parser.add_argument(
        "--calibration-samples",
        default="128",
        help="Calibration sample count (int) or 'auto' to use the full dataset.",
    )
    parser.add_argument(
        "--calibration-seq-len",
        default="256",
        help="Calibration token length (int), 'auto', or 'auto:<cap>' to clamp.",
    )
    parser.add_argument(
        "--calibration-padding",
        default="longest",
        choices=["max_length", "longest"],
        help="Padding strategy for calibration tokenization.",
    )
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument("--calibration-max-expert-samples", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument(
        "--calibration-streaming",
        action="store_true",
        help="Use streaming dataset loading for calibration.",
    )
    parser.add_argument(
        "--calibration-shuffle-buffer",
        type=int,
        default=None,
        help="Shuffle buffer size for streaming calibration datasets.",
    )
    parser.add_argument(
        "--calibration-packed",
        action="store_true",
        help=(
            "Concat-then-chunk calibration (GPTQ/Wanda recipe). "
            "Each sample becomes exactly --calibration-seq-len real tokens, no padding. "
            "When set, --calibration-padding is ignored."
        ),
    )
    parser.add_argument(
        "--calibration-exclude-domains",
        default="",
        help=(
            "Comma-separated SlimPajama subdomain names to exclude from calibration "
            "(e.g. 'RedPajamaCommonCrawl' or short alias 'commoncrawl'). "
            "Reads each item's meta.redpajama_set_name field. Empty = no filtering. "
            "Aliases: cc/commoncrawl, c4, github/code, books/book, arxiv, wiki/wikipedia, se/stackexchange."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip saving if blocks already exist.")
    parser.add_argument(
        "--timing-methods",
        type=str,
        default="",
        help=(
            "Comma-separated compression method list (the runner's METHODS). "
            "Calibration wall time is recorded per method name in "
            "blocks/calibration_timing.json so downstream timing aggregation is "
            "purely per-method. Methods sharing a calibration artifact each get "
            "the same measured time."
        ),
    )
    return parser.parse_args()


def _calibration_kind(method: str) -> Optional[str]:
    """Map a compression method name to the calibration artifact it consumes.

    parser* -> activation_stats. Returns None for methods with no calibration
    (e.g. 'original').
    """
    m = (method or "").strip().lower()
    if not m or m in ("original", "dense", "none", "uncompressed"):
        return None
    if m.startswith("parser"):
        return "activation"
    return None


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


def _infer_non_meta_device(module: torch.nn.Module, fallback: str) -> str:
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        device = torch.device(execution_device)
        if device.type != "meta":
            return str(device)
    for param in module.parameters(recurse=True):
        if param.device.type != "meta":
            return str(param.device)
    for buf in module.buffers(recurse=True):
        if buf.device.type != "meta":
            return str(buf.device)
    fallback_device = torch.device(fallback)
    if fallback_device.type == "meta":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(fallback_device)


def _materialize_block_to_cpu(block: torch.nn.Module) -> torch.nn.Module:
    """Move a block to CPU, materializing accelerate-offloaded weights.

    With ``device_map=auto`` accelerate may offload whole layers to the CPU,
    leaving meta-device placeholders backed by an offload weights map. A plain
    ``block.to("cpu")`` then raises ``NotImplementedError: Cannot copy out of
    meta tensor``. For each submodule with offloaded params we onload the real
    weights to CPU (via accelerate's own hook), capture them, and reinstate
    them permanently with offloading disabled, so the block can be pickled.
    """
    from accelerate.utils import (
        align_module_device,
        has_offloaded_params,
        named_module_tensors,
        set_module_tensor_to_device,
    )

    for submodule in block.modules():
        if not has_offloaded_params(submodule):
            continue
        hook = submodule._hf_hook
        captured: dict[str, torch.Tensor] = {}
        with align_module_device(submodule, "cpu"):
            for name, tensor in named_module_tensors(
                submodule,
                include_buffers=hook.offload_buffers,
                recurse=hook.place_submodules,
                remove_non_persistent=True,
            ):
                if tensor.device.type != "meta":
                    captured[name] = tensor.detach().to("cpu").clone()
        hook.offload = False
        for name, value in captured.items():
            set_module_tensor_to_device(submodule, name, "cpu", value=value)

    return block.to("cpu")


def resolve_calibration_samples(value: object, *, log: logging.Logger | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"auto", "all", "none"}:
        if log:
            log.info("calibration samples set to auto (defer to dataset sizing)")
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid calibration samples value: {value!r}") from exc


def format_calibration_samples(value: int | None) -> str:
    return "all" if value is None else str(value)


def _split_csv(value: object) -> list[str]:
    text = "" if value is None else str(value)
    return [item.strip() for item in text.split(",")]


def _align_optional_list(
    raw: object,
    count: int,
    *,
    field_name: str,
    default_value: Optional[str],
) -> list[Optional[str]]:
    items = _split_csv(raw)
    if len(items) == 1 and count > 1:
        item = items[0] or default_value
        return [item] * count
    if len(items) != count:
        raise ValueError(
            f"{field_name} expects 1 or {count} entries, got {len(items)}: {items!r}"
        )
    return [item or default_value for item in items]


def build_arc_prompt(item: dict) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    question = item.get("question")
    if not question:
        return None
    parts = [f"Question: {question}"]
    choices = item.get("choices")
    if isinstance(choices, dict):
        labels = choices.get("label") or []
        texts = choices.get("text") or []
        if texts:
            lines = []
            for idx, text in enumerate(texts):
                label = labels[idx] if idx < len(labels) else chr(ord("A") + idx)
                if text:
                    lines.append(f"{label}) {text}")
            if lines:
                parts.append("Choices: " + " ".join(lines))
    answer = item.get("answerKey")
    if answer:
        parts.append(f"Answer: {answer}")
    return "\n".join(parts)


def build_hellaswag_prompt(item: dict) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    endings = item.get("endings") or []
    if not isinstance(endings, (list, tuple)):
        endings = []
    label = item.get("label", 0)
    try:
        label_idx = int(label)
    except Exception:
        label_idx = 0
    chosen = ""
    if endings:
        if 0 <= label_idx < len(endings):
            chosen = endings[label_idx]
        else:
            chosen = endings[0]
    ctx = item.get("ctx", "")
    if isinstance(ctx, str) and isinstance(chosen, str) and ctx and chosen:
        return f"{ctx.strip()} {chosen.strip()}"
    return None


def build_dolly_prompt(item: dict) -> Optional[str]:
    """Build instruction-style prompt from a Databricks Dolly-15k record.

    Dolly records expose `instruction`, optional `context`, and `response`. We
    serialize them with explicit role markers ("Instruction:", "Context:",
    "Response:") so MoE routing receives the same QA-format token sequences
    that appear in evaluation prompts (Question:/Answer:/A)/B)/...). The
    instruction-style markers are NOT a verbatim match for benchmark templates
    but cover the broader "instruction-following" routing pattern that MoE
    experts learn during pretraining.
    """
    if not isinstance(item, dict):
        return None
    instruction = item.get("instruction") or ""
    context = item.get("context") or ""
    response = item.get("response") or ""
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None
    instruction = instruction.strip()
    response = response.strip()
    if not instruction or not response:
        return None
    if isinstance(context, str) and context.strip():
        return (
            f"Instruction: {instruction}\n"
            f"Context: {context.strip()}\n"
            f"Response: {response}"
        )
    return f"Instruction: {instruction}\nResponse: {response}"


PROMPT_BUILDERS = {
    "arc_prompt": build_arc_prompt,
    "hellaswag_prompt": build_hellaswag_prompt,
    "dolly_prompt": build_dolly_prompt,
}


CALIBRATION_DATASET_DEFAULTS = {
    "wikitext": {
        "dataset": "wikitext",
        "config": "wikitext-2-raw-v1",
        "text_field": "text",
        "prompt_builder": None,
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "config": None,
        "text_field": "ctx",
        "prompt_builder": "hellaswag_prompt",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "text_field": "question",
        "prompt_builder": "arc_prompt",
    },
    "c4": {
        "dataset": "allenai/c4",
        "config": "en",
        "text_field": "text",
        "prompt_builder": None,
    },
    "slimpajama-6b": {
        "dataset": "DKYoon/SlimPajama-6B",
        "config": None,
        "text_field": "text",
        "prompt_builder": None,
    },
    "slimpajama": {
        "dataset": "DKYoon/SlimPajama-6B",
        "config": None,
        "text_field": "text",
        "prompt_builder": None,
    },
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "config": None,
        "text_field": "instruction",
        "prompt_builder": "dolly_prompt",
    },
    "dolly-15k": {
        "dataset": "databricks/databricks-dolly-15k",
        "config": None,
        "text_field": "instruction",
        "prompt_builder": "dolly_prompt",
    },
    "databricks-dolly-15k": {
        "dataset": "databricks/databricks-dolly-15k",
        "config": None,
        "text_field": "instruction",
        "prompt_builder": "dolly_prompt",
    },
}


def _dataset_key(name: str) -> str:
    return name.strip().lower().split("/")[-1]


def resolve_calibration_dataset(name: str, log: logging.Logger) -> dict:
    key = _dataset_key(name)
    defaults = CALIBRATION_DATASET_DEFAULTS.get(key)
    if defaults:
        dataset_name = defaults.get("dataset") or name
        if dataset_name != name:
            log.info("calibration dataset alias %s -> %s", name, dataset_name)
        builder_name = defaults.get("prompt_builder")
        builder_fn = None
        if builder_name:
            builder_fn = PROMPT_BUILDERS.get(builder_name)
            if builder_fn is None:
                log.warning(
                    "prompt builder %s not found for %s; falling back to text_field",
                    builder_name,
                    name,
                )
        return {
            "dataset": dataset_name,
            "config": defaults["config"],
            "text_field": defaults["text_field"],
            "prompt_builder": builder_fn,
            "prompt_builder_name": builder_name,
            "mapped_from": key,
        }
    log.warning(
        "no calibration dataset mapping for %s; using config=None text_field='text'",
        name,
    )
    return {
        "dataset": name,
        "config": None,
        "text_field": "text",
        "prompt_builder": None,
        "prompt_builder_name": None,
        "mapped_from": key,
    }


def parse_calibration_sources(args: argparse.Namespace, log: logging.Logger) -> list[dict]:
    names = [name for name in _split_csv(args.calibration_dataset) if name]
    if not names:
        raise ValueError("calibration dataset is empty")
    splits = _align_optional_list(
        args.calibration_split,
        len(names),
        field_name="calibration-split",
        default_value="train",
    )
    sources = []
    for name, split in zip(names, splits):
        resolved = resolve_calibration_dataset(name, log)
        sources.append(
            {
                "name": resolved["dataset"],
                "config": resolved["config"],
                "split": split,
                "text_field": resolved["text_field"],
                "prompt_builder": resolved.get("prompt_builder"),
                "prompt_builder_name": resolved.get("prompt_builder_name"),
            }
        )
    if len(sources) > 1:
        log.info(
            "calibration datasets=%s configs=%s splits=%s text_fields=%s prompt_builders=%s",
            [s["name"] for s in sources],
            [s["config"] for s in sources],
            [s["split"] for s in sources],
            [s["text_field"] for s in sources],
            [s["prompt_builder_name"] for s in sources],
        )
    return sources


def calibration_metadata_fields(sources: list[dict]) -> dict:
    def _builder_name(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return getattr(value, "__name__", "custom")

    if len(sources) == 1:
        src = sources[0]
        return {
            "dataset": src["name"],
            "dataset_config": src["config"],
            "split": src["split"],
            "text_field": src["text_field"],
            "prompt_builder": _builder_name(src.get("prompt_builder_name") or src.get("prompt_builder")),
        }
    return {
        "dataset": [src["name"] for src in sources],
        "dataset_config": [src["config"] for src in sources],
        "split": [src["split"] for src in sources],
        "text_field": [src["text_field"] for src in sources],
        "prompt_builder": [
            _builder_name(src.get("prompt_builder_name") or src.get("prompt_builder"))
            for src in sources
        ],
    }


def format_calibration_datasets(sources: list[dict]) -> str:
    names = [src["name"] for src in sources]
    return ",".join(names)


def format_calibration_splits(sources: list[dict]) -> str:
    splits = [src["split"] for src in sources]
    if all(split == splits[0] for split in splits):
        return splits[0]
    return ",".join(splits)


def resolve_calibration_seq_len(
    value: object,
    model,
    *,
    tokenizer=None,
    log: logging.Logger | None = None,
) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("calibration seq len is empty")
    lowered = text.lower()
    cap = None
    auto_prefixes = ("auto", "max", "model_max")
    if any(lowered == prefix or lowered.startswith(f"{prefix}:") for prefix in auto_prefixes):
        for prefix in auto_prefixes:
            if lowered.startswith(f"{prefix}:"):
                cap_text = lowered.split(":", 1)[1].strip()
                if cap_text:
                    cap = int(cap_text)
                break
    else:
        return int(text)

    candidates: list[int] = []
    config = getattr(model, "config", None)
    if config is not None:
        for key in (
            "max_position_embeddings",
            "seq_length",
            "n_positions",
            "max_seq_len",
            "max_sequence_length",
        ):
            val = getattr(config, key, None)
            if isinstance(val, int) and val > 0:
                candidates.append(val)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            original = rope_scaling.get("original_max_position_embeddings")
            factor = rope_scaling.get("factor")
            if isinstance(original, int) and original > 0 and isinstance(factor, (int, float)) and factor > 1:
                candidates.append(int(original * factor))

    if tokenizer is not None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 10**6:
            candidates.append(tok_max)

    if candidates:
        resolved = min(candidates)
        if cap is not None:
            resolved = min(resolved, cap)
        if log:
            if cap is None:
                log.info(
                    "calibration seq_len auto resolved to %d (candidates=%s)",
                    resolved,
                    candidates,
                )
            else:
                log.info(
                    "calibration seq_len auto resolved to %d (candidates=%s cap=%d)",
                    resolved,
                    candidates,
                    cap,
                )
        return resolved

    fallback = 2048
    if log:
        log.warning(
            "unable to resolve calibration seq len from model config; fallback to %d",
            fallback,
        )
    return fallback


def _metadata_mismatches(metadata: dict, expected: dict) -> list[tuple[str, object, object]]:
    mismatches: list[tuple[str, object, object]] = []
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            mismatches.append((key, metadata.get(key), expected_value))
    return mismatches


def _format_mismatches(mismatches: list[tuple[str, object, object]]) -> str:
    return ", ".join(
        f"{key} cached={cached!r} expected={expected!r}"
        for key, cached, expected in mismatches
    )


def _resolve_shuffle_buffer(
    *,
    streaming: bool,
    shuffle_buffer: int | None,
    num_samples: int | None,
) -> int | None:
    if not streaming:
        return shuffle_buffer
    if shuffle_buffer is not None:
        return int(shuffle_buffer)
    if num_samples is not None:
        return max(1000, min(10000, int(num_samples) * 4))
    return 10000


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _activation_sample_count(entry: object) -> int:
    if not isinstance(entry, dict):
        return 0
    samples = entry.get("samples")
    if samples is None:
        return 0
    if torch.is_tensor(samples):
        return int(samples.shape[0])
    try:
        return len(samples)
    except Exception:
        return 0


def _summarize_activation_samples(
    activation_stats: dict, log: logging.Logger, *, max_show: int = 20
) -> None:
    experts = activation_stats.get("experts") if isinstance(activation_stats, dict) else None
    if not isinstance(experts, dict):
        log.warning("activation stats missing experts payload; skip sample validation")
        return

    empty_keys: list[str] = []
    min_count: int | None = None
    min_keys: list[str] = []

    for key, entry in experts.items():
        count = _activation_sample_count(entry)
        if min_count is None or count < min_count:
            min_count = count
            min_keys = [key]
        elif count == min_count:
            min_keys.append(key)
        if count == 0:
            empty_keys.append(key)

    if min_count is None:
        log.warning("activation stats have no expert entries")
        return

    def _format_keys(keys: list[str]) -> str:
        if not keys:
            return "<none>"
        shown = keys[:max_show]
        suffix = ""
        if len(keys) > max_show:
            suffix = f", ...(+{len(keys) - max_show} more)"
        return ", ".join(shown) + suffix

    log.info(
        "activation stats: min_samples=%d experts_with_min=%d total_experts=%d min_keys=%s",
        min_count,
        len(min_keys),
        len(experts),
        _format_keys(min_keys),
    )

    if empty_keys:
        log.error(
            "activation stats: empty-sample experts=%d keys=%s",
            len(empty_keys),
            _format_keys(empty_keys),
        )
        raise RuntimeError(
            "Found experts with zero calibration samples. "
            "Increase --calibration-samples or adjust the dataset/sequence length."
        )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger("save_blocks")
    args = parse_args()
    args.calibration_samples = resolve_calibration_samples(
        args.calibration_samples,
        log=log,
    )
    calibration_sources = parse_calibration_sources(args, log)
    exclude_domains = normalize_exclude_domains(args.calibration_exclude_domains)
    if exclude_domains:
        log.info("calibration exclude_domains=%s", sorted(exclude_domains))
    log.info(
        "start model=%s device=%s device_map=%s dtype=%s output_dir=%s",
        args.model,
        args.device,
        args.device_map,
        args.dtype,
        args.output_dir,
    )
    if args.output_dir == "auto":
        from moe_pipeline.output_paths import default_blocks_dir

        output_dir = default_blocks_dir(args.model)
    else:
        output_dir = Path(args.output_dir)
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    log.info("resolved output_dir=%s", output_dir)

    skip_blocks = False
    if args.skip_existing:
        existing = list(original_dir.glob("layer_*.pt"))
        if existing:
            skip_blocks = True
            if not args.calibrate:
                log.info("skip: found %d existing blocks in %s", len(existing), original_dir)
                return
            log.info("skip saving blocks (found %d existing blocks)", len(existing))

    torch_dtype = resolve_hf_torch_dtype(args.dtype)
    device_map = resolve_device_map(args.device_map)
    log.info("loading model %s (dtype=%s device_map=%s)", args.model, args.dtype, device_map or "none")
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device_map is None:
        model.to(args.device)
        input_device = args.device
        log.info("model loaded to %s", args.device)
    else:
        input_device = _infer_non_meta_device(model, "cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("model loaded with device_map=%s (input_device=%s)", device_map, input_device)

    activation_path = output_dir / "activation_stats.pt"
    shuffle_buffer = _resolve_shuffle_buffer(
        streaming=args.calibration_streaming,
        shuffle_buffer=args.calibration_shuffle_buffer,
        num_samples=args.calibration_samples,
    )

    tokenizer_for_calibration = None
    if args.calibrate:
        seq_len_raw = str(args.calibration_seq_len).strip().lower()
        auto_prefixes = ("auto", "max", "model_max")
        needs_auto = any(
            seq_len_raw == prefix or seq_len_raw.startswith(f"{prefix}:")
            for prefix in auto_prefixes
        )
        if needs_auto:
            log.info("resolving calibration seq len using model/tokenizer (auto)")
            tokenizer_for_calibration = AutoTokenizer.from_pretrained(
                args.model,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            if tokenizer_for_calibration.pad_token is None:
                tokenizer_for_calibration.pad_token = tokenizer_for_calibration.eos_token
            args.calibration_seq_len = resolve_calibration_seq_len(
                args.calibration_seq_len,
                model,
                tokenizer=tokenizer_for_calibration,
                log=log,
            )
        else:
            args.calibration_seq_len = resolve_calibration_seq_len(
                args.calibration_seq_len,
                model,
                log=log,
            )

    reuse_activation = False
    if args.calibrate and activation_path.exists():
        try:
            cached = torch.load(activation_path, map_location="cpu")
        except Exception as exc:
            log.warning("failed to read %s (%s), recomputing", activation_path, exc)
        else:
            meta = cached.get("metadata") if isinstance(cached, dict) else None
            expected_meta = {
                "model_id": args.model,
                **calibration_metadata_fields(calibration_sources),
                "num_samples": args.calibration_samples,
                "seq_len": args.calibration_seq_len,
                "batch_size": args.calibration_batch_size,
                "padding": args.calibration_padding,
                "max_expert_samples": args.calibration_max_expert_samples,
                "seed": args.calibration_seed,
                "streaming": args.calibration_streaming,
                "shuffle_buffer": shuffle_buffer,
                "routing_weight_stats": ROUTING_WEIGHT_STATS_VERSION,
                "packed": bool(args.calibration_packed),
                "exclude_domains": sorted(exclude_domains),
            }
            if isinstance(meta, dict):
                mismatches = _metadata_mismatches(meta, expected_meta)
                if not mismatches:
                    _summarize_activation_samples(cached, log)
                    reuse_activation = True
                    log.info("reuse activation stats from %s (metadata match)", activation_path)
                else:
                    log.info(
                        "activation stats metadata mismatch: %s",
                        _format_mismatches(mismatches),
                    )
            else:
                log.info("activation stats missing metadata, recomputing")

    need_activation = args.calibrate and not reuse_activation

    if need_activation:
        log.info("set calibration seed=%s", args.calibration_seed)
        seed_everything(args.calibration_seed)
        if tokenizer_for_calibration is None:
            log.info("loading tokenizer for calibration")
            tokenizer_for_calibration = AutoTokenizer.from_pretrained(
                args.model,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            if tokenizer_for_calibration.pad_token is None:
                tokenizer_for_calibration.pad_token = tokenizer_for_calibration.eos_token
        tokenizer = tokenizer_for_calibration
        if need_activation:
            log.info(
                "collecting activation stats dataset=%s split=%s samples=%s batch_size=%d seq_len=%d padding=%s packed=%s",
                format_calibration_datasets(calibration_sources),
                format_calibration_splits(calibration_sources),
                format_calibration_samples(args.calibration_samples),
                args.calibration_batch_size,
                args.calibration_seq_len,
                args.calibration_padding if not args.calibration_packed else "<ignored:packed>",
                bool(args.calibration_packed),
            )
            _cal_t0 = time.perf_counter()
            activation_stats = collect_activation_stats(
                model,
                tokenizer,
                dataset_sources=calibration_sources,
                num_samples=args.calibration_samples,
                seq_len=args.calibration_seq_len,
                batch_size=args.calibration_batch_size,
                padding=args.calibration_padding,
                max_expert_samples=args.calibration_max_expert_samples,
                device=input_device,
                seed=args.calibration_seed,
                streaming=args.calibration_streaming,
                shuffle_buffer=shuffle_buffer,
                packed=bool(args.calibration_packed),
                exclude_domains=exclude_domains,
            )
            global_timer().add("calibration_activation", time.perf_counter() - _cal_t0)
            _summarize_activation_samples(activation_stats, log)
            activation_stats.setdefault("metadata", {})["model_id"] = args.model
            torch.save(activation_stats, activation_path)
            log.info("activation stats saved to %s", activation_path)

        timing = global_timer().as_dict()
        if timing["phases"]:
            phases = timing["phases"]
            kind_seconds = {
                "activation": phases.get("calibration_activation", {}).get("seconds"),
            }
            timing_path = output_dir / "calibration_timing.json"
            existing: dict = {}
            if timing_path.exists():
                try:
                    existing = json.loads(timing_path.read_text())
                except Exception:
                    existing = {}
            methods_map = existing.get("methods", {})
            if not isinstance(methods_map, dict):
                methods_map = {}
            requested = [m.strip() for m in (args.timing_methods or "").split(",") if m.strip()]
            recorded = []
            for method in requested:
                kind = _calibration_kind(method)
                if kind is None:
                    continue
                secs = kind_seconds.get(kind)
                if secs is None:
                    continue
                methods_map[method] = {
                    "calibration_kind": kind,
                    "calibration_seconds": round(float(secs), 4),
                }
                recorded.append(f"{method}={secs:.1f}s")
            payload = {
                "model_id": args.model,
                "methods": methods_map,
            }
            timing_path.write_text(json.dumps(payload, indent=2))
            log.info(
                "calibration timing recorded per method in %s (%s)",
                timing_path,
                ", ".join(recorded) if recorded else "no --timing-methods given",
            )

    adapter = HfModelAdapter(model)
    blocks = adapter.blocks()

    model_dtype = str(model.dtype)

    if not skip_blocks:
        total_blocks = len(blocks)
        log.info("saving %d blocks to %s", total_blocks, original_dir)
        for idx, block in enumerate(blocks):
            log.info("saving block %d/%d (layer_%d.pt)", idx + 1, total_blocks, idx)
            block_cpu = _materialize_block_to_cpu(block)
            restored_wrappers = remove_runtime_forward_wrappers(block_cpu)
            if restored_wrappers:
                log.info(
                    "removed %d accelerate forward wrapper(s) from layer_%d before save",
                    restored_wrappers,
                    idx,
            )
            torch.save(block_cpu, original_dir / f"layer_{idx}.pt")
            block_cpu.to("meta")
            del block_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest = {
        "model": describe_model(model),
        "model_id": args.model,
        "block_container_path": adapter.container_path,
        "num_layers": len(blocks),
        "dtype": model_dtype,
        "activation_stats": bool(args.calibrate),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    log.info("saved %d blocks to %s", len(blocks), original_dir)


if __name__ == "__main__":
    main()
