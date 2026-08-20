from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from moe_pipeline.eval.vram import (
    VramConcurrentPeakMonitor,
    collect_vram_stats,
    reset_vram_stats,
)

@dataclass
class LmEvalTaskResult:
    task: str
    primary_metric: str
    primary_value: float
    metrics: Dict[str, float]
    eval_time_sec: float
    fewshot: int
    tokens: Optional[int]
    tokens_per_sec: Optional[float]
    vram_total_allocated_mb: Optional[float]
    vram_total_reserved_mb: Optional[float]
    vram_per_device_allocated_mb: Optional[Dict[str, float]]
    vram_per_device_reserved_mb: Optional[Dict[str, float]]
    vram_concurrent_peak_allocated_mb: Optional[float]
    vram_concurrent_peak_reserved_mb: Optional[float]
    vram_per_device_allocated_at_concurrent_peak_mb: Optional[Dict[str, float]]
    vram_per_device_reserved_at_concurrent_peak_mb: Optional[Dict[str, float]]
    vram_concurrent_peak_sample_count: Optional[int]
    vram_concurrent_peak_sample_interval_sec: Optional[float]
    vram_model_size_mb: Optional[float]
    vram_per_device_model_size_mb: Optional[Dict[str, float]]


class TokenCounter:
    def __init__(self) -> None:
        self.total_tokens = 0
        self._wrapped = False
        self._orig_forward = None

    def reset(self) -> None:
        self.total_tokens = 0

    def attach(self, model) -> None:
        if self._wrapped:
            return
        self._orig_forward = model.forward

        def _count_tokens(args, kwargs) -> int:
            for key in ("input_ids", "decoder_input_ids"):
                if key in kwargs and kwargs[key] is not None and torch.is_tensor(kwargs[key]):
                    ids = kwargs[key]
                    mask = kwargs.get("attention_mask")
                    if mask is not None and torch.is_tensor(mask):
                        return int(mask.sum().item())
                    return int(ids.numel())
            if args:
                first = args[0]
                if torch.is_tensor(first):
                    return int(first.numel())
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and torch.is_tensor(embeds):
                return int(embeds.shape[0] * embeds.shape[1])
            return 0

        def wrapped_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
            self.total_tokens += _count_tokens(args, kwargs)
            return self._orig_forward(*args, **kwargs)

        model.forward = wrapped_forward  # type: ignore[method-assign]
        self._wrapped = True


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _parse_tasks(tasks: List[str], default_fewshot: int) -> Tuple[List[str], Dict[str, int]]:
    parsed: List[str] = []
    fewshot_map: Dict[str, int] = {}
    for raw in tasks:
        if ":" in raw:
            name, shot_str = raw.split(":", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Invalid task name in '{raw}'.")
            fewshot_map[name] = int(shot_str)
            parsed.append(name)
        else:
            parsed.append(raw)
    return parsed, fewshot_map


def _select_primary_metric(metrics: Dict[str, float]) -> Tuple[str, float]:
    candidates = [
        "acc,none",
        "acc",
        "accuracy",
        "exact_match",
        "f1",
        "pass@1",
        "pass@10",
        "perplexity,none",
        "perplexity",
        "ppl",
    ]
    for key in candidates:
        if key in metrics:
            return key, metrics[key]
    if metrics:
        first_key = next(iter(metrics))
        return first_key, metrics[first_key]
    return "unknown", 0.0


def _filter_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    keep = {}
    for key, value in metrics.items():
        key_lower = key.lower()
        if any(token in key_lower for token in ["acc", "accuracy", "perplexity", "ppl", "exact_match", "f1", "pass@"]):
            keep[key] = value
    return keep or metrics


def run_lm_eval(
    model,
    tokenizer,
    *,
    tasks: List[str],
    device: str,
    batch_size: int,
    num_fewshot: int,
    limit: Optional[int],
    token_counter: Optional[TokenCounter] = None,
    track_vram: bool = True,
) -> Tuple[Dict[str, LmEvalTaskResult], int]:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    parsed_tasks, fewshot_map = _parse_tasks(tasks, num_fewshot)
    current_batch = max(1, int(batch_size))

    def _build_lm(batch: int):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*pretrained.*model kwarg is not of type.*"
            )
            return HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch,
                max_batch_size=batch,
                use_accelerate=True,
            )

    if token_counter is not None:
        token_counter.attach(model)
    hf_model = _build_lm(current_batch)
    results: Dict[str, LmEvalTaskResult] = {}

    for task in parsed_tasks:
        task_fewshot = fewshot_map.get(task, num_fewshot)
        if track_vram:
            reset_vram_stats()
        if token_counter is not None:
            token_counter.reset()
        while True:
            start = time.time()
            vram_monitor = VramConcurrentPeakMonitor(interval_sec=0.05) if track_vram else None
            vram_concurrent = None
            if vram_monitor is not None:
                vram_monitor.start()
            try:
                lm_results = evaluator.simple_evaluate(
                    model=hf_model,
                    tasks=[task],
                    num_fewshot=task_fewshot,
                    limit=limit,
                    batch_size=current_batch,
                    max_batch_size=current_batch,
                    cache_requests=True,
                    rewrite_requests_cache=False,
                    delete_requests_cache=False,
                    confirm_run_unsafe_code=True,
                )
            except Exception as exc:
                if _is_oom(exc):
                    raise
                raise
            finally:
                if vram_monitor is not None:
                    vram_concurrent = vram_monitor.stop()

            elapsed = time.time() - start
            tokens = token_counter.total_tokens if token_counter is not None else None
            tps = tokens / elapsed if tokens is not None and tokens > 0 and elapsed > 0 else None
            vram = collect_vram_stats(model=model) if track_vram else None
            task_result = lm_results.get("results", {}).get(task, {})
            metrics = {
                key: float(value)
                for key, value in task_result.items()
                if isinstance(value, (int, float))
            }
            primary_metric, primary_value = _select_primary_metric(metrics)
            filtered = _filter_metrics(metrics)
            results[task] = LmEvalTaskResult(
                task=task,
                primary_metric=primary_metric,
                primary_value=float(primary_value),
                metrics=filtered,
                eval_time_sec=elapsed,
                fewshot=task_fewshot,
                tokens=tokens,
                tokens_per_sec=tps,
                vram_total_allocated_mb=vram.total_allocated_mb if vram else None,
                vram_total_reserved_mb=vram.total_reserved_mb if vram else None,
                vram_per_device_allocated_mb=vram.per_device_allocated_mb if vram else None,
                vram_per_device_reserved_mb=vram.per_device_reserved_mb if vram else None,
                vram_concurrent_peak_allocated_mb=(
                    vram_concurrent.total_allocated_mb if vram_concurrent else None
                ),
                vram_concurrent_peak_reserved_mb=(
                    vram_concurrent.total_reserved_mb if vram_concurrent else None
                ),
                vram_per_device_allocated_at_concurrent_peak_mb=(
                    vram_concurrent.per_device_allocated_mb if vram_concurrent else None
                ),
                vram_per_device_reserved_at_concurrent_peak_mb=(
                    vram_concurrent.per_device_reserved_mb if vram_concurrent else None
                ),
                vram_concurrent_peak_sample_count=(
                    vram_concurrent.sample_count if vram_concurrent else None
                ),
                vram_concurrent_peak_sample_interval_sec=(
                    vram_concurrent.interval_sec if vram_concurrent else None
                ),
                vram_model_size_mb=vram.model_size_mb if vram else None,
                vram_per_device_model_size_mb=vram.per_device_model_size_mb if vram else None,
            )
            break

    return results, current_batch
