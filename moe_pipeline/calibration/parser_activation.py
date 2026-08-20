from __future__ import annotations

import random
import types
from dataclasses import dataclass
import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from moe_pipeline.model_adapter import HfModelAdapter
from moe_pipeline.moe_introspection import expert_proj_attr_map, find_moe_blocks, get_experts
from moe_pipeline.methods.parser import combine_gate_up, make_expert_key, resolve_activation_fn

logger = logging.getLogger("calibration.activation")
ROUTING_WEIGHT_STATS_VERSION = "pre_topk_softmax_sq_v1"


def _format_samples(value: Optional[int]) -> str:
    return "all" if value is None else str(value)


def _normalize_dataset_sources(dataset_sources: Sequence[dict]) -> list[dict]:
    if not dataset_sources:
        raise ValueError("dataset_sources is empty")
    sources: list[dict] = []
    for entry in dataset_sources:
        if not isinstance(entry, dict):
            raise TypeError(f"dataset source must be dict, got {type(entry)!r}")
        name = entry.get("name")
        if not name:
            raise ValueError(f"dataset source missing name: {entry!r}")
        builder = entry.get("prompt_builder")
        builder_name = entry.get("prompt_builder_name")
        if builder_name is None:
            if callable(builder):
                builder_name = getattr(builder, "__name__", "custom")
            elif isinstance(builder, str):
                builder_name = builder
                builder = None
        if isinstance(builder, str):
            builder_name = builder_name or builder
            builder = None
        sources.append(
            {
                "name": str(name),
                "config": entry.get("config"),
                "split": entry.get("split") or "train",
                "text_field": entry.get("text_field") or "text",
                "prompt_builder": builder,
                "prompt_builder_name": builder_name,
            }
        )
    return sources


def _metadata_fields_from_sources(sources: list[dict]) -> dict:
    if len(sources) == 1:
        src = sources[0]
        return {
            "dataset": src["name"],
            "dataset_config": src["config"],
            "split": src["split"],
            "text_field": src["text_field"],
            "prompt_builder": src.get("prompt_builder_name"),
        }
    return {
        "dataset": [src["name"] for src in sources],
        "dataset_config": [src["config"] for src in sources],
        "split": [src["split"] for src in sources],
        "text_field": [src["text_field"] for src in sources],
        "prompt_builder": [src.get("prompt_builder_name") for src in sources],
    }


def _dataset_name_summary(sources: list[dict]) -> str:
    return ",".join(src["name"] for src in sources)


def _make_moe_key(layer_idx: int, moe_name: str) -> str:
    return f"{layer_idx}/{moe_name}"


def _resolve_router_module(moe_block: torch.nn.Module) -> Optional[torch.nn.Module]:
    for name in ("gate", "router", "router_gate", "router_linear", "gate_linear", "router_proj"):
        module = getattr(moe_block, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    return None


def _as_router_logits(output: object, num_experts: int) -> Optional[torch.Tensor]:
    logits = output
    if isinstance(logits, (tuple, list)):
        logits = logits[0] if logits else None
    if not torch.is_tensor(logits):
        return None
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if logits.dim() != 2 or logits.shape[-1] != num_experts:
        return None
    return logits


def _router_probs_from_module(
    module: torch.nn.Module,
    inputs: tuple,
    output: object,
    num_experts: int,
) -> Optional[torch.Tensor]:
    logits = _as_router_logits(output, num_experts)
    if logits is not None:
        if bool(getattr(module, "_router_output_is_probability", False)):
            return logits.detach().float()
        return torch.softmax(logits.detach().float(), dim=-1)

    if not inputs:
        return None
    hidden_states = inputs[0]
    if not torch.is_tensor(hidden_states):
        return None
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight):
        return None
    hidden_2d = hidden_states.detach()
    if hidden_2d.dim() == 3:
        hidden_2d = hidden_2d.reshape(-1, hidden_2d.shape[-1])
    elif hidden_2d.dim() != 2:
        return None
    if weight.dim() != 2 or weight.shape[0] != num_experts:
        return None
    logits = F.linear(hidden_2d.float(), weight.detach().float(), None)
    scoring_func = str(getattr(module, "scoring_func", "softmax") or "softmax").lower()
    if scoring_func == "sigmoid":
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def _resolve_samples_per_dataset(
    num_samples: Optional[int],
    datasets: Sequence,
    *,
    streaming: bool,
) -> list[Optional[int]]:
    num_datasets = len(datasets)
    if num_datasets == 1:
        return [num_samples]
    if num_samples is not None:
        base = num_samples // num_datasets
        remainder = num_samples % num_datasets
        return [base + (1 if i < remainder else 0) for i in range(num_datasets)]
    if streaming:
        raise ValueError(
            "calibration-samples=auto with multiple datasets requires explicit sample count "
            "when streaming is enabled."
        )
    lengths: list[int] = []
    for dataset in datasets:
        try:
            lengths.append(len(dataset))
        except Exception as exc:
            raise ValueError(
                "Unable to resolve dataset length for auto sampling across multiple datasets. "
                "Set --calibration-samples explicitly."
            ) from exc
    min_len = min(lengths) if lengths else 0
    logger.info(
        "calibration samples auto with multiple datasets: using %d per dataset (lengths=%s)",
        min_len,
        lengths,
    )
    return [min_len] * num_datasets


_DOMAIN_ALIASES = {
    "redpajamacommoncrawl": "RedPajamaCommonCrawl",
    "commoncrawl": "RedPajamaCommonCrawl",
    "cc": "RedPajamaCommonCrawl",
    "redpajamac4": "RedPajamaC4",
    "c4": "RedPajamaC4",
    "redpajamagithub": "RedPajamaGithub",
    "github": "RedPajamaGithub",
    "code": "RedPajamaGithub",
    "redpajamabook": "RedPajamaBook",
    "book": "RedPajamaBook",
    "books": "RedPajamaBook",
    "redpajamaarxiv": "RedPajamaArXiv",
    "arxiv": "RedPajamaArXiv",
    "redpajamawikipedia": "RedPajamaWikipedia",
    "wikipedia": "RedPajamaWikipedia",
    "wiki": "RedPajamaWikipedia",
    "redpajamastackexchange": "RedPajamaStackExchange",
    "stackexchange": "RedPajamaStackExchange",
    "se": "RedPajamaStackExchange",
}


def normalize_exclude_domains(values) -> frozenset:
    """Map user-supplied domain names to canonical SlimPajama set names.

    Accepts a comma-separated string, an iterable of strings, or None/empty.
    Returns a frozenset of canonical names (e.g. {'RedPajamaCommonCrawl'}).
    Unknown names are kept as-is so users can target non-SlimPajama tags.
    """
    if values is None:
        return frozenset()
    if isinstance(values, str):
        items = [v for v in values.split(",")]
    else:
        items = list(values)
    out: set = set()
    for raw in items:
        name = (raw or "").strip()
        if not name:
            continue
        canonical = _DOMAIN_ALIASES.get(name.lower(), name)
        out.add(canonical)
    return frozenset(out)


def _item_domain(item: object) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    meta = item.get("meta")
    if not isinstance(meta, dict):
        return None
    name = meta.get("redpajama_set_name")
    if isinstance(name, str) and name:
        return name
    return None


def _extract_text(
    item: object,
    text_field: str,
    *,
    prompt_builder=None,
    exclude_domains: Optional[frozenset] = None,
) -> Optional[str]:
    if exclude_domains:
        domain = _item_domain(item)
        if domain is not None and domain in exclude_domains:
            return None
    if callable(prompt_builder):
        built = prompt_builder(item)
        if isinstance(built, str) and built:
            return built
    if isinstance(item, dict):
        text = item.get(text_field)
        if text is None:
            text = item.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def _iter_text_batches_uniform(
    datasets: Sequence,
    *,
    text_fields: Sequence[str],
    prompt_builders: Sequence,
    batch_size: int,
    per_dataset_samples: Sequence[Optional[int]],
    sources: Sequence[dict],
    exclude_domains: Optional[frozenset] = None,
) -> Iterable[List[str]]:
    iterators = [iter(ds) for ds in datasets]
    counts = [0 for _ in datasets]
    done = [False for _ in datasets]
    remaining = len(datasets)
    batch: List[str] = []
    idx = 0
    while remaining > 0:
        if done[idx]:
            idx = (idx + 1) % len(datasets)
            continue
        quota = per_dataset_samples[idx]
        if quota is not None and counts[idx] >= quota:
            done[idx] = True
            remaining -= 1
            idx = (idx + 1) % len(datasets)
            continue
        try:
            item = next(iterators[idx])
        except StopIteration:
            done[idx] = True
            remaining -= 1
            idx = (idx + 1) % len(datasets)
            continue
        text = _extract_text(item, text_fields[idx], prompt_builder=prompt_builders[idx], exclude_domains=exclude_domains)
        if not text:
            idx = (idx + 1) % len(datasets)
            continue
        counts[idx] += 1
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
        idx = (idx + 1) % len(datasets)
    if batch:
        yield batch
    for i, quota in enumerate(per_dataset_samples):
        if quota is None:
            continue
        if counts[i] < quota:
            logger.warning(
                "dataset %s exhausted early: got %d/%d samples",
                sources[i]["name"],
                counts[i],
                quota,
            )


@dataclass
class ExpertSampleStats:
    samples: torch.Tensor
    seen: int
    hidden_dim: int


class ExpertInputCollector:
    def __init__(
        self,
        expert: torch.nn.Module,
        key: str,
        attr_map: Dict[str, str],
        *,
        max_samples: int,
        sample_dtype: torch.dtype,
    ) -> None:
        self.expert = expert
        self.key = key
        self.attr_map = attr_map
        self.max_samples = max(1, int(max_samples))
        self.sample_dtype = sample_dtype
        self.samples: Optional[torch.Tensor] = None
        self.filled: int = 0
        self.seen: int = 0
        gate_module = getattr(expert, attr_map["gate"])
        self.hidden_dim = int(gate_module.weight.shape[0])
        self.input_dim = int(gate_module.weight.shape[1])
        self.act_fn = resolve_activation_fn(expert)
        self.original_forward = expert.forward

    def _maybe_init_buffer(self) -> None:
        if self.samples is None:
            self.samples = torch.empty(
                (self.max_samples, self.input_dim),
                dtype=self.sample_dtype,
            )

    def patch(self) -> None:
        collector = self
        attr_map = self.attr_map

        def wrapped_forward(
            module_self,
            hidden_states: torch.Tensor,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            gate_module = getattr(module_self, attr_map["gate"])
            up_module = getattr(module_self, attr_map["up"])
            down_module = getattr(module_self, attr_map["down"])

            with torch.no_grad():
                collector._maybe_init_buffer()
                if collector.samples is None:
                    return down_module(
                        combine_gate_up(
                            collector.act_fn,
                            gate_module(hidden_states),
                            up_module(hidden_states),
                        )
                    )
                inputs_cpu = hidden_states.detach().to("cpu", dtype=collector.sample_dtype)
                if inputs_cpu.ndim == 1:
                    inputs_cpu = inputs_cpu.unsqueeze(0)
                for row in inputs_cpu:
                    collector.seen += 1
                    if collector.filled < collector.max_samples:
                        collector.samples[collector.filled] = row
                        collector.filled += 1
                    else:
                        j = random.randint(0, collector.seen - 1)
                        if j < collector.max_samples:
                            collector.samples[j] = row

            gate_out = gate_module(hidden_states)
            up_out = up_module(hidden_states)
            hidden = combine_gate_up(collector.act_fn, gate_out, up_out)
            return down_module(hidden)

        self.expert.forward = types.MethodType(wrapped_forward, self.expert)

    def restore(self) -> None:
        self.expert.forward = self.original_forward

    def to_stats(self) -> ExpertSampleStats:
        if self.samples is None:
            self.samples = torch.zeros(
                (0, self.input_dim), dtype=self.sample_dtype
            )
        if self.filled > 0:
            samples = self.samples[: self.filled].clone()
        else:
            samples = torch.zeros((0, self.input_dim), dtype=self.sample_dtype)
        return ExpertSampleStats(
            samples=samples.to(device="cpu"),
            seen=int(self.seen),
            hidden_dim=int(self.hidden_dim),
        )


def _iter_text_batches(
    dataset,
    *,
    text_field: str,
    prompt_builder=None,
    batch_size: int,
    num_samples: Optional[int],
    exclude_domains: Optional[frozenset] = None,
) -> Iterable[List[str]]:
    batch: List[str] = []
    seen = 0
    for item in dataset:
        text = _extract_text(item, text_field, prompt_builder=prompt_builder, exclude_domains=exclude_domains)
        if not text:
            continue
        batch.append(text)
        seen += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if num_samples is not None and seen >= num_samples:
            break
    if batch:
        yield batch


def _iter_packed_token_batches(
    datasets,
    *,
    text_fields: Sequence[str],
    prompt_builders: Sequence,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    per_dataset_samples: Sequence[Optional[int]],
    sources: Sequence[dict],
    exclude_domains: Optional[frozenset] = None,
) -> Iterable[Dict[str, torch.Tensor]]:
    """GPTQ/Wanda-style concat-then-chunk calibration loader.

    For each dataset source, pulls texts from the (already shuffled) iterator,
    tokenizes them without special tokens, appends them into a per-source buffer
    with EOS as separator, and emits seq_len-token chunks. Datasets are mixed
    via per-chunk round-robin so multi-source quotas are honored. Each emitted
    sample contains exactly seq_len real tokens; attention_mask is all ones.
    No padding is ever produced.
    """
    iterators = [iter(ds) for ds in datasets]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    sep: List[int] = [int(eos_id)] if isinstance(eos_id, int) else []
    buffers: List[List[int]] = [[] for _ in datasets]
    counts_emitted: List[int] = [0 for _ in datasets]
    done: List[bool] = [False for _ in datasets]
    pending_chunks: List[torch.Tensor] = []
    remaining = len(datasets)
    idx = 0

    while remaining > 0:
        if done[idx]:
            idx = (idx + 1) % len(datasets)
            continue
        quota = per_dataset_samples[idx]
        if quota is not None and counts_emitted[idx] >= quota:
            done[idx] = True
            remaining -= 1
            idx = (idx + 1) % len(datasets)
            continue

        while len(buffers[idx]) < seq_len:
            try:
                item = next(iterators[idx])
            except StopIteration:
                done[idx] = True
                remaining -= 1
                break
            text = _extract_text(item, text_fields[idx], prompt_builder=prompt_builders[idx], exclude_domains=exclude_domains)
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buffers[idx].extend(ids)
            if sep:
                buffers[idx].extend(sep)

        if len(buffers[idx]) >= seq_len:
            chunk = buffers[idx][:seq_len]
            buffers[idx] = buffers[idx][seq_len:]
            counts_emitted[idx] += 1
            pending_chunks.append(torch.tensor(chunk, dtype=torch.long))
            if len(pending_chunks) >= batch_size:
                input_ids = torch.stack(pending_chunks, dim=0)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                pending_chunks = []

        idx = (idx + 1) % len(datasets)

    if pending_chunks:
        input_ids = torch.stack(pending_chunks, dim=0)
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def collect_activation_stats(
    model,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset_sources: Sequence[dict],
    num_samples: Optional[int],
    seq_len: int,
    batch_size: int,
    padding: str,
    max_expert_samples: int,
    device: str,
    seed: int = 0,
    streaming: bool = False,
    shuffle_buffer: Optional[int] = None,
    packed: bool = False,
    exclude_domains: Optional[frozenset] = None,
) -> Dict[str, dict]:
    sources = _normalize_dataset_sources(dataset_sources)
    logger.info(
        "activation calibration start dataset=%s split=%s samples=%s batch_size=%d seq_len=%d padding=%s packed=%s exclude_domains=%s",
        _dataset_name_summary(sources),
        ",".join(src["split"] for src in sources),
        _format_samples(num_samples),
        batch_size,
        seq_len,
        padding if not packed else "<ignored:packed>",
        bool(packed),
        sorted(exclude_domains) if exclude_domains else "<none>",
    )
    model.eval()
    adapter = HfModelAdapter(model)

    collectors: List[ExpertInputCollector] = []
    router_stats: Dict[str, dict] = {}
    router_handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, block in enumerate(adapter.blocks()):
        for moe_name, moe_block in find_moe_blocks(block):
            experts = get_experts(moe_block)
            if not experts:
                continue
            attr_map = expert_proj_attr_map(experts[0])
            moe_key = _make_moe_key(layer_idx, moe_name)
            router_stats[moe_key] = {
                "prob_sum": torch.zeros(len(experts), dtype=torch.float64),
                "prob_sq_sum": torch.zeros(len(experts), dtype=torch.float64),
                "total_tokens": 0,
                "num_experts": len(experts),
            }
            router_module = _resolve_router_module(moe_block)
            if router_module is None:
                logger.warning(
                    "unable to locate router module for layer=%d moe=%s; routing weights disabled",
                    layer_idx,
                    moe_name,
                )
            else:
                def make_router_hook(key: str, num_experts: int):
                    def hook(module, inputs, output):
                        probs = _router_probs_from_module(module, inputs, output, num_experts)
                        if probs is None or probs.numel() == 0:
                            return
                        probs_cpu = probs.to("cpu", dtype=torch.float64)
                        entry = router_stats[key]
                        entry["prob_sum"] += probs_cpu.sum(dim=0)
                        entry["prob_sq_sum"] += probs_cpu.pow(2).sum(dim=0)
                        entry["total_tokens"] += int(probs.shape[0])

                    return hook

                router_handles.append(
                    router_module.register_forward_hook(
                        make_router_hook(moe_key, len(experts))
                    )
                )
            for expert_idx, expert in enumerate(experts):
                key = make_expert_key(layer_idx, moe_name, expert_idx)
                collector = ExpertInputCollector(
                    expert,
                    key,
                    attr_map,
                    max_samples=max_expert_samples,
                    sample_dtype=torch.float16,
                )
                collector.patch()
                collectors.append(collector)

    datasets = []
    shuffle_buffer_value = shuffle_buffer
    if streaming and shuffle_buffer_value is None:
        if num_samples is None:
            shuffle_buffer_value = 10000
        else:
            shuffle_buffer_value = max(1000, min(10000, int(num_samples) * 4))
    for src in sources:
        dataset = load_dataset(
            src["name"],
            src["config"],
            split=src["split"],
            streaming=streaming,
        )
        if seed is not None:
            if streaming:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_value, seed=seed)
            else:
                dataset = dataset.shuffle(seed=seed)
        datasets.append(dataset)

    per_dataset_samples = _resolve_samples_per_dataset(
        num_samples,
        datasets,
        streaming=streaming,
    )
    total_samples = None
    if per_dataset_samples and all(sample is not None for sample in per_dataset_samples):
        total_samples = sum(sample for sample in per_dataset_samples if sample is not None)
    total_batches = None
    if total_samples and batch_size:
        total_batches = math.ceil(total_samples / batch_size)
    log_every = 10
    if total_batches:
        log_every = max(1, total_batches // 10)

    batch_idx = 0

    def _log_batch(idx: int) -> None:
        if total_batches:
            if idx == 1 or idx % log_every == 0 or idx == total_batches:
                logger.info("activation calibration batch %d/%d", idx, total_batches)
        else:
            if idx == 1 or idx % log_every == 0:
                logger.info("activation calibration batch %d", idx)

    with torch.no_grad():
        if packed:
            for tokens in _iter_packed_token_batches(
                datasets,
                text_fields=[src_["text_field"] for src_ in sources],
                prompt_builders=[src_.get("prompt_builder") for src_ in sources],
                tokenizer=tokenizer,
                seq_len=seq_len,
                batch_size=batch_size,
                per_dataset_samples=per_dataset_samples,
                sources=sources,
                exclude_domains=exclude_domains,
            ):
                batch_idx += 1
                _log_batch(batch_idx)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                model(**tokens, use_cache=False)
        elif len(datasets) == 1:
            dataset = datasets[0]
            text_field = sources[0]["text_field"]
            for batch in _iter_text_batches(
                dataset,
                text_field=text_field,
                prompt_builder=sources[0].get("prompt_builder"),
                batch_size=batch_size,
                num_samples=per_dataset_samples[0],
                exclude_domains=exclude_domains,
            ):
                batch_idx += 1
                _log_batch(batch_idx)
                tokens = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=padding,
                    truncation=True,
                    max_length=seq_len,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                model(**tokens, use_cache=False)
        else:
            for batch in _iter_text_batches_uniform(
                datasets,
                text_fields=[src_["text_field"] for src_ in sources],
                prompt_builders=[src_.get("prompt_builder") for src_ in sources],
                batch_size=batch_size,
                per_dataset_samples=per_dataset_samples,
                sources=sources,
                exclude_domains=exclude_domains,
            ):
                batch_idx += 1
                _log_batch(batch_idx)
                tokens = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=padding,
                    truncation=True,
                    max_length=seq_len,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                model(**tokens, use_cache=False)
    for collector in collectors:
        collector.restore()
    for handle in router_handles:
        handle.remove()

    for moe_key, entry in sorted(router_stats.items()):
        total_tokens = int(entry.get("total_tokens", 0))
        if total_tokens <= 0:
            logger.info("routing softmax stats %s: no router tokens collected", moe_key)
            continue
        prob_sum = entry["prob_sum"].to(dtype=torch.float64)
        prob_sq_sum = entry["prob_sq_sum"].to(dtype=torch.float64)
        mean_prob = prob_sum / float(total_tokens)
        sq_factor = prob_sq_sum / float(total_tokens)
        logger.info(
            "routing softmax stats %s: tokens=%d mean_prob[min=%.6g mean=%.6g max=%.6g] "
            "sq_factor[min=%.6g mean=%.6g max=%.6g]",
            moe_key,
            total_tokens,
            float(mean_prob.min().item()),
            float(mean_prob.mean().item()),
            float(mean_prob.max().item()),
            float(sq_factor.min().item()),
            float(sq_factor.mean().item()),
            float(sq_factor.max().item()),
        )

    experts_payload: Dict[str, dict] = {}
    for collector in collectors:
        stats = collector.to_stats()
        moe_key, expert_idx_text = collector.key.rsplit("/", 1)
        expert_idx = int(expert_idx_text)
        routing_entry = router_stats.get(moe_key)
        routing_payload = {}
        if routing_entry is not None:
            routing_payload = {
                "routing_softmax_sum": float(routing_entry["prob_sum"][expert_idx].item()),
                "routing_softmax_sq_sum": float(
                    routing_entry["prob_sq_sum"][expert_idx].item()
                ),
                "routing_total_tokens": int(routing_entry["total_tokens"]),
            }
        experts_payload[collector.key] = {
            "samples": stats.samples,
            "seen": stats.seen,
            "hidden_dim": stats.hidden_dim,
            "input_dim": collector.input_dim,
            **routing_payload,
        }

    logger.info("activation calibration done: experts=%d", len(experts_payload))
    return {
        "metadata": {
            **_metadata_fields_from_sources(sources),
            "num_samples": num_samples,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "padding": padding,
            "max_expert_samples": max_expert_samples,
            "seed": seed,
            "streaming": streaming,
            "shuffle_buffer": shuffle_buffer_value if streaming else shuffle_buffer,
            "routing_weight_stats": ROUTING_WEIGHT_STATS_VERSION,
            "packed": bool(packed),
            "exclude_domains": sorted(exclude_domains) if exclude_domains else [],
        },
        "experts": experts_payload,
    }
