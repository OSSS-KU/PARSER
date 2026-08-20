"""Benchmark decode TPS for PARSER-compressed (or original) models.

Pipeline:
  1) Load model via the same loading sequence used in evaluate_blocks.py.
     - Loads pretrained checkpoint via AutoModelForCausalLM.
     - Patches parser_forward.
     - Replaces selected blocks with compressed (or original) blocks from disk.
     - Re-dispatches under device_map if applicable.
  2) Build prompts from WikiText-2 test split using the same chunking pipeline
     as our calibration set (concat -> tokenize -> chunk to seq_len tokens).
  3) For each chunk: warmup, then time decode of `max_new_tokens` tokens with
     KV cache enabled. Prefill is excluded from timing.
  4) Aggregate decode TPS across chunks x repeats; write JSON to output dir.

Usage (sweep): see run_tps_sweep_*.sh templates.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_blocks import (
    _attach_local_device_align_hook,
    _infer_module_device,
    _load_block,
    _log_current_vram,
    _redispatch_model,
    parse_layer_spec,
    resolve_device_map,
    resolve_hf_torch_dtype,
    setup_logging,
)
from moe_pipeline.eval.vram import reset_vram_stats
from moe_pipeline.methods.parser import ensure_parser_forward
from moe_pipeline.model_adapter import HfModelAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode-TPS benchmark for compressed MoE.")
    p.add_argument("--model", required=True)
    p.add_argument("--blocks-dir", type=Path, required=True)
    p.add_argument("--compressed-dir", type=Path, default=None,
                   help="Optional compressed blocks dir; if omitted, uses original blocks.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write tps_results.json (created if missing).")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="HF attention backend to pass to from_pretrained.",
    )
    p.add_argument("--layers", default="all",
                   help="Same syntax as evaluate_blocks.py (all | last:k | start:end | comma list).")
    p.add_argument("--prompt-source", default="wikitext",
                   choices=["wikitext"],
                   help="Prompt corpus. Currently only wikitext-2 supported.")
    p.add_argument("--prompt-length", type=int, default=2048,
                   help="Tokens per prompt chunk (matches calibration seq_len).")
    p.add_argument("--num-chunks", type=int, default=64,
                   help="Number of prompt chunks to measure over.")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Decode budget per measured run.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup runs per chunk (untimed).")
    p.add_argument("--repeats", type=int, default=5,
                   help="Measured runs per chunk after warmup.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_wikitext_prompts(
    tokenizer,
    *,
    seq_len: int,
    num_chunks: int,
    seed: int,
    log: logging.Logger,
) -> torch.Tensor:
    """Construct prompts using the same chunking pipeline as our calibration set.

    1. Load wikitext-2-raw-v1 test split.
    2. Tokenize and concatenate non-empty lines into a single token stream.
    3. Split into seq_len-token chunks.
    4. Take a deterministic random subset of size num_chunks.

    Returns: long tensor of shape [num_chunks, seq_len].
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_stream = encoded["input_ids"][0]
    n_full = token_stream.numel() // seq_len
    if n_full == 0:
        raise RuntimeError(
            f"wikitext-2 test split tokenized to {token_stream.numel()} tokens, "
            f"less than seq_len={seq_len}; cannot form a single chunk."
        )
    chunks = token_stream[: n_full * seq_len].view(n_full, seq_len)
    log.info("built %d full chunks of length %d from wikitext-2 test", n_full, seq_len)

    if num_chunks < n_full:
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(n_full, generator=gen)[:num_chunks]
        chunks = chunks[idx]
    elif num_chunks > n_full:
        log.warning(
            "requested num_chunks=%d > available %d; using all %d.",
            num_chunks, n_full, n_full,
        )
    return chunks.long()


@torch.no_grad()
def measure_decode_tps(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    """Decode-only TPS measurement. Prefill is performed (untimed) before each
    measured run to set up a fresh KV cache; only the decode loop is timed.
    """
    device = _infer_module_device(model, "cuda:0" if torch.cuda.is_available() else "cpu")
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    def _one_run(decode_tokens: int) -> float:
        attn = torch.ones_like(prompt_ids)
        out = model(prompt_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(decode_tokens):
            attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
            out = model(
                tok,
                past_key_values=past,
                attention_mask=attn,
                use_cache=True,
            )
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - t0

    short = max(8, max_new_tokens // 8)
    for _ in range(warmup):
        _one_run(short)

    decode_times: List[float] = []
    for _ in range(repeats):
        decode_times.append(_one_run(max_new_tokens))

    tps_per_run = [max_new_tokens / t for t in decode_times]
    mean = sum(tps_per_run) / len(tps_per_run)
    var = sum((x - mean) ** 2 for x in tps_per_run) / max(1, len(tps_per_run) - 1)
    return {
        "decode_tps_mean": mean,
        "decode_tps_std": var ** 0.5,
        "decode_tps_runs": tps_per_run,
        "decode_times_sec": decode_times,
    }


def load_model_and_apply_blocks(args: argparse.Namespace, log: logging.Logger):
    """Load pretrained model, patch parser_forward, and apply compressed/original
    blocks to the selected layer indices. Mirrors the loading flow used in
    evaluate_blocks.main().
    """
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
    else:
        input_device = str(_infer_module_device(model, "cuda:0" if torch.cuda.is_available() else "cpu"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_current_vram(log, "model load")

    patched = ensure_parser_forward(model)
    if patched:
        log.info("registered parser_forward for %d MoE expert classes", patched)

    adapter = HfModelAdapter(model)
    blocks_dir = args.blocks_dir / "original"
    compressed_dir: Optional[Path] = args.compressed_dir

    total_blocks = adapter.block_count()
    layer_indices = parse_layer_spec(args.layers, total_blocks)
    log.info("loading blocks total=%d selected=%d", total_blocks, len(layer_indices))

    for pos, idx in enumerate(layer_indices, start=1):
        compressed_path = compressed_dir / f"layer_{idx}.pt" if compressed_dir else None
        if compressed_path and compressed_path.exists():
            block = _load_block(compressed_path, "cpu")
            source = "compressed"
        else:
            original_path = blocks_dir / f"layer_{idx}.pt"
            if not original_path.exists():
                log.warning("missing block layer_%d.pt, skipping", idx)
                continue
            block = _load_block(original_path, "cpu")
            source = "original"

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
        log.info("set block %d/%d (layer_%d.pt, %s)", pos, len(layer_indices), idx, source)

    if device_map is not None:
        _redispatch_model(model, log)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_vram_stats()
    _log_current_vram(log, "post block replace")

    return model, tokenizer


def main() -> None:
    setup_logging()
    log = logging.getLogger("benchmark_tps")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "start model=%s blocks_dir=%s compressed_dir=%s output_dir=%s",
        args.model, args.blocks_dir, args.compressed_dir, output_dir,
    )

    model, tokenizer = load_model_and_apply_blocks(args, log)
    model.eval()

    prompts = build_wikitext_prompts(
        tokenizer,
        seq_len=args.prompt_length,
        num_chunks=args.num_chunks,
        seed=args.seed,
        log=log,
    )
    log.info("measuring decode TPS on %d prompts (length=%d, max_new_tokens=%d)",
             prompts.shape[0], prompts.shape[1], args.max_new_tokens)

    per_chunk: List[dict] = []
    for i in range(prompts.shape[0]):
        res = measure_decode_tps(
            model,
            prompts[i],
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        per_chunk.append(res)
        log.info(
            "chunk %3d/%d: decode_tps mean=%.2f std=%.2f",
            i + 1, prompts.shape[0],
            res["decode_tps_mean"], res["decode_tps_std"],
        )

    chunk_means = [r["decode_tps_mean"] for r in per_chunk]
    overall_mean = sum(chunk_means) / len(chunk_means)
    overall_var = (
        sum((x - overall_mean) ** 2 for x in chunk_means)
        / max(1, len(chunk_means) - 1)
    )
    overall_std = overall_var ** 0.5

    payload = {
        "config": {
            "model": args.model,
            "blocks_dir": str(args.blocks_dir),
            "compressed_dir": str(args.compressed_dir) if args.compressed_dir else None,
            "output_dir": str(output_dir),
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "layers": args.layers,
            "prompt_source": args.prompt_source,
            "prompt_length": args.prompt_length,
            "num_chunks": int(prompts.shape[0]),
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "summary": {
            "decode_tps_mean": overall_mean,
            "decode_tps_std": overall_std,
            "num_chunks_measured": len(chunk_means),
        },
        "per_chunk": per_chunk,
    }
    out_path = output_dir / "tps_results.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    log.info("wrote %s (decode_tps mean=%.2f std=%.2f)",
             out_path, overall_mean, overall_std)


if __name__ == "__main__":
    main()
