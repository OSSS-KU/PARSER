from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch import nn
from transformers import PreTrainedTokenizerBase


@dataclass
class PerplexityResult:
    dataset: str
    split: str
    perplexity: float
    loss: float
    tokens: int


def compute_perplexity(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 1024,
    stride: int = 512,
    device: str = "cuda",
) -> PerplexityResult:
    model.eval()
    data = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(data["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        max_len = int(model.config.max_position_embeddings)
        max_length = min(max_length, max_len)

    stride = min(stride, max_length)
    nlls = []
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, input_ids.size(1), stride):
            end = min(start + stride, input_ids.size(1))
            begin = max(end - max_length, 0)
            trg_len = end - start
            if trg_len <= 0:
                break
            input_slice = input_ids[:, begin:end]
            target_ids = input_slice.clone()
            target_ids[:, :-trg_len] = -100
            outputs = model(input_slice, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            total_tokens += trg_len
            if end == input_ids.size(1):
                break

    total_nll = torch.stack(nlls).sum()
    loss = (total_nll / total_tokens).item()
    ppl = math.exp(loss)

    return PerplexityResult(
        dataset=f"{dataset_name}/{dataset_config}",
        split=split,
        perplexity=ppl,
        loss=loss,
        tokens=total_tokens,
    )
