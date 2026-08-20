from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def slugify_model_id(model_id: str) -> str:
    if not model_id:
        return "unknown-model"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip())
    return slug.strip("_") or "unknown-model"


def default_blocks_dir(model_id: str, *, root: str = "artifacts") -> Path:
    return Path(root) / slugify_model_id(model_id) / "blocks"


def compression_tag(
    *,
    method: str,
    sparsity: float,
    residual_mode: str,
    residual_scope: str,
    backbone: str,
    importance: str = "formula",
    parser_routing_score: Optional[bool] = None,
) -> str:
    method_norm = (method or "").strip().lower()
    importance_norm = (importance or "formula").strip().lower()
    s_val = int(round(float(sparsity) * 100))
    if method_norm == "original":
        return "original"
    if method_norm.startswith("parser-"):
        if importance_norm not in {"", "formula", "wanda"}:
            raise ValueError(
                f"Unsupported parser importance '{importance_norm}'. Use formula or wanda."
            )
        if importance_norm == "wanda":
            routing = ""
            if parser_routing_score is not None:
                routing = "-rsoftmax" if parser_routing_score else "-rnone"
            return f"{method}-i{importance_norm}-s{s_val:03d}-{backbone}{routing}"
        if parser_routing_score is not None:
            routing = "rsoftmax" if parser_routing_score else "rnone"
            return f"{method}-s{s_val:03d}-{backbone}-{routing}"
        return f"{method}-s{s_val:03d}-{backbone}"
    return f"{method}-{residual_mode}-{residual_scope}-s{s_val:03d}-{backbone}"


def default_compressed_dir(
    blocks_dir: Path,
    *,
    method: str,
    sparsity: float,
    residual_mode: str,
    residual_scope: str,
    backbone: str,
    importance: str = "formula",
    parser_routing_score: Optional[bool] = None,
) -> Path:
    tag = compression_tag(
        method=method,
        sparsity=sparsity,
        residual_mode=residual_mode,
        residual_scope=residual_scope,
        backbone=backbone,
        importance=importance,
        parser_routing_score=parser_routing_score,
    )
    return blocks_dir.parent / tag / "compressed"


def default_eval_dir(
    *,
    compressed_dir: Optional[Path],
    blocks_dir: Path,
) -> Path:
    if compressed_dir is not None:
        return compressed_dir.parent / "eval"
    return blocks_dir.parent / "eval-original"
