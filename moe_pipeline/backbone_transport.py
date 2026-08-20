from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from moe_pipeline.phase_timing import global_timer


def _ensure_permutation_indices(indices: torch.Tensor, size: int) -> torch.Tensor:
    if indices.numel() != size:
        raise ValueError(f"Permutation size mismatch: {indices.numel()} vs {size}")
    perm = indices.clone().to(dtype=torch.long)
    used = set()
    missing = [i for i in range(size) if i not in perm.tolist()]
    for i in range(size):
        val = int(perm[i].item())
        if val in used:
            perm[i] = missing.pop(0)
        else:
            used.add(val)
    return perm


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pt"


def _make_cache_key(
    *,
    backbone_method: str,
    layer_idx: int,
    hidden_dim: int,
    gate_in: int,
    up_in: int,
    down_out: int,
    num_experts: int,
) -> str:
    return (
        f"{backbone_method}_L{layer_idx}_H{hidden_dim}_G{gate_in}_U{up_in}_D{down_out}_E{num_experts}"
    )


def _load_cache(
    cache_dir: Path,
    key: str,
    *,
    hidden_dim: int,
    combined_width: int,
    num_experts: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu")
    backbone = payload.get("backbone")
    transport = payload.get("transport")
    if backbone is None or transport is None:
        return None, None
    if backbone.shape != (hidden_dim, combined_width):
        return None, None
    if transport.shape != (num_experts, hidden_dim):
        return None, None
    return backbone, transport


def _save_cache(
    cache_dir: Path,
    key: str,
    *,
    backbone: torch.Tensor,
    transport: torch.Tensor,
    num_experts: int,
    hidden_dim: int,
    combined_width: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone": backbone.to("cpu"),
        "transport": transport.to("cpu"),
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "combined_width": combined_width,
    }
    torch.save(payload, _cache_path(cache_dir, key))


def _compute_barycenter_and_transport(
    combined_matrices: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        import ot  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("POT (ot) is required for backbone=barycenter") from exc

    if not combined_matrices:
        raise ValueError("combined_matrices must be non-empty")

    matrices = torch.stack([w.detach().float().cpu() for w in combined_matrices], dim=0)
    num_experts, hidden_dim, width = matrices.shape

    weights_uniform = torch.stack(
        [
            torch.full((hidden_dim,), 1.0 / hidden_dim, dtype=torch.float32)
            for _ in range(num_experts)
        ],
        dim=0,
    )

    barycenter = ot.lp.free_support_barycenter(
        matrices,
        weights_uniform,
        matrices[0],
        numItermax=100,
        stopThr=1e-9,
        numThreads="max",
    )

    barycenter = torch.tensor(barycenter, dtype=torch.float32)
    transport = torch.zeros((num_experts, hidden_dim), dtype=torch.long)

    for i in range(num_experts):
        cost = ot.utils.dist(barycenter, matrices[i])
        T = ot.lp.emd(weights_uniform[i], weights_uniform[i], cost, numThreads="max")
        perm = T.argmax(1)
        perm = _ensure_permutation_indices(perm.to(dtype=torch.long), hidden_dim)
        transport[i] = perm

    return barycenter, transport


def compute_backbone_and_transport(
    *,
    gates: Sequence[torch.Tensor],
    ups: Sequence[torch.Tensor],
    downs: Sequence[torch.Tensor],
    backbone_method: str,
    layer_idx: int,
    cache_dir: Optional[Path] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not gates:
        raise ValueError("gates must be non-empty")
    num_experts = len(gates)
    device = gates[0].device
    dtype = gates[0].dtype

    hidden_dim = gates[0].shape[0]
    gate_in = gates[0].shape[1]
    up_in = ups[0].shape[1]
    down_out = downs[0].shape[0]
    combined_width = gate_in + up_in + down_out

    combined_matrices = [
        torch.cat([g, u, d.transpose(0, 1)], dim=1)
        for g, u, d in zip(gates, ups, downs)
    ]

    cache_key = _make_cache_key(
        backbone_method=backbone_method,
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        gate_in=gate_in,
        up_in=up_in,
        down_out=down_out,
        num_experts=num_experts,
    )

    timer = global_timer()
    with timer.timed("backbone"):
        cached_backbone = cached_transport = None
        if cache_dir is not None:
            cached_backbone, cached_transport = _load_cache(
                cache_dir,
                cache_key,
                hidden_dim=hidden_dim,
                combined_width=combined_width,
                num_experts=num_experts,
            )

        if cached_backbone is not None and cached_transport is not None:
            backbone_combined = cached_backbone
            transport = cached_transport
            timer.add("backbone_cache_hit", 0.0)
        else:
            timer.add("backbone_cache_miss", 0.0)
            if backbone_method == "mean":
                backbone_combined = torch.stack(combined_matrices, dim=0).mean(dim=0)
                transport = torch.stack(
                    [torch.arange(hidden_dim, dtype=torch.long) for _ in range(num_experts)],
                    dim=0,
                )
            elif backbone_method == "barycenter":
                backbone_combined, transport = _compute_barycenter_and_transport(combined_matrices)
            else:
                raise ValueError(f"Unsupported backbone method: {backbone_method}")

            if cache_dir is not None:
                _save_cache(
                    cache_dir,
                    cache_key,
                    backbone=backbone_combined,
                    transport=transport,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    combined_width=combined_width,
                )

    backbone_gate = backbone_combined[:, :gate_in].to(device=device, dtype=dtype)
    backbone_up = backbone_combined[:, gate_in : gate_in + up_in].to(device=device, dtype=dtype)
    backbone_down = backbone_combined[:, gate_in + up_in :].transpose(0, 1).to(
        device=device, dtype=dtype
    )

    return {
        "gate": backbone_gate,
        "up": backbone_up,
        "down": backbone_down,
    }, transport.to(device=device)


def permute_rows(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return weight.index_select(0, perm.to(device=weight.device))


def permute_columns(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return weight.index_select(1, perm.to(device=weight.device))
