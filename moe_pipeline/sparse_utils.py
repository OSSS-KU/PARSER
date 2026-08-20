from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def is_sparse_tensor(tensor: Optional[torch.Tensor]) -> bool:
    return torch.is_tensor(tensor) and tensor.layout != torch.strided


def dense_to_sparse_csr(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor for sparse conversion, got shape={tuple(tensor.shape)}")
    if tensor.layout == torch.sparse_csr:
        return tensor
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce().to_sparse_csr()
    if tensor.layout != torch.strided:
        return tensor.to_sparse_coo().coalesce().to_sparse_csr()
    return tensor.detach().to_sparse_csr()


def dense_to_sparse_coo(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor for sparse conversion, got shape={tuple(tensor.shape)}")
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce()
    if tensor.layout != torch.strided:
        return tensor.to_sparse_coo().coalesce()
    return tensor.detach().to_sparse_coo().coalesce()


def to_dense_if_sparse(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.layout == torch.strided:
        return tensor
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce().to_dense()
    return tensor.to_dense()


def linear_with_optional_sparse(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.layout == torch.strided:
        return F.linear(input, weight)

    if weight.dim() != 2:
        raise ValueError(f"Sparse linear expects rank-2 weight, got shape={tuple(weight.shape)}")

    sparse_weight = dense_to_sparse_csr(weight)
    input_2d = input.reshape(-1, input.shape[-1])
    input_mat = input_2d
    if input_mat.device != sparse_weight.device or input_mat.dtype != sparse_weight.dtype:
        input_mat = input_mat.to(device=sparse_weight.device, dtype=sparse_weight.dtype)

    output_dtype = input.dtype

    mm_weight = sparse_weight
    mm_input = input_mat
    if sparse_weight.device.type == "cuda" and sparse_weight.dtype in {torch.float16, torch.bfloat16}:
        mm_weight = sparse_weight.to(dtype=torch.float16)
        mm_input = input_mat.to(dtype=torch.float16)

    output_2d = torch.sparse.mm(mm_weight, mm_input.transpose(0, 1)).transpose(0, 1)

    if output_2d.dtype != output_dtype or output_2d.device != input.device:
        output_2d = output_2d.to(device=input.device, dtype=output_dtype)
    return output_2d.reshape(*input.shape[:-1], sparse_weight.shape[0])


def count_nonzero_tensor(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return int(torch.count_nonzero(tensor).item())
    if tensor.layout == torch.sparse_coo:
        return int(tensor.coalesce()._nnz())
    if tensor.layout == torch.sparse_csr:
        return int(tensor.col_indices().numel())
    return int(dense_to_sparse_csr(tensor).col_indices().numel())
