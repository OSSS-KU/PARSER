from __future__ import annotations

import math
import os
import types
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from moe_pipeline.backbone_transport import (
    compute_backbone_and_transport,
    permute_columns,
    permute_rows,
)
from moe_pipeline.moe_introspection import (
    expert_proj_attr_map,
    find_moe_blocks,
    get_experts,
    is_supported_moe_block,
)
from moe_pipeline.methods.base import BaseBlockCompressor, CompressionSpec
from moe_pipeline.sparse_utils import (
    dense_to_sparse_csr,
    linear_with_optional_sparse,
    to_dense_if_sparse,
)

logger = logging.getLogger("moe_pipeline.methods.parser")


def make_expert_key(layer_idx: int, moe_name: str, expert_idx: int) -> str:
    return f"{layer_idx}/{moe_name}/{expert_idx}"


def resolve_activation_fn(expert: nn.Module):
    for attr in ("act_fn", "activation_fn", "activation", "act"):
        fn = getattr(expert, attr, None)
        if callable(fn):
            return fn
    act_name = None
    if hasattr(expert, "config") and getattr(expert.config, "hidden_act", None):
        act_name = expert.config.hidden_act
    if isinstance(act_name, str):
        act_name = act_name.lower()
        if act_name in {"silu", "swish"}:
            return F.silu
        if act_name == "gelu":
            return F.gelu
        if act_name == "relu":
            return F.relu
    return F.silu


def combine_gate_up(act_fn, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    custom = getattr(act_fn, "combine_gate_up", None)
    if callable(custom):
        return custom(gate, up)
    return act_fn(gate) * up


def parser_forward(
    self,
    hidden_states: torch.Tensor,
    gate_backbone: Optional[torch.Tensor] = None,
    up_backbone: Optional[torch.Tensor] = None,
    *,
    backbone_has_bias: bool = False,
    **kwargs,
) -> torch.Tensor:
    attr_map = getattr(self, "_parser_attr_map", None)
    if attr_map is None:
        attr_map = expert_proj_attr_map(self)
        setattr(self, "_parser_attr_map", attr_map)

    gate_module = getattr(self, attr_map["gate"])
    up_module = getattr(self, attr_map["up"])
    down_module = getattr(self, attr_map["down"])

    if isinstance(gate_module, ParserRowProjection):
        backbone_gate, residual_gate = gate_module(
            hidden_states,
            backbone_override=gate_backbone,
            backbone_has_bias=backbone_has_bias,
        )
    else:
        backbone_gate = gate_module(hidden_states)
        residual_gate = None

    if isinstance(up_module, ParserRowProjection):
        backbone_up, residual_up = up_module(
            hidden_states,
            backbone_override=up_backbone,
            backbone_has_bias=backbone_has_bias,
        )
    else:
        backbone_up = up_module(hidden_states)
        residual_up = None

    act_fn = getattr(self, "_parser_act_fn", None)
    if act_fn is None:
        act_fn = resolve_activation_fn(self)
        setattr(self, "_parser_act_fn", act_fn)

    indices = getattr(gate_module, "residual_indices", None)
    has_residual = (
        indices is not None
        and indices.numel() > 0
        and residual_gate is not None
        and residual_up is not None
    )

    if has_residual:
        backbone_gate.index_add_(1, indices, residual_gate)
        backbone_up.index_add_(1, indices, residual_up)
        combined_hidden = combine_gate_up(act_fn, backbone_gate, backbone_up)
        residual_hidden = combined_hidden.index_select(1, indices)
        backbone_hidden = combined_hidden
    else:
        backbone_hidden = combine_gate_up(act_fn, backbone_gate, backbone_up)
        residual_hidden = None

    if isinstance(down_module, ParserColumnProjection):
        return down_module(backbone_hidden, residual_hidden)
    return down_module(backbone_hidden)


def ensure_parser_forward(model: nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if not is_supported_moe_block(module):
            continue
        ensure_parser_moe_forward(module)
        for expert in get_experts(module):
            if not hasattr(expert.__class__, "parser_forward"):
                setattr(expert.__class__, "parser_forward", parser_forward)
                patched += 1
    return patched


def _call_original_moe_forward(self: nn.Module, hidden_states: torch.Tensor, *args, **kwargs):
    original = getattr(self.__class__, "_parser_original_moe_forward", None)
    if original is None:
        raise RuntimeError(
            f"{self.__class__.__name__} has no original MoE forward to fall back to."
        )
    return original(self, hidden_states, *args, **kwargs)


def _infer_moe_top_k(moe_block: nn.Module) -> int:
    for attr in ("top_k", "num_experts_per_tok", "num_experts_per_token", "topk"):
        value = getattr(moe_block, attr, None)
        if value is None:
            continue
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            continue
        if top_k > 0:
            return top_k
    config = getattr(moe_block, "config", None)
    if config is not None:
        for attr in ("num_experts_per_tok", "num_experts_per_token", "top_k", "topk"):
            value = getattr(config, attr, None)
            if value is None:
                continue
            try:
                top_k = int(value)
            except (TypeError, ValueError):
                continue
            if top_k > 0:
                return top_k
    return 2


def _infer_moe_num_experts(moe_block: nn.Module) -> int:
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(moe_block, attr, None)
        if value is None:
            continue
        try:
            num_experts = int(value)
        except (TypeError, ValueError):
            continue
        if num_experts > 0:
            return num_experts
    experts = getattr(moe_block, "experts", None)
    if isinstance(experts, (list, tuple, nn.ModuleList)):
        return len(experts)
    raise RuntimeError(f"Unable to infer num_experts for {moe_block.__class__.__name__}.")


def _indexed_experts(moe_block: nn.Module) -> List[Tuple[int, nn.Module]]:
    experts = getattr(moe_block, "experts", None)
    if not isinstance(experts, (list, tuple, nn.ModuleList)) or not experts:
        raise RuntimeError(
            f"Unsupported MoE block structure for '{moe_block.__class__.__name__}': "
            "expected an '.experts' attribute."
        )
    indexed: List[Tuple[int, nn.Module]] = []
    for idx, expert in enumerate(experts):
        if isinstance(expert, nn.Module):
            indexed.append((idx, expert))
    return indexed


def _parser_moe_layout(
    moe_block: nn.Module,
) -> Optional[Tuple[List[Tuple[int, nn.Module]], Dict[str, str], SharedBackbone]]:
    cached = getattr(moe_block, "_parser_moe_layout_cache", None)
    if cached is False:
        return None
    if cached is not None:
        return cached

    try:
        indexed = _indexed_experts(moe_block)
    except RuntimeError:
        setattr(moe_block, "_parser_moe_layout_cache", False)
        return None
    if not indexed:
        setattr(moe_block, "_parser_moe_layout_cache", False)
        return None

    experts = getattr(moe_block, "experts", None)
    if isinstance(experts, (list, tuple, nn.ModuleList)) and len(indexed) != len(experts):
        setattr(moe_block, "_parser_moe_layout_cache", False)
        return None

    first_expert = indexed[0][1]
    try:
        attr_map = getattr(first_expert, "_parser_attr_map", None) or expert_proj_attr_map(first_expert)
    except ValueError:
        setattr(moe_block, "_parser_moe_layout_cache", False)
        return None

    shared_backbone = None
    for _, expert in indexed:
        try:
            gate_module = getattr(expert, attr_map["gate"])
            up_module = getattr(expert, attr_map["up"])
            down_module = getattr(expert, attr_map["down"])
        except AttributeError:
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None
        if not (
            isinstance(gate_module, ParserRowProjection)
            and isinstance(up_module, ParserRowProjection)
            and isinstance(down_module, ParserColumnProjection)
        ):
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None
        expert_shared = getattr(gate_module, "_shared_backbone", None)
        if not isinstance(expert_shared, SharedBackbone):
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None
        if shared_backbone is None:
            shared_backbone = expert_shared
        elif shared_backbone is not expert_shared:
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None
        if getattr(up_module, "_shared_backbone", None) is not shared_backbone:
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None
        if getattr(down_module, "_shared_backbone", None) is not shared_backbone:
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None

        indices = getattr(gate_module, "residual_indices", None)
        up_indices = getattr(up_module, "residual_indices", None)
        down_indices = getattr(down_module, "residual_indices", None)
        if (
            torch.is_tensor(indices)
            and torch.is_tensor(up_indices)
            and torch.is_tensor(down_indices)
            and (
                indices.numel() != up_indices.numel()
                or indices.numel() != down_indices.numel()
                or not torch.equal(indices, up_indices)
                or not torch.equal(indices, down_indices)
            )
        ):
            setattr(moe_block, "_parser_moe_layout_cache", False)
            return None

    if shared_backbone is None:
        setattr(moe_block, "_parser_moe_layout_cache", False)
        return None
    layout = (indexed, attr_map, shared_backbone)
    setattr(moe_block, "_parser_moe_layout_cache", layout)
    return layout


def _parse_tuple_routing(gate_out: object) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if not isinstance(gate_out, (tuple, list)):
        return None

    selected_experts = None
    floating_candidates: List[torch.Tensor] = []
    aux_loss = None
    for item in gate_out:
        if not torch.is_tensor(item):
            continue
        if item.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8} and item.ndim >= 1:
            selected_experts = item
        elif item.is_floating_point():
            if item.ndim >= 1:
                floating_candidates.append(item)
            elif item.numel() == 1:
                aux_loss = item

    if selected_experts is None:
        return None
    routing_weights = None
    for candidate in floating_candidates:
        if candidate.shape == selected_experts.shape:
            routing_weights = candidate
            break
    if routing_weights is None:
        for candidate in floating_candidates:
            if candidate.ndim == 2 and selected_experts.ndim >= 1 and selected_experts.numel() > 0:
                selected_2d = selected_experts
                if selected_2d.ndim == 1:
                    selected_2d = selected_2d.unsqueeze(-1)
                else:
                    selected_2d = selected_2d.reshape(-1, selected_2d.shape[-1])
                if candidate.shape[0] == selected_2d.shape[0] and candidate.shape[1] > int(selected_2d.max().item()):
                    routing_weights = candidate.gather(1, selected_2d.to(device=candidate.device, dtype=torch.long))
                    break
    if routing_weights is None:
        return None
    if selected_experts.shape != routing_weights.shape:
        return None
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
        routing_weights = routing_weights.unsqueeze(-1)
    else:
        selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
        routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])
    return selected_experts.to(dtype=torch.long), routing_weights, aux_loss


def _compute_moe_routing(
    moe_block: nn.Module,
    hidden_states: torch.Tensor,
    hidden_states_2d: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    gate = getattr(moe_block, "gate", None)
    if gate is None:
        raise RuntimeError(f"{moe_block.__class__.__name__} has no gate module.")

    router_logits: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    if isinstance(gate, nn.Linear):
        router_logits = gate(hidden_states_2d)
        top_k = _infer_moe_top_k(moe_block)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        normalize = bool(getattr(moe_block, "norm_topk_prob", True))
        if normalize:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_2d.dtype)
        return router_logits, selected_experts.to(dtype=torch.long), routing_weights, aux_loss

    if bool(getattr(gate, "_parser_topk_softmax_logits", False)):
        weight = getattr(gate, "weight", None)
        bias = getattr(gate, "bias", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            router_logits = F.linear(
                hidden_states_2d.to(device=weight.device, dtype=weight.dtype),
                weight,
                bias if torch.is_tensor(bias) else None,
            )
            top_k = _infer_moe_top_k(moe_block)
            top_k = max(1, min(int(top_k), int(router_logits.shape[-1])))
            router_top_value, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            routing_weights = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
            routing_weights = routing_weights.to(device=hidden_states_2d.device, dtype=hidden_states_2d.dtype)
            selected_experts = selected_experts.to(device=hidden_states_2d.device, dtype=torch.long)
            return router_logits, selected_experts, routing_weights, aux_loss

    gate_out = gate(hidden_states)
    parsed = _parse_tuple_routing(gate_out)
    if parsed is None:
        raise RuntimeError(
            f"Unsupported parser MoE gate output for {moe_block.__class__.__name__}: "
            f"{type(gate_out)!r}."
        )
    selected_experts, routing_weights, aux_loss = parsed
    routing_weights = routing_weights.to(device=hidden_states_2d.device, dtype=hidden_states_2d.dtype)
    selected_experts = selected_experts.to(device=hidden_states_2d.device, dtype=torch.long)
    return router_logits, selected_experts, routing_weights, aux_loss


def _parser_expert_hidden_and_residual_output(
    expert: nn.Module,
    hidden_states: torch.Tensor,
    gate_backbone: torch.Tensor,
    up_backbone: torch.Tensor,
    attr_map: Dict[str, str],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    gate_module = getattr(expert, attr_map["gate"])
    up_module = getattr(expert, attr_map["up"])
    down_module = getattr(expert, attr_map["down"])

    if not (
        isinstance(gate_module, ParserRowProjection)
        and isinstance(up_module, ParserRowProjection)
        and isinstance(down_module, ParserColumnProjection)
    ):
        raise RuntimeError("Expert is not backed by Parser projection wrappers.")

    backbone_gate, residual_gate = gate_module(
        hidden_states,
        backbone_override=gate_backbone,
        backbone_has_bias=False,
    )
    backbone_up, residual_up = up_module(
        hidden_states,
        backbone_override=up_backbone,
        backbone_has_bias=False,
    )

    act_fn = getattr(expert, "_parser_act_fn", None)
    if act_fn is None:
        act_fn = resolve_activation_fn(expert)
        setattr(expert, "_parser_act_fn", act_fn)

    indices = getattr(gate_module, "residual_indices", None)
    has_residual = (
        indices is not None
        and indices.numel() > 0
        and residual_gate is not None
        and residual_up is not None
    )
    if has_residual:
        backbone_gate.index_add_(1, indices, residual_gate)
        backbone_up.index_add_(1, indices, residual_up)

    hidden = combine_gate_up(act_fn, backbone_gate, backbone_up)

    residual_output: Optional[torch.Tensor] = None
    residual_weight = getattr(down_module, "residual_weight", None)
    if (
        has_residual
        and residual_weight is not None
        and torch.is_tensor(residual_weight)
        and getattr(down_module, "residual_indices", None) is not None
        and down_module.residual_indices.numel() > 0
    ):
        residual_hidden = hidden.index_select(1, down_module.residual_indices)
        residual_output = linear_with_optional_sparse(residual_hidden, residual_weight)

    bias = getattr(down_module, "bias", None)
    if bias is not None:
        bias_value = bias.to(device=hidden.device, dtype=hidden.dtype).view(1, -1)
        if residual_output is None:
            residual_output = bias_value.expand(hidden.shape[0], -1)
        else:
            residual_output = residual_output + bias_value

    return hidden, residual_output


def _shared_expert_output(moe_block: nn.Module, hidden_states_2d: torch.Tensor) -> Optional[torch.Tensor]:
    shared_expert = getattr(moe_block, "shared_expert", None)
    if shared_expert is None:
        shared_expert = getattr(moe_block, "shared_experts", None)
    if not isinstance(shared_expert, nn.Module):
        return None

    output = shared_expert(hidden_states_2d)
    shared_gate = getattr(moe_block, "shared_expert_gate", None)
    if isinstance(shared_gate, nn.Module):
        output = torch.sigmoid(shared_gate(hidden_states_2d)) * output
    return output


def parser_moe_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
    if args or kwargs or self.training:
        return _call_original_moe_forward(self, hidden_states, *args, **kwargs)

    layout = _parser_moe_layout(self)
    if layout is None:
        return _call_original_moe_forward(self, hidden_states, *args, **kwargs)

    indexed_experts, attr_map, shared_backbone = layout
    num_experts = _infer_moe_num_experts(self)
    if len(indexed_experts) != num_experts:
        return _call_original_moe_forward(self, hidden_states, *args, **kwargs)
    experts_by_index = {idx: expert for idx, expert in indexed_experts}

    orig_shape = hidden_states.shape
    if hidden_states.ndim != 3:
        return _call_original_moe_forward(self, hidden_states, *args, **kwargs)
    batch_size, sequence_length, hidden_dim = orig_shape
    hidden_states_2d = hidden_states.reshape(-1, hidden_dim)
    token_count = hidden_states_2d.shape[0]

    try:
        router_logits, selected_experts, routing_weights, aux_loss = _compute_moe_routing(
            self,
            hidden_states,
            hidden_states_2d,
        )
    except RuntimeError:
        return _call_original_moe_forward(self, hidden_states, *args, **kwargs)

    if selected_experts.numel() == 0:
        final_hidden_states = hidden_states_2d.new_zeros((token_count, hidden_dim))
    else:
        selected_experts = selected_experts.reshape(token_count, -1)
        routing_weights = routing_weights.reshape(token_count, -1)

        gate_backbone_all = F.linear(hidden_states_2d, shared_backbone.gate)
        up_backbone_all = F.linear(hidden_states_2d, shared_backbone.up)
        shared_hidden = hidden_states_2d.new_zeros((token_count, shared_backbone.down.shape[1]))
        residual_output = hidden_states_2d.new_zeros((token_count, shared_backbone.down.shape[0]))

        expert_hit = torch.unique(selected_experts, sorted=True)
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.item())
            expert = experts_by_index.get(expert_idx)
            if expert is None:
                return _call_original_moe_forward(self, hidden_states, *args, **kwargs)

            token_idx, topk_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states_2d.index_select(0, token_idx)
            gate_backbone = gate_backbone_all.index_select(0, token_idx)
            up_backbone = up_backbone_all.index_select(0, token_idx)

            expert_hidden, expert_residual_output = _parser_expert_hidden_and_residual_output(
                expert,
                current_state,
                gate_backbone,
                up_backbone,
                attr_map,
            )
            weights = routing_weights[token_idx, topk_idx].view(-1, 1)
            weighted_hidden = expert_hidden * weights
            shared_hidden.index_add_(0, token_idx, weighted_hidden.to(shared_hidden.dtype))

            if expert_residual_output is not None:
                weighted_residual = expert_residual_output * weights
                residual_output.index_add_(
                    0,
                    token_idx,
                    weighted_residual.to(residual_output.dtype),
                )

        final_hidden_states = F.linear(shared_hidden, shared_backbone.down) + residual_output

    shared_output = _shared_expert_output(self, hidden_states_2d)
    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    if aux_loss is not None and hasattr(self, "training") and self.training:
        _ = aux_loss
    if router_logits is not None:
        return final_hidden_states, router_logits
    return final_hidden_states


def ensure_parser_moe_forward(moe_block: nn.Module) -> bool:
    cls = moe_block.__class__
    if getattr(cls, "_parser_moe_forward_patched", False):
        return False
    original = getattr(cls, "forward", None)
    if original is None:
        return False
    setattr(cls, "_parser_original_moe_forward", original)
    setattr(cls, "parser_moe_forward", parser_moe_forward)
    setattr(cls, "forward", parser_moe_forward)
    setattr(cls, "_parser_moe_forward_patched", True)
    return True


def register_optional_param(module: nn.Module, name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        module.register_parameter(name, None)
        return
    module.register_parameter(name, nn.Parameter(tensor.detach(), requires_grad=False))


def register_optional_buffer(
    module: nn.Module,
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    persistent: bool = True,
) -> None:
    if tensor is None:
        buffer = torch.tensor([], dtype=dtype or torch.float32)
        module.register_buffer(name, buffer, persistent=persistent)
        return
    buffer = tensor.detach()
    if dtype is not None:
        buffer = buffer.to(dtype=dtype)
    module.register_buffer(name, buffer, persistent=persistent)


class SharedBackbone(nn.Module):
    """Stores a layer-shared backbone for Parser projection wrappers."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.register_parameter("gate", nn.Parameter(gate.detach(), requires_grad=False))
        self.register_parameter("up", nn.Parameter(up.detach(), requires_grad=False))
        self.register_parameter("down", nn.Parameter(down.detach(), requires_grad=False))


class SharedProjectionBase(nn.Module):
    """Common projection bookkeeping for shared-backbone adapters."""

    def __init__(
        self,
        shared_backbone: SharedBackbone,
        projection: str,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_shared_backbone", shared_backbone)
        self.projection = projection

        base_weight = getattr(self._shared_backbone, projection)
        self.in_features = base_weight.shape[1]
        self.out_features = base_weight.shape[0]

        register_optional_param(self, "bias", bias)

    @property
    def backbone_weight(self) -> torch.Tensor:
        return getattr(self._shared_backbone, self.projection)

    def _apply(self, fn):  # type: ignore[override]
        shared_backbone = getattr(self, "_shared_backbone", None)
        if isinstance(shared_backbone, nn.Module):
            shared_backbone._apply(fn)
        return super()._apply(fn)

    def _materialize_residual_matrix(self) -> Optional[torch.Tensor]:
        return None

    @property
    def weight(self) -> torch.Tensor:
        weight = self.backbone_weight.detach()
        residual = self._materialize_residual_matrix()
        if residual is not None:
            weight = weight + residual
        return weight


class ParserRowProjection(SharedProjectionBase):
    """Row-structured projection returning backbone and residual activations separately."""

    def __init__(
        self,
        shared_backbone: SharedBackbone,
        projection: str,
        residual: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        store_sparse: bool = False,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(shared_backbone=shared_backbone, projection=projection, bias=bias)
        if residual is None or residual[0].numel() == 0 or residual[1].numel() == 0:
            if store_sparse:
                register_optional_buffer(self, "residual_weight", None)
            else:
                register_optional_param(self, "residual_weight", None)
            register_optional_buffer(self, "residual_indices", None, dtype=torch.long)
        else:
            indices, weights = residual
            register_optional_buffer(self, "residual_indices", indices, dtype=torch.long)
            if store_sparse:
                register_optional_buffer(self, "residual_weight", dense_to_sparse_csr(weights))
            else:
                register_optional_param(self, "residual_weight", weights)

    def _materialize_residual_matrix(self) -> Optional[torch.Tensor]:
        if self.residual_weight is None or self.residual_indices.numel() == 0:
            return None

        residual_weight = to_dense_if_sparse(self.residual_weight)
        if residual_weight is None:
            return None
        residual_matrix = torch.zeros(
            self.out_features,
            self.in_features,
            device=residual_weight.device,
            dtype=residual_weight.dtype,
        )
        residual_matrix[self.residual_indices] = residual_weight
        return residual_matrix

    def forward(  # type: ignore[override]
        self,
        input: torch.Tensor,
        *,
        backbone_override: Optional[torch.Tensor] = None,
        backbone_has_bias: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if backbone_override is None:
            backbone_out = F.linear(input, self.backbone_weight)
            if self.bias is not None:
                backbone_out = backbone_out + self.bias
        else:
            backbone_out = backbone_override
            if self.bias is not None and not backbone_has_bias:
                backbone_out = backbone_out + self.bias

        residual_out: Optional[torch.Tensor] = None
        if self.residual_weight is not None and self.residual_indices.numel() > 0:
            residual_out = linear_with_optional_sparse(input, self.residual_weight)

        return backbone_out, residual_out


class ParserColumnProjection(SharedProjectionBase):
    """Column-structured projection that consumes split backbone/residual inputs."""

    def __init__(
        self,
        shared_backbone: SharedBackbone,
        projection: str,
        residual: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        store_sparse: bool = False,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(shared_backbone=shared_backbone, projection=projection, bias=bias)
        if residual is None or residual[0].numel() == 0 or residual[1].numel() == 0:
            if store_sparse:
                register_optional_buffer(self, "residual_weight", None)
            else:
                register_optional_param(self, "residual_weight", None)
            register_optional_buffer(self, "residual_indices", None, dtype=torch.long)
        else:
            indices, weights = residual
            register_optional_buffer(self, "residual_indices", indices, dtype=torch.long)
            if store_sparse:
                register_optional_buffer(self, "residual_weight", dense_to_sparse_csr(weights))
            else:
                register_optional_param(self, "residual_weight", weights)

    def _materialize_residual_matrix(self) -> Optional[torch.Tensor]:
        if self.residual_weight is None or self.residual_indices.numel() == 0:
            return None

        residual_weight = to_dense_if_sparse(self.residual_weight)
        if residual_weight is None:
            return None
        residual_matrix = torch.zeros(
            self.out_features,
            self.in_features,
            device=residual_weight.device,
            dtype=residual_weight.dtype,
        )
        residual_matrix[:, self.residual_indices] = residual_weight
        return residual_matrix

    def forward(  # type: ignore[override]
        self,
        backbone_input: torch.Tensor,
        residual_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backbone_output = F.linear(backbone_input, self.backbone_weight)
        residual_output: Optional[torch.Tensor] = None

        if (
            residual_input is not None
            and residual_input.numel() > 0
            and self.residual_weight is not None
            and self.residual_indices.numel() > 0
        ):
            residual_output = linear_with_optional_sparse(residual_input, self.residual_weight)
            output = backbone_output + residual_output
        else:
            output = backbone_output

        if self.bias is not None:
            output = output + self.bias

        return output


@dataclass
class ParserStats:
    pruned_units: int
    total_units: int


@dataclass
class ExpertImportance:
    unit_scores: torch.Tensor
    gate_scale: Optional[torch.Tensor]
    up_scale: Optional[torch.Tensor]
    down_scale: Optional[torch.Tensor]
    source: str


class ParserBlockCompressor(BaseBlockCompressor):
    """Minimal Parser implementation for block-level compression."""

    def __init__(self, spec: CompressionSpec, *, activation_stats: Optional[Dict[str, dict]] = None):
        super().__init__(spec)
        if spec.residual_mode not in {"structured", "unstructured"}:
            raise ValueError(f"Unsupported residual_mode: {spec.residual_mode}")
        if spec.residual_scope not in {"expert", "layer"}:
            raise ValueError(f"Unsupported residual_scope: {spec.residual_scope}")
        self._importance_mode = str(getattr(spec, "importance", "formula") or "formula").strip().lower()
        _VALID_IMPORTANCE = {
            "formula",
            "wanda",
        }
        if self._importance_mode not in _VALID_IMPORTANCE:
            raise ValueError(
                f"Unsupported parser importance '{self._importance_mode}'. "
                "Use formula or wanda."
            )
        if activation_stats is None:
            self._activation_experts = None
        else:
            self._activation_experts = activation_stats.get("experts", activation_stats)

    def compress_block(self, block: nn.Module, layer_idx: int) -> nn.Module:
        moe_blocks = find_moe_blocks(block)
        for moe_name, moe_block in moe_blocks:
            self._compress_moe_block(moe_block, layer_idx, moe_name)
        return block

    def _compress_moe_block(self, moe_block: nn.Module, layer_idx: int, moe_name: str) -> None:
        experts = get_experts(moe_block)
        if not experts:
            return

        attr_map = expert_proj_attr_map(experts[0])
        gate_modules = [getattr(expert, attr_map["gate"]) for expert in experts]
        up_modules = [getattr(expert, attr_map["up"]) for expert in experts]
        down_modules = [getattr(expert, attr_map["down"]) for expert in experts]

        gate_weights = [mod.weight.detach() for mod in gate_modules]
        up_weights = [mod.weight.detach() for mod in up_modules]
        down_weights = [mod.weight.detach() for mod in down_modules]

        backbone, transport = compute_backbone_and_transport(
            gates=gate_weights,
            ups=up_weights,
            downs=down_weights,
            backbone_method=self.spec.backbone,
            layer_idx=layer_idx,
            cache_dir=Path(os.environ.get("BACKBONE_CACHE_DIR", ".cache/backbone")),
        )
        shared_backbone = SharedBackbone(backbone["gate"], backbone["up"], backbone["down"])

        residuals: List[Dict[str, torch.Tensor]] = []
        for expert_idx, (gate_w, up_w, down_w) in enumerate(
            zip(gate_weights, up_weights, down_weights)
        ):
            perm = transport[expert_idx]
            gate_perm = permute_rows(gate_w, perm)
            up_perm = permute_rows(up_w, perm)
            down_perm = permute_columns(down_w, perm)
            residuals.append(
                {
                    "gate": gate_perm - backbone["gate"],
                    "up": up_perm - backbone["up"],
                    "down": down_perm - backbone["down"],
                }
            )
        importance = self._compute_importance_state(
            experts=experts,
            residuals=residuals,
            backbone=shared_backbone,
            layer_idx=layer_idx,
            moe_name=moe_name,
            need_param_scores=(self.spec.residual_mode == "unstructured"),
        )
        if self.spec.residual_mode == "structured":
            unit_scores = [item.unit_scores for item in importance]
            pruned = self._prune_structured(residuals, unit_scores)
        else:
            pruned = self._prune_unstructured(residuals, importance)

        for expert_idx, expert in enumerate(experts):
            gate_module = getattr(expert, attr_map["gate"])
            up_module = getattr(expert, attr_map["up"])
            down_module = getattr(expert, attr_map["down"])

            perm = transport[expert_idx]
            gate_bias = (
                permute_rows(gate_module.bias.detach().unsqueeze(1), perm).squeeze(1)
                if gate_module.bias is not None
                else None
            )
            up_bias = (
                permute_rows(up_module.bias.detach().unsqueeze(1), perm).squeeze(1)
                if up_module.bias is not None
                else None
            )
            down_bias = down_module.bias.detach() if down_module.bias is not None else None

            residual_bundle = pruned[expert_idx]
            indices = residual_bundle["indices"]

            gate_residual = (
                (indices, residual_bundle["gate"]) if residual_bundle["gate"].numel() > 0 else None
            )
            up_residual = (
                (indices, residual_bundle["up"]) if residual_bundle["up"].numel() > 0 else None
            )
            down_residual = (
                (indices, residual_bundle["down"]) if residual_bundle["down"].numel() > 0 else None
            )

            setattr(
                expert,
                attr_map["gate"],
                ParserRowProjection(
                    shared_backbone,
                    "gate",
                    gate_residual,
                    store_sparse=(self.spec.residual_mode == "unstructured"),
                    bias=gate_bias,
                ),
            )
            setattr(
                expert,
                attr_map["up"],
                ParserRowProjection(
                    shared_backbone,
                    "up",
                    up_residual,
                    store_sparse=(self.spec.residual_mode == "unstructured"),
                    bias=up_bias,
                ),
            )
            setattr(
                expert,
                attr_map["down"],
                ParserColumnProjection(
                    shared_backbone,
                    "down",
                    down_residual,
                    store_sparse=(self.spec.residual_mode == "unstructured"),
                    bias=down_bias,
                ),
            )

            self._patch_expert_forward(expert, attr_map)

        ensure_parser_moe_forward(moe_block)

    def _compute_backbone(self, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.spec.backbone == "mean":
            stacked = torch.stack([w.detach() for w in weights], dim=0)
            return stacked.mean(dim=0)
        if self.spec.backbone == "barycenter":
            return self._compute_barycenter(weights)
        raise ValueError(f"Unsupported backbone method: {self.spec.backbone}")

    def _compute_barycenter(self, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        try:
            import ot  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("POT (ot) is required for backbone=barycenter") from exc

        matrices = torch.stack([w.detach().float() for w in weights], dim=0)
        n = matrices.shape[0]
        points = matrices.shape[1]
        weights_uniform = torch.stack(
            [torch.full((points,), 1.0 / points, dtype=torch.float32) for _ in range(n)],
            dim=0,
        ).to(matrices.device)
        barycenter = ot.lp.free_support_barycenter(
            matrices,
            weights_uniform,
            matrices[0],
            numItermax=100,
            stopThr=1e-9,
        )
        return torch.tensor(barycenter, dtype=weights[0].dtype, device=weights[0].device)

    def _build_residuals(
        self,
        gate_weights: Sequence[torch.Tensor],
        up_weights: Sequence[torch.Tensor],
        down_weights: Sequence[torch.Tensor],
        backbone: SharedBackbone,
    ) -> List[Dict[str, torch.Tensor]]:
        residuals: List[Dict[str, torch.Tensor]] = []
        for gate, up, down in zip(gate_weights, up_weights, down_weights):
            residuals.append(
                {
                    "gate": gate - backbone.gate,
                    "up": up - backbone.up,
                    "down": down - backbone.down,
                }
            )
        return residuals

    def _require_activation_stats(self) -> None:
        if self._activation_experts is None:
            raise RuntimeError(
                "PARSER calibration stats are required (input samples). "
                "Run save_blocks.py with --calibrate or provide --calibration to compress_blocks.py."
            )

    def _get_expert_samples(
        self, layer_idx: int, moe_name: str, expert_idx: int
    ) -> Optional[torch.Tensor]:
        self._require_activation_stats()
        key = make_expert_key(layer_idx, moe_name, expert_idx)
        entry = self._activation_experts.get(key) if self._activation_experts else None
        if entry is None:
            return None
        samples = entry.get("samples")
        if samples is None:
            return None
        if not torch.is_tensor(samples):
            samples = torch.tensor(samples)
        return samples

    def _get_routing_factor(self, layer_idx: int, moe_name: str, expert_idx: int) -> float:
        if not bool(getattr(self.spec, "parser_routing_score", True)):
            return 1.0
        self._require_activation_stats()
        key = make_expert_key(layer_idx, moe_name, expert_idx)
        entry = self._activation_experts.get(key) if self._activation_experts else None
        if entry is None:
            return 1.0
        weight_sq_sum = entry.get("routing_softmax_sq_sum")
        if weight_sq_sum is None:
            weight_sq_sum = entry.get("routing_weight_sq_sum")
        total_tokens = entry.get("routing_total_tokens")
        if weight_sq_sum is None or total_tokens is None:
            return 1.0
        try:
            total = float(total_tokens)
            if total <= 0.0:
                return 1.0
            factor = float(weight_sq_sum) / total
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(factor):
            return 1.0
        return max(factor, 0.0)

    @staticmethod
    def _compute_magnitude_scores(
        gate_tensor: torch.Tensor,
        up_tensor: torch.Tensor,
        down_tensor: torch.Tensor,
    ) -> torch.Tensor:
        gate_norm = torch.sum(gate_tensor.pow(2), dim=1) / max(gate_tensor.shape[1], 1)
        up_norm = torch.sum(up_tensor.pow(2), dim=1) / max(up_tensor.shape[1], 1)
        down_norm = torch.sum(down_tensor.pow(2), dim=0) / max(down_tensor.shape[0], 1)
        return (gate_norm + up_norm + down_norm).to(
            device=gate_tensor.device, dtype=torch.float32
        )

    @staticmethod
    def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    @staticmethod
    def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
        sqrt_2 = math.sqrt(2.0)
        sqrt_2pi = math.sqrt(2.0 * math.pi)
        erf_term = torch.erf(x / sqrt_2)
        exp_term = torch.exp(-0.5 * x.pow(2))
        return 0.5 * (1.0 + erf_term) + (x * exp_term) / sqrt_2pi

    @classmethod
    def _activation_derivative(cls, act, preact: torch.Tensor) -> torch.Tensor:
        custom = getattr(act, "derivative", None)
        if callable(custom):
            return custom(preact)
        act_name = None
        if isinstance(act, nn.Module):
            act_name = act.__class__.__name__.lower()
        elif hasattr(act, "__name__"):
            act_name = str(getattr(act, "__name__", "") or "").lower()
        if act_name and "silu" in act_name:
            return cls._silu_derivative(preact)
        if act_name and "gelu" in act_name:
            return cls._gelu_derivative(preact)
        if act_name and "relu" in act_name:
            return (preact > 0).to(preact.dtype)
        return torch.ones_like(preact)

    def _compute_importance_state(
        self,
        *,
        experts: Sequence[nn.Module],
        residuals: List[Dict[str, torch.Tensor]],
        backbone: SharedBackbone,
        layer_idx: int,
        moe_name: str,
        need_param_scores: bool,
    ) -> List[ExpertImportance]:
        importance_list: List[ExpertImportance] = []
        routing_factors: List[float] = []
        routing_enabled = bool(getattr(self.spec, "parser_routing_score", True))
        for expert_idx, residual in enumerate(residuals):
            samples = self._get_expert_samples(layer_idx, moe_name, expert_idx)
            if samples is None or samples.numel() == 0:
                raise RuntimeError(
                    f"Missing calibration samples for "
                    f"layer={layer_idx} moe={moe_name} expert={expert_idx}. "
                    "Re-run save_blocks.py --calibrate."
                )
            input_dim = residual["gate"].shape[1]
            if samples.ndim == 1:
                samples = samples.unsqueeze(0)
            if samples.shape[1] != input_dim:
                raise ValueError(
                    f"Calibration samples dim mismatch for "
                    f"layer={layer_idx} moe={moe_name} expert={expert_idx}: "
                    f"samples dim={samples.shape[1]} vs gate input dim={input_dim}. "
                    "Re-run save_blocks.py --calibrate to regenerate stats."
                )
            act_fn = resolve_activation_fn(experts[expert_idx])
            if self._importance_mode == "formula":
                scores, scale_state = self._compute_unit_scores_formula(
                    samples=samples,
                    residual=residual,
                    backbone=backbone,
                    act_fn=act_fn,
                    compute_param_scores=need_param_scores,
                )
                source = "formula_calibration"
            elif self._importance_mode == "wanda":
                scores, scale_state = self._compute_unit_scores_wanda(
                    samples=samples,
                    residual=residual,
                    backbone=backbone,
                    act_fn=act_fn,
                )
                source = "wanda_activation"
            else:
                raise RuntimeError(f"Unsupported importance mode: {self._importance_mode}")
            if routing_enabled:
                routing_factor = self._get_routing_factor(layer_idx, moe_name, expert_idx)
                routing_factors.append(routing_factor)
                if routing_factor != 1.0:
                    scores = scores * routing_factor
                    source = f"{source}+routing_softmax_sq"
            importance_list.append(
                ExpertImportance(
                    unit_scores=scores,
                    gate_scale=scale_state["gate"],
                    up_scale=scale_state["up"],
                    down_scale=scale_state["down"],
                    source=source,
                )
            )
        if not routing_enabled:
            logger.info(
                "parser routing factors layer=%d moe=%s: disabled",
                layer_idx,
                moe_name,
            )
        elif routing_factors:
            finite_factors = [x for x in routing_factors if math.isfinite(x)]
            if finite_factors:
                logger.info(
                    "parser routing factors layer=%d moe=%s: min=%.6g mean=%.6g max=%.6g",
                    layer_idx,
                    moe_name,
                    min(finite_factors),
                    sum(finite_factors) / float(len(finite_factors)),
                    max(finite_factors),
                )
        return importance_list

    def _compute_unit_scores_formula(
        self,
        *,
        samples: torch.Tensor,
        residual: Dict[str, torch.Tensor],
        backbone: SharedBackbone,
        act_fn,
        compute_param_scores: bool,
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        gate_res = residual["gate"]
        up_res = residual["up"]
        down_res = residual["down"]
        hidden_dim = gate_res.shape[0]
        if up_res.shape[0] != hidden_dim or down_res.shape[1] != hidden_dim:
            raise ValueError("Mismatched hidden dimensions across residual projections.")
        if gate_res.shape[1] != up_res.shape[1]:
            raise ValueError("Gate/Up input dimension mismatch in residual projections.")

        device = gate_res.device
        gate_base = backbone.gate.to(device=device, dtype=torch.float32)
        up_base = backbone.up.to(device=device, dtype=torch.float32)
        down_base = backbone.down.to(device=device, dtype=torch.float32)
        gate_full = gate_base + gate_res.to(device=device, dtype=torch.float32)
        up_full = up_base + up_res.to(device=device, dtype=torch.float32)
        down_full = down_base + down_res.to(device=device, dtype=torch.float32)

        n_full = down_full.pow(2).sum(dim=0)
        n_base = down_base.pow(2).sum(dim=0)
        d_fb = (down_full * down_base).sum(dim=0)

        samples = samples.to(device=device, dtype=torch.float32)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        total = 0
        sum_ff = torch.zeros_like(n_full)
        sum_bb = torch.zeros_like(n_full)
        sum_fb = torch.zeros_like(n_full)
        sum_abs_hidden = torch.zeros_like(n_full)
        sum_gate_jac = None
        sum_up_m2 = None
        if compute_param_scores:
            sum_gate_jac = torch.zeros(
                (gate_res.shape[0], gate_res.shape[1]),
                device=device,
                dtype=torch.float32,
            )
            sum_up_m2 = torch.zeros(
                (gate_res.shape[1],),
                device=device,
                dtype=torch.float32,
            )

        for start in range(0, samples.shape[0], chunk_size):
            batch = samples[start : start + chunk_size]
            if batch.numel() == 0:
                continue
            pre_full = torch.matmul(batch, gate_full.transpose(0, 1))
            pre_base = torch.matmul(batch, gate_base.transpose(0, 1))

            up_full_hidden = torch.matmul(batch, up_full.transpose(0, 1))
            up_base_hidden = torch.matmul(batch, up_base.transpose(0, 1))

            hidden_full = combine_gate_up(act_fn, pre_full, up_full_hidden)
            hidden_base = combine_gate_up(act_fn, pre_base, up_base_hidden)

            sum_ff += hidden_full.pow(2).sum(dim=0)
            sum_bb += hidden_base.pow(2).sum(dim=0)
            sum_fb += (hidden_full * hidden_base).sum(dim=0)
            sum_abs_hidden += hidden_full.abs().sum(dim=0)
            if compute_param_scores:
                x_sq = batch.pow(2)
                act_prime = self._activation_derivative(act_fn, pre_full)
                gate_factor = act_prime.pow(2) * up_full_hidden.pow(2)
                sum_gate_jac += gate_factor.transpose(0, 1) @ x_sq
                sum_up_m2 += x_sq.sum(dim=0)
            total += int(batch.shape[0])

        if total <= 0:
            fallback = self._compute_magnitude_scores(gate_res, up_res, down_res)
            return fallback, {"gate": None, "up": None, "down": None}

        m_ff = sum_ff / float(total)
        m_bb = sum_bb / float(total)
        m_fb = sum_fb / float(total)

        scores = (m_ff * n_full + m_bb * n_base - 2.0 * m_fb * d_fb).clamp_min(0.0)
        scores = scores.to(device=gate_res.device, dtype=torch.float32)

        scale_state: Dict[str, Optional[torch.Tensor]] = {"gate": None, "up": None, "down": None}
        if compute_param_scores:
            gate_scale = sum_gate_jac / float(total) if sum_gate_jac is not None else None
            up_scale = sum_up_m2 / float(total) if sum_up_m2 is not None else None

            scale_state = {
                "gate": gate_scale,
                "up": up_scale,
                "down": None,
            }

        return scores, scale_state

    def _wanda_score_terms(
        self,
        *,
        samples: torch.Tensor,
        residual: Dict[str, torch.Tensor],
        backbone: SharedBackbone,
        act_fn,
        chunk_size: int = 256,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]]:
        """Compute the three per-channel Wanda components separately.

        Returns ``(gate_unit, up_unit, down_unit, scale_state)`` where each
        ``*_unit`` is a per-channel vector, or ``None`` when no calibration
        tokens were available (caller should fall back to magnitude scores).
        """
        gate_res = residual["gate"]
        up_res = residual["up"]
        down_res = residual["down"]
        hidden_dim = gate_res.shape[0]
        if up_res.shape[0] != hidden_dim or down_res.shape[1] != hidden_dim:
            raise ValueError("Mismatched hidden dimensions across residual projections.")
        if gate_res.shape[1] != up_res.shape[1]:
            raise ValueError("Gate/Up input dimension mismatch in residual projections.")

        device = gate_res.device
        gate_base = backbone.gate.to(device=device, dtype=torch.float32)
        up_base = backbone.up.to(device=device, dtype=torch.float32)
        gate_full = gate_base + gate_res.to(device=device, dtype=torch.float32)
        up_full = up_base + up_res.to(device=device, dtype=torch.float32)

        samples = samples.to(device=device, dtype=torch.float32)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        total = 0
        input_dim = gate_res.shape[1]
        sum_sq_x = torch.zeros((input_dim,), device=device, dtype=torch.float32)
        sum_sq_hidden = torch.zeros((hidden_dim,), device=device, dtype=torch.float32)

        for start in range(0, samples.shape[0], chunk_size):
            batch = samples[start : start + chunk_size]
            if batch.numel() == 0:
                continue

            sum_sq_x += batch.pow(2).sum(dim=0)

            pre_full = torch.matmul(batch, gate_full.transpose(0, 1))
            up_full_hidden = torch.matmul(batch, up_full.transpose(0, 1))
            hidden_full = combine_gate_up(act_fn, pre_full, up_full_hidden)
            sum_sq_hidden += hidden_full.pow(2).sum(dim=0)
            total += int(batch.shape[0])

        if total <= 0:
            return None

        x_l2 = torch.nan_to_num(
            (sum_sq_x / float(total)).sqrt(),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        h_l2 = torch.nan_to_num(
            (sum_sq_hidden / float(total)).sqrt(),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        x_l2 = x_l2.clamp_min(1e-12)
        h_l2 = h_l2.clamp_min(1e-12)

        gate_abs = gate_res.to(device=device, dtype=torch.float32).abs()
        up_abs = up_res.to(device=device, dtype=torch.float32).abs()
        down_abs = down_res.to(device=device, dtype=torch.float32).abs()

        gate_unit = (gate_abs * x_l2.view(1, -1)).sum(dim=1)
        up_unit = (up_abs * x_l2.view(1, -1)).sum(dim=1)
        down_unit = (down_abs * h_l2.view(1, -1)).sum(dim=0)

        scale_state: Dict[str, Optional[torch.Tensor]] = {
            "gate": x_l2.to(device=gate_res.device, dtype=torch.float32),
            "up": x_l2.to(device=gate_res.device, dtype=torch.float32),
            "down": h_l2.to(device=gate_res.device, dtype=torch.float32),
        }
        return gate_unit, up_unit, down_unit, scale_state

    def _compute_unit_scores_wanda(
        self,
        *,
        samples: torch.Tensor,
        residual: Dict[str, torch.Tensor],
        backbone: SharedBackbone,
        act_fn,
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        terms = self._wanda_score_terms(
            samples=samples,
            residual=residual,
            backbone=backbone,
            act_fn=act_fn,
            chunk_size=chunk_size,
        )
        if terms is None:
            fallback = self._compute_magnitude_scores(
                residual["gate"], residual["up"], residual["down"]
            )
            return fallback, {"gate": None, "up": None, "down": None}
        gate_unit, up_unit, down_unit, scale_state = terms
        scores = (gate_unit + up_unit + down_unit).clamp_min(0.0)
        scores = scores.to(device=residual["gate"].device, dtype=torch.float32)
        return scores, scale_state

    def _prune_structured(
        self, residuals: List[Dict[str, torch.Tensor]], unit_scores_list: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        sparsity = float(max(0.0, min(1.0, self.spec.sparsity)))
        min_units = max(0, int(self.spec.min_units))
        scope = self.spec.residual_scope

        scores_per_expert = [scores.cpu() for scores in unit_scores_list]
        hidden_dims = [int(score.numel()) for score in scores_per_expert]
        num_experts = len(hidden_dims)

        keep_map: Dict[int, List[int]] = {i: [] for i in range(num_experts)}

        if scope == "expert":
            for idx, scores in enumerate(scores_per_expert):
                hidden_dim = scores.numel()
                keep = max(min_units, int(round((1.0 - sparsity) * hidden_dim)))
                keep = min(max(keep, 0), hidden_dim)
                if keep == 0:
                    keep_map[idx] = []
                    continue
                topk = torch.topk(scores, keep).indices.tolist()
                keep_map[idx] = sorted(set(int(i) for i in topk))
        else:
            total_units = sum(hidden_dims)
            target_keep = int(round((1.0 - sparsity) * total_units))
            target_keep = max(target_keep, min_units * num_experts)
            reserved = 0
            for idx, scores in enumerate(scores_per_expert):
                if scores.numel() == 0:
                    continue
                keep = min(min_units, scores.numel())
                if keep > 0:
                    topk = torch.topk(scores, keep).indices.tolist()
                    keep_map[idx].extend(int(i) for i in topk)
                    reserved += keep

            remaining = max(target_keep - reserved, 0)
            if remaining > 0:
                candidates: List[Tuple[float, int, int]] = []
                for idx, scores in enumerate(scores_per_expert):
                    existing = set(keep_map[idx])
                    for unit_idx, score in enumerate(scores.tolist()):
                        if unit_idx in existing:
                            continue
                        candidates.append((score, idx, unit_idx))
                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, exp_idx, unit_idx in candidates[:remaining]:
                    keep_map[exp_idx].append(unit_idx)

            for idx in keep_map:
                keep_map[idx] = sorted(set(keep_map[idx]))

        return self._build_pruned_residuals(residuals, keep_map)

    def _prune_unstructured(
        self,
        residuals: List[Dict[str, torch.Tensor]],
        importance: List[ExpertImportance],
    ) -> List[Dict[str, torch.Tensor]]:
        sparsity = float(max(0.0, min(1.0, self.spec.sparsity)))
        scope = self.spec.residual_scope

        pruned: List[Dict[str, torch.Tensor]] = []

        def _score_tensors(
            gate: torch.Tensor,
            up: torch.Tensor,
            down: torch.Tensor,
            imp: ExpertImportance,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            unit_scores = imp.unit_scores.to(gate.device)

            def _abs_power(tensor: torch.Tensor, power: float) -> torch.Tensor:
                base = tensor.abs().to(torch.float32)
                if power == 1.0:
                    return base
                return base.pow(power)

            def _normalize_scale(scale_tensor: torch.Tensor) -> torch.Tensor:
                flat = scale_tensor.reshape(-1)
                if flat.numel() == 0:
                    return scale_tensor
                center = flat.mean()
                if (not torch.isfinite(center).item()) or (center.item() <= 0.0):
                    center = flat.median()
                center = center.clamp_min(1e-12)
                return scale_tensor / center

            def _vector_scale(
                scale_tensor: Optional[torch.Tensor],
                width: int,
                *,
                device: torch.device,
            ) -> Optional[torch.Tensor]:
                if scale_tensor is None:
                    return None
                vec = scale_tensor.to(device=device, dtype=torch.float32).reshape(-1)
                if vec.numel() != width:
                    return None
                return _normalize_scale(vec)

            importance_source = (imp.source or "").strip().lower()
            if importance_source.startswith("wanda"):
                gate_scale = _vector_scale(imp.gate_scale, gate.shape[1], device=gate.device)
                up_scale = _vector_scale(imp.up_scale, up.shape[1], device=up.device)
                down_scale = _vector_scale(imp.down_scale, down.shape[1], device=down.device)
                if gate_scale is None:
                    gate_scale = torch.ones((gate.shape[1],), device=gate.device, dtype=torch.float32)
                if up_scale is None:
                    up_scale = torch.ones((up.shape[1],), device=up.device, dtype=torch.float32)
                if down_scale is None:
                    down_scale = torch.ones((down.shape[1],), device=down.device, dtype=torch.float32)

                gate_scores = _abs_power(gate, 1.0) * gate_scale.view(1, -1)
                up_scores = _abs_power(up, 1.0) * up_scale.view(1, -1)
                down_scores = _abs_power(down, 1.0) * down_scale.view(1, -1)
                return gate_scores, up_scores, down_scores

            score_power = 2.0
            if imp.gate_scale is None or imp.up_scale is None:
                gate_scores = unit_scores.view(-1, 1) * _abs_power(gate, score_power)
                up_scores = unit_scores.view(-1, 1) * _abs_power(up, score_power)
                down_scores = _abs_power(down, score_power) * unit_scores.view(1, -1)
                down_scale = _vector_scale(imp.down_scale, down.shape[1], device=down.device)
                if down_scale is not None:
                    down_scores = down_scores * down_scale.view(1, -1)
                return gate_scores, up_scores, down_scores

            gate_scale = imp.gate_scale.to(gate.device)
            up_scale = imp.up_scale.to(gate.device)
            down_scale = _vector_scale(imp.down_scale, down.shape[1], device=down.device)

            gate_scores = unit_scores.view(-1, 1) * _normalize_scale(
                gate_scale
            ) * _abs_power(gate, score_power)
            up_scores = unit_scores.view(-1, 1) * _normalize_scale(
                up_scale
            ).view(1, -1) * _abs_power(up, score_power)
            down_scores = (
                _abs_power(down, score_power) * unit_scores.view(1, -1)
            )
            if down_scale is not None:
                down_scores = down_scores * down_scale.view(1, -1)
            return gate_scores, up_scores, down_scores

        def _compact_pruned_tensors(
            gate_pruned: torch.Tensor,
            up_pruned: torch.Tensor,
            down_pruned: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            gate_nonzero = gate_pruned.abs().sum(dim=1) > 0
            up_nonzero = up_pruned.abs().sum(dim=1) > 0
            down_nonzero = down_pruned.abs().sum(dim=0) > 0
            active_mask = gate_nonzero | up_nonzero | down_nonzero
            active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()

            if active_indices.numel() > 0:
                gate_compact = gate_pruned.index_select(0, active_indices)
                up_compact = up_pruned.index_select(0, active_indices)
                down_compact = down_pruned.index_select(1, active_indices)
            else:
                gate_compact = gate_pruned[:0]
                up_compact = up_pruned[:0]
                down_compact = down_pruned[:, :0]

            return (
                active_indices.to(device=gate_pruned.device, dtype=torch.long),
                gate_compact,
                up_compact,
                down_compact,
            )

        if scope == "expert":
            for expert_idx, residual in enumerate(residuals):
                gate = residual["gate"]
                up = residual["up"]
                down = residual["down"]
                total_params = gate.numel() + up.numel() + down.numel()
                keep = int(round((1.0 - sparsity) * total_params))
                keep = max(keep, 0)
                if keep == 0 and total_params > 0:
                    keep = 1

                if total_params == 0:
                    pruned.append(
                        {
                            "indices": torch.tensor([], device=gate.device, dtype=torch.long),
                            "gate": gate[:0],
                            "up": up[:0],
                            "down": down[:, :0],
                        }
                    )
                    continue

                gate_scores, up_scores, down_scores = _score_tensors(
                    gate, up, down, importance[expert_idx]
                )

                all_scores = torch.cat(
                    [
                        gate_scores.reshape(-1),
                        up_scores.reshape(-1),
                        down_scores.reshape(-1),
                    ],
                    dim=0,
                )
                top_vals, _ = torch.topk(all_scores, min(keep, all_scores.numel()), largest=True)
                threshold = top_vals.min() if top_vals.numel() > 0 else all_scores.max()

                gate_mask = gate_scores >= threshold
                up_mask = up_scores >= threshold
                down_mask = down_scores >= threshold

                gate_pruned = gate * gate_mask
                up_pruned = up * up_mask
                down_pruned = down * down_mask

                active_indices, gate_compact, up_compact, down_compact = _compact_pruned_tensors(
                    gate_pruned, up_pruned, down_pruned
                )

                pruned.append(
                    {
                        "indices": active_indices,
                        "gate": gate_compact,
                        "up": up_compact,
                        "down": down_compact,
                    }
                )
        else:
            score_segments: List[torch.Tensor] = []
            segment_meta: List[Tuple[int, str, torch.Size, int]] = []
            total_weights = 0

            for expert_idx, residual in enumerate(residuals):
                gate = residual["gate"]
                up = residual["up"]
                down = residual["down"]
                gate_scores, up_scores, down_scores = _score_tensors(
                    gate, up, down, importance[expert_idx]
                )
                for name, scores in (
                    ("gate", gate_scores),
                    ("up", up_scores),
                    ("down", down_scores),
                ):
                    flat = scores.reshape(-1).cpu()
                    score_segments.append(flat)
                    segment_meta.append((expert_idx, name, scores.shape, flat.numel()))
                    total_weights += flat.numel()

            keep_weights = int(round((1.0 - sparsity) * total_weights))
            if keep_weights == 0 and total_weights > 0:
                keep_weights = 1
            if total_weights == 0:
                return pruned

            all_scores = torch.cat(score_segments, dim=0)
            top_vals, top_idx = torch.topk(
                all_scores, min(keep_weights, all_scores.numel()), largest=True
            )
            mask_flat = torch.zeros_like(all_scores, dtype=torch.bool)
            mask_flat[top_idx] = True

            offset = 0
            masks: Dict[int, Dict[str, torch.Tensor]] = {}
            for expert_idx, name, shape, length in segment_meta:
                segment_mask = mask_flat[offset : offset + length].reshape(shape)
                offset += length
                entry = masks.setdefault(expert_idx, {})
                entry[name] = segment_mask

            for expert_idx, residual in enumerate(residuals):
                gate = residual["gate"]
                up = residual["up"]
                down = residual["down"]
                expert_masks = masks.get(expert_idx, {})
                gate_mask = expert_masks.get("gate", torch.zeros_like(gate, dtype=torch.bool))
                up_mask = expert_masks.get("up", torch.zeros_like(up, dtype=torch.bool))
                down_mask = expert_masks.get("down", torch.zeros_like(down, dtype=torch.bool))

                gate_pruned = gate * gate_mask.to(device=gate.device)
                up_pruned = up * up_mask.to(device=up.device)
                down_pruned = down * down_mask.to(device=down.device)

                active_indices, gate_compact, up_compact, down_compact = _compact_pruned_tensors(
                    gate_pruned, up_pruned, down_pruned
                )
                pruned.append(
                    {
                        "indices": active_indices,
                        "gate": gate_compact,
                        "up": up_compact,
                        "down": down_compact,
                    }
                )

        return pruned

    def _build_pruned_residuals(
        self,
        residuals: List[Dict[str, torch.Tensor]],
        keep_map: Dict[int, List[int]],
    ) -> List[Dict[str, torch.Tensor]]:
        pruned: List[Dict[str, torch.Tensor]] = []
        for idx, residual in enumerate(residuals):
            keep_indices = torch.tensor(
                keep_map.get(idx, []),
                device=residual["gate"].device,
                dtype=torch.long,
            )
            if keep_indices.numel() > 0:
                gate_res = residual["gate"].index_select(0, keep_indices)
                up_res = residual["up"].index_select(0, keep_indices)
                down_res = residual["down"].index_select(1, keep_indices)
            else:
                gate_res = residual["gate"][:0]
                up_res = residual["up"][:0]
                down_res = residual["down"][:, :0]

            pruned.append(
                {
                    "indices": keep_indices,
                    "gate": gate_res,
                    "up": up_res,
                    "down": down_res,
                }
            )
        return pruned

    def _prepare_hidden_components(
        self,
        gate_module: ParserRowProjection,
        backbone_gate: torch.Tensor,
        residual_gate: Optional[torch.Tensor],
        backbone_up: torch.Tensor,
        residual_up: Optional[torch.Tensor],
        act_fn,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        indices = getattr(gate_module, "residual_indices", None)
        has_residual = (
            indices is not None
            and indices.numel() > 0
            and residual_gate is not None
            and residual_up is not None
        )

        if has_residual:
            use_inplace = (
                getattr(gate_module, "residual_indices", None) is not None
                and gate_module is not None
                and not torch.is_grad_enabled()
            )
            if not use_inplace:
                backbone_gate = backbone_gate.clone()
                backbone_up = backbone_up.clone()
            backbone_gate.index_add_(1, indices, residual_gate)
            backbone_up.index_add_(1, indices, residual_up)
            combined_hidden = combine_gate_up(act_fn, backbone_gate, backbone_up)
            residual_hidden = combined_hidden.index_select(1, indices)
            return combined_hidden, residual_hidden

        backbone_hidden = combine_gate_up(act_fn, backbone_gate, backbone_up)
        return backbone_hidden, None

    def _patch_expert_forward(self, expert: nn.Module, attr_map: Dict[str, str]) -> None:
        if getattr(expert, "_parser_forward_patched", False):
            return

        original_forward = expert.forward

        expert._parser_original_forward = original_forward
        expert._parser_attr_map = attr_map
        if not hasattr(expert.__class__, "parser_forward"):
            setattr(expert.__class__, "parser_forward", parser_forward)
        expert.forward = types.MethodType(parser_forward, expert)
        expert._parser_forward_patched = True
