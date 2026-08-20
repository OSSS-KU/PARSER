from __future__ import annotations

from typing import Dict, List, Tuple

from torch import nn

QWEN_EXPERT_CLASS_NAMES = {
    "Qwen2MoeMLP",
}

DEEPSEEK_EXPERT_CLASS_NAMES = {
    "DeepseekV2MLP",
}

OLMOE_EXPERT_CLASS_NAMES = {
    "OlmoeMLP",
}

QWEN_MOE_BLOCK_CLASS_NAMES = {
    "Qwen2MoeSparseMoeBlock",
}

DEEPSEEK_MOE_BLOCK_CLASS_NAMES = {
    "DeepseekV2MoE",
}

OLMOE_MOE_BLOCK_CLASS_NAMES = {
    "OlmoeSparseMoeBlock",
}

MOE_BLOCK_CLASS_NAMES = (
    QWEN_MOE_BLOCK_CLASS_NAMES
    | DEEPSEEK_MOE_BLOCK_CLASS_NAMES
    | OLMOE_MOE_BLOCK_CLASS_NAMES
)

_EXPERT_ATTR_MAP_CANDIDATES: Tuple[Dict[str, str], ...] = (
    {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
)


def _matches_attr_map(expert: nn.Module, attr_map: Dict[str, str]) -> bool:
    for attr_name in attr_map.values():
        module = getattr(expert, attr_name, None)
        if not isinstance(module, nn.Module):
            return False
    return True


def _infer_expert_attr_map(expert: nn.Module) -> Dict[str, str]:
    for attr_map in _EXPERT_ATTR_MAP_CANDIDATES:
        if _matches_attr_map(expert, attr_map):
            return dict(attr_map)
    raise ValueError(f"Unsupported expert module class '{expert.__class__.__name__}'.")


def expert_proj_attr_map(expert: nn.Module) -> Dict[str, str]:
    name = expert.__class__.__name__
    if (
        name in QWEN_EXPERT_CLASS_NAMES
        or name in DEEPSEEK_EXPERT_CLASS_NAMES
        or name in OLMOE_EXPERT_CLASS_NAMES
    ):
        return {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    return _infer_expert_attr_map(expert)


def is_supported_moe_block(module: nn.Module) -> bool:
    if module.__class__.__name__ in MOE_BLOCK_CLASS_NAMES:
        return True
    experts = getattr(module, "experts", None)
    if not isinstance(experts, (list, tuple, nn.ModuleList)) or not experts:
        return False
    first = experts[0]
    if not isinstance(first, nn.Module):
        return False
    try:
        expert_proj_attr_map(first)
        return True
    except ValueError:
        return False


def get_experts(moe_block: nn.Module) -> List[nn.Module]:
    if hasattr(moe_block, "experts"):
        experts = getattr(moe_block, "experts")
        if isinstance(experts, (list, tuple, nn.ModuleList)):
            modules = [expert for expert in experts if isinstance(expert, nn.Module)]
            if modules:
                return modules
    raise ValueError(
        f"Unsupported MoE block structure for '{moe_block.__class__.__name__}': "
        "expected an '.experts' attribute."
    )


def find_moe_blocks(block: nn.Module) -> List[Tuple[str, nn.Module]]:
    moe_blocks = []
    for name, module in block.named_modules():
        if is_supported_moe_block(module):
            moe_blocks.append((name, module))
    return moe_blocks
