from __future__ import annotations

import sys
from types import MethodType

import torch
from torch import nn


def _is_bound_unimplemented_forward(module: nn.Module, value: object) -> bool:
    return (
        isinstance(value, MethodType)
        and value.__self__ is module
        and getattr(value, "__name__", "") == "_forward_unimplemented"
    )


def _looks_like_accelerate_forward_wrapper(value: object) -> bool:
    name = getattr(value, "__name__", "")
    qualname = getattr(value, "__qualname__", "")
    return name == "new_forward" or "add_hook_to_module.<locals>.new_forward" in qualname


def remove_runtime_forward_wrappers(module: nn.Module) -> int:
    """
    Remove runtime-only forward wrappers before serializing a module tree.

    This handles accelerate hooks (e.g. when using device_map=auto) and avoids
    storing instance-bound ``forward`` methods that can fail on unpickle.
    """
    restored = 0
    for submodule in module.modules():
        had_old_forward = hasattr(submodule, "_old_forward")
        if had_old_forward:
            restored += 1
            if "forward" in submodule.__dict__:
                try:
                    delattr(submodule, "forward")
                except Exception:
                    old_forward = getattr(submodule, "_old_forward", None)
                    if (
                        old_forward is not None
                        and not _is_bound_unimplemented_forward(submodule, old_forward)
                    ):
                        submodule.forward = old_forward  # type: ignore[assignment]
        else:
            forward_value = submodule.__dict__.get("forward")
            if forward_value is not None and _looks_like_accelerate_forward_wrapper(forward_value):
                try:
                    delattr(submodule, "forward")
                except Exception:
                    pass

        try:
            delattr(submodule, "_old_forward")
        except AttributeError:
            pass
        try:
            delattr(submodule, "_hf_hook")
        except AttributeError:
            pass
    return restored


def install_modulelist_forward_unimplemented_alias() -> bool:
    """
    Compatibility shim for old checkpoints that serialized ModuleList.forward.

    Some artifacts may contain a bound method named ``_forward_unimplemented``.
    Unpickling then expects ``ModuleList._forward_unimplemented`` to exist.
    """
    module_list_cls = nn.ModuleList
    if hasattr(module_list_cls, "_forward_unimplemented"):
        return False
    try:
        from torch.nn.modules.module import _forward_unimplemented
    except Exception:
        return False
    setattr(module_list_cls, "_forward_unimplemented", _forward_unimplemented)
    return True


def install_parser_forward_alias() -> bool:
    """
    Compatibility shim for parser-compressed checkpoints.

    Some parser checkpoints pickle a reference to ``expert.parser_forward``.
    If that method is missing on the loaded expert class, unpickling fails with
    ``AttributeError``.
    """
    try:
        from moe_pipeline.methods.parser import parser_forward as pf
        from moe_pipeline.methods.parser import parser_moe_forward as pmf
        from moe_pipeline.moe_introspection import (
            DEEPSEEK_EXPERT_CLASS_NAMES,
            MOE_BLOCK_CLASS_NAMES,
            OLMOE_EXPERT_CLASS_NAMES,
            QWEN_EXPERT_CLASS_NAMES,
        )
    except Exception:
        return False

    installed = False
    all_names = (
        QWEN_EXPERT_CLASS_NAMES
        | DEEPSEEK_EXPERT_CLASS_NAMES
        | OLMOE_EXPERT_CLASS_NAMES
    )
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for name in all_names:
            try:
                cls = getattr(mod, name, None)
                if cls is not None and isinstance(cls, type) and not hasattr(cls, "parser_forward"):
                    setattr(cls, "parser_forward", pf)
                    installed = True
            except Exception:
                continue
        for name in MOE_BLOCK_CLASS_NAMES:
            try:
                cls = getattr(mod, name, None)
                if cls is not None and isinstance(cls, type) and not hasattr(cls, "parser_moe_forward"):
                    setattr(cls, "parser_moe_forward", pmf)
                    installed = True
            except Exception:
                continue
    return installed


def sanitize_loaded_block(module: nn.Module) -> int:
    """
    Drop non-essential runtime forward attributes after loading a block.
    """
    cleaned = 0
    for submodule in module.modules():
        forward_value = submodule.__dict__.get("forward")
        if forward_value is not None and (
            _is_bound_unimplemented_forward(submodule, forward_value)
            or _looks_like_accelerate_forward_wrapper(forward_value)
        ):
            try:
                delattr(submodule, "forward")
                cleaned += 1
            except Exception:
                pass
        try:
            delattr(submodule, "_old_forward")
        except AttributeError:
            pass
        try:
            delattr(submodule, "_hf_hook")
        except AttributeError:
            pass
    return cleaned
