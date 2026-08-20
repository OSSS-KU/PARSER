from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class BlockContainer:
    path: str
    container: nn.ModuleList


class HfModelAdapter:
    """Locate and manipulate transformer blocks across common HF model layouts."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._block_container = self._resolve_block_container()

    @property
    def container_path(self) -> str:
        return self._block_container.path

    def blocks(self) -> List[nn.Module]:
        return list(self._block_container.container)

    def set_block(self, index: int, block: nn.Module) -> None:
        self._block_container.container[index] = block

    def block_count(self) -> int:
        return len(self._block_container.container)

    def _resolve_block_container(self) -> BlockContainer:
        candidates: List[Tuple[str, callable]] = [
            ("model.layers", lambda m: m.model.layers),
        ]

        for path, getter in candidates:
            try:
                container = getter(self.model)
            except AttributeError:
                continue
            if isinstance(container, nn.ModuleList):
                return BlockContainer(path=path, container=container)

        available = [
            name
            for name, value in self.model.__dict__.items()
            if isinstance(value, nn.ModuleList)
        ]
        raise ValueError(
            "Unable to locate transformer blocks. "
            f"Checked paths: {[p for p, _ in candidates]}. "
            f"ModuleList attributes found: {available}."
        )


def ensure_device(module: nn.Module, device: torch.device) -> nn.Module:
    return module.to(device)


def describe_model(model: nn.Module) -> str:
    name = model.__class__.__name__
    if hasattr(model, "config") and getattr(model.config, "_name_or_path", None):
        return f"{name} ({model.config._name_or_path})"
    return name
