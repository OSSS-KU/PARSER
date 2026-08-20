from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from torch import nn


@dataclass
class CompressionSpec:
    method: str
    sparsity: float
    residual_mode: str = "structured"
    residual_scope: str = "expert"
    backbone: str = "mean"
    importance: str = "formula"
    parser_routing_score: bool = True
    min_units: int = 4


class BaseBlockCompressor(ABC):
    """Base interface for block-level compression methods."""

    def __init__(self, spec: CompressionSpec):
        self.spec = spec

    def prepare(
        self,
        total_layers: int,
        layer_indices: List[int],
        *,
        blocks_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """Cross-layer planning hook. Override for methods needing global optimization.

        Called once before compress_block() loop. Default is no-op.
        """
        pass

    @abstractmethod
    def compress_block(self, block: nn.Module, layer_idx: int) -> nn.Module:
        raise NotImplementedError
