from __future__ import annotations

from typing import Dict

from moe_pipeline.methods.base import BaseBlockCompressor, CompressionSpec
from moe_pipeline.methods.parser import ParserBlockCompressor


def build_compressor(
    spec: CompressionSpec,
    *,
    calibration_stats: Dict | None = None,
) -> BaseBlockCompressor:
    method = spec.method.lower()
    if method in {"parser-l-sp", "parser-e-sp"}:
        return ParserBlockCompressor(spec, activation_stats=calibration_stats)
    raise ValueError(f"Unsupported compression method: {spec.method}")
