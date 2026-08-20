from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional

import torch

_LOG = logging.getLogger(__name__)


@dataclass
class VramStats:
    total_allocated_mb: Optional[float]
    total_reserved_mb: Optional[float]
    per_device_allocated_mb: Optional[Dict[str, float]]
    per_device_reserved_mb: Optional[Dict[str, float]]
    model_size_mb: Optional[float]
    per_device_model_size_mb: Optional[Dict[str, float]]


@dataclass
class VramConcurrentPeak:
    total_allocated_mb: Optional[float]
    total_reserved_mb: Optional[float]
    per_device_allocated_mb: Optional[Dict[str, float]]
    per_device_reserved_mb: Optional[Dict[str, float]]
    sample_count: int
    interval_sec: float


def _iter_valid_cuda_indices():
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(idx)
            yield idx
        except Exception as exc:  # pragma: no cover - depends on runtime/driver state.
            _LOG.warning("Skipping CUDA device index %d for VRAM stats: %s", idx, exc)


def reset_vram_stats() -> None:
    for idx in _iter_valid_cuda_indices():
        try:
            with torch.cuda.device(idx):
                torch.cuda.reset_peak_memory_stats()
        except RuntimeError as exc:
            _LOG.warning("Failed to reset VRAM peak stats for cuda:%d: %s", idx, exc)


class VramConcurrentPeakMonitor:
    """Poll current CUDA memory to estimate peak total usage at one instant.

    CUDA exposes exact per-device allocator peaks, but not an exact process-wide
    simultaneous peak across devices. This monitor samples current allocations on
    all visible CUDA devices and keeps the sample with the largest total.
    """

    def __init__(self, *, interval_sec: float = 0.05) -> None:
        self.interval_sec = max(0.001, float(interval_sec))
        self._devices = list(_iter_valid_cuda_indices()) if torch.cuda.is_available() else []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._sample_count = 0
        self._peak_total_allocated_mb: Optional[float] = None
        self._peak_total_reserved_mb: Optional[float] = None
        self._peak_per_device_allocated_mb: Optional[Dict[str, float]] = None
        self._peak_per_device_reserved_mb: Optional[Dict[str, float]] = None

    def start(self) -> "VramConcurrentPeakMonitor":
        if not self._devices:
            return self
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="vram-concurrent-peak", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> VramConcurrentPeak:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))
            self._thread = None
        if self._devices:
            self._sample_once()
        with self._lock:
            return VramConcurrentPeak(
                total_allocated_mb=self._peak_total_allocated_mb,
                total_reserved_mb=self._peak_total_reserved_mb,
                per_device_allocated_mb=dict(self._peak_per_device_allocated_mb or {}),
                per_device_reserved_mb=dict(self._peak_per_device_reserved_mb or {}),
                sample_count=int(self._sample_count),
                interval_sec=float(self.interval_sec),
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample_once()

    def _sample_once(self) -> None:
        per_alloc: Dict[str, float] = {}
        per_reserved: Dict[str, float] = {}
        total_alloc = 0.0
        total_reserved = 0.0
        for idx in self._devices:
            try:
                allocated = torch.cuda.memory_allocated(idx) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            except RuntimeError as exc:
                _LOG.warning("Failed to sample VRAM for cuda:%d: %s", idx, exc)
                continue
            device = f"cuda:{idx}"
            per_alloc[device] = allocated
            per_reserved[device] = reserved
            total_alloc += allocated
            total_reserved += reserved
        with self._lock:
            self._sample_count += 1
            if self._peak_total_allocated_mb is None or total_alloc > self._peak_total_allocated_mb:
                self._peak_total_allocated_mb = total_alloc
                self._peak_total_reserved_mb = total_reserved
                self._peak_per_device_allocated_mb = dict(per_alloc)
                self._peak_per_device_reserved_mb = dict(per_reserved)


def _storage_key_nbytes(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())


def _iter_tensor_components(tensor: torch.Tensor):
    if tensor.layout == torch.sparse_csr:
        yield tensor.crow_indices()
        yield tensor.col_indices()
        yield tensor.values()
        return
    if tensor.layout == torch.sparse_coo:
        coalesced = tensor.coalesce()
        yield coalesced.indices()
        yield coalesced.values()
        return
    yield tensor


def _iter_model_size_tensors(model: torch.nn.Module):
    yield from chain(model.parameters(), model.buffers())

    seen_shared = set()
    for module in model.modules():
        shared = getattr(module, "_shared_backbone", None)
        if not isinstance(shared, torch.nn.Module):
            continue
        sid = id(shared)
        if sid in seen_shared:
            continue
        seen_shared.add(sid)
        yield from chain(shared.parameters(), shared.buffers())


def _collect_model_size_stats(model: Optional[torch.nn.Module]) -> tuple[Optional[float], Optional[Dict[str, float]]]:
    if model is None or not torch.cuda.is_available():
        return None, None

    per_device_bytes: Dict[str, int] = {}
    seen_storages = set()
    for tensor in _iter_model_size_tensors(model):
        if not tensor.is_cuda:
            continue
        device = str(tensor.device)
        for component in _iter_tensor_components(tensor):
            try:
                ptr, nbytes = _storage_key_nbytes(component)
                key = (str(component.device), ptr, nbytes)
            except Exception:
                nbytes = int(component.numel() * component.element_size())
                key = (str(component.device), id(component), nbytes)
            if key in seen_storages:
                continue
            seen_storages.add(key)
            per_device_bytes[str(component.device)] = (
                per_device_bytes.get(str(component.device), 0) + nbytes
            )

    if not per_device_bytes:
        return 0.0, {}

    per_device_mb = {
        device: nbytes / (1024 ** 2)
        for device, nbytes in per_device_bytes.items()
    }
    return sum(per_device_mb.values()), per_device_mb


def collect_vram_stats(model: Optional[torch.nn.Module] = None) -> VramStats:
    if not torch.cuda.is_available():
        return VramStats(
            total_allocated_mb=None,
            total_reserved_mb=None,
            per_device_allocated_mb=None,
            per_device_reserved_mb=None,
            model_size_mb=None,
            per_device_model_size_mb=None,
        )
    model_size_mb, per_device_model_size_mb = _collect_model_size_stats(model)
    per_alloc: Dict[str, float] = {}
    per_reserved: Dict[str, float] = {}
    total_alloc = 0.0
    total_reserved = 0.0
    for idx in _iter_valid_cuda_indices():
        try:
            with torch.cuda.device(idx):
                allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        except RuntimeError as exc:
            _LOG.warning("Failed to read VRAM peak stats for cuda:%d: %s", idx, exc)
            continue
        per_alloc[f"cuda:{idx}"] = allocated
        per_reserved[f"cuda:{idx}"] = reserved
        total_alloc += allocated
        total_reserved += reserved
    return VramStats(
        total_allocated_mb=total_alloc,
        total_reserved_mb=total_reserved,
        per_device_allocated_mb=per_alloc,
        per_device_reserved_mb=per_reserved,
        model_size_mb=model_size_mb,
        per_device_model_size_mb=per_device_model_size_mb,
    )
