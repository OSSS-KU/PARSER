"""Compatibility shims for transformers version drift.

Custom modeling files distributed via ``trust_remote_code`` (e.g.
``modeling_deepseek.py`` for DeepSeek-V2-Lite) call cache-API methods that
have been renamed/removed in newer ``transformers`` releases. To keep these
trust_remote_code models working without forking the cached files, we
monkey-patch the missing methods onto the relevant cache classes at import
time.

Currently patched:
    * ``transformers.cache_utils.DynamicCache.get_usable_length``
        Older transformers had this; newer versions only ship
        ``get_seq_length``. DeepSeek-V2 modeling code calls the legacy name.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_PATCHED = False


def install_cache_compat_shim() -> None:
    """Idempotently install the DynamicCache.get_usable_length compat shim."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from transformers.cache_utils import DynamicCache
    except Exception as exc:
        _log.debug("transformers.cache_utils.DynamicCache unavailable: %s", exc)
        return

    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:  # type: ignore[no-redef]
            try:
                return int(self.get_seq_length(layer_idx))
            except Exception:
                return 0

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]
        _log.info(
            "compat: added DynamicCache.get_usable_length shim "
            "(forwards to get_seq_length for trust_remote_code models)."
        )

    for cls_name in ("StaticCache", "EncoderDecoderCache", "SinkCache", "SlidingWindowCache"):
        try:
            cls = getattr(__import__("transformers.cache_utils", fromlist=[cls_name]), cls_name, None)
        except Exception:
            cls = None
        if cls is not None and not hasattr(cls, "get_usable_length"):
            def _gul(self, new_seq_length: int, layer_idx: int = 0) -> int:  # type: ignore[no-redef]
                try:
                    return int(self.get_seq_length(layer_idx))
                except Exception:
                    return 0
            try:
                cls.get_usable_length = _gul  # type: ignore[attr-defined]
            except Exception:
                pass

    _PATCHED = True
