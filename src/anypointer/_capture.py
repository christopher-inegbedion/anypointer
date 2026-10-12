"""Read the pointing heads' queries out of Qwen3-VL attention.

GUI-Owl is about to write an x coordinate; some of its attention heads are, at
that moment, looking at the target. ``point`` scores the screen's blocks with
those heads: it needs each chosen layer's *query* for the item's last token, and
nothing else from the attention. This module wraps ``qlang.Attention.__call__``
once, and records that query while ``STATE.on`` is set.
"""

from __future__ import annotations

from typing import Any

from mlx_vlm.models.qwen3_vl import language as qlang


class _State:
    on: bool = False
    layers: set[int] | None = None  # record only these layers (None: all)
    queries: dict[int, Any]

    def __init__(self) -> None:
        self.queries = {}


STATE = _State()
_ORIGINAL = qlang.Attention.__call__
_INSTALLED = False


def _attention(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None):
    li = getattr(self, "_li", None)
    if STATE.on and (STATE.layers is None or li in STATE.layers):
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, n, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, n, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        if position_embeddings is None:
            q, _ = self.rotary_emb.apply_rotary(q, k, position_ids, unsqueeze_dim=1)
        else:
            cos, sin = position_embeddings
            q, _ = qlang.apply_multimodal_rotary_pos_emb(q, k, cos, sin)
        STATE.queries[li] = q[0, :, -1, :]  # (heads, head_dim): the last token, rotated
    return _ORIGINAL(self, x, mask, cache, position_ids, position_embeddings)


def install() -> None:
    """Idempotent; the wrapper does nothing unless ``STATE.on``."""
    global _INSTALLED
    if not _INSTALLED:
        qlang.Attention.__call__ = _attention
        _INSTALLED = True
