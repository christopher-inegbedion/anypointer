"""GUI-Owl-1.5 pointing in two steps: read the screen once, then answer items fast.

GUI-Owl reads ``[screen]["Locate the element: "][item]". Everything before the
item is the same for any item on that screen, so it is computed once
(``Engine.read``: vision encoder and language layers ``0..K-1``, with their KV
caches kept) and reused for every item (``Engine.locate``): the item's tokens go
through layers ``0..K-1`` against the cache, the screen is cut to the ``R``
blocks the pointing heads look at, layers ``K..`` run on that short sequence,
and the answer ``x, y`` is written in the model's 0-1000 coordinates.

The read does not depend on the item, so it can run while a planner is still
deciding what to click (``Pointer.prime``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_vlm import load
from mlx_vlm.models.cache import KVCache
from mlx_vlm.utils import prepare_inputs
from PIL import Image

from . import _capture

TRIGGER = "Locate the element: {item}. Output its click coordinate."
LAYERS_BEFORE_PRUNE = 24  # K: the pointing heads live in layers 17-23; the screen is pruned here
HEADS_USED = 32
LORA = {"rank": 16, "dropout": 0.0, "scale": 2.0}
BLOCK = 32  # pixels per image token (2x2 patches of 16)


def fit_pixels(size: tuple[int, int], pixels: int, enlarge: bool = True) -> tuple[int, int]:
    """Size close to ``pixels`` in area, both sides multiples of 32."""
    w, h = size
    s = math.sqrt(pixels / (w * h))
    if not enlarge:
        s = min(1.0, s)
    return max(BLOCK, round(w * s / BLOCK) * BLOCK), max(BLOCK, round(h * s / BLOCK) * BLOCK)


@dataclass
class Reading:
    """One screen, read: the item-independent part of pointing."""

    page: dict[str, Any]
    caches: list[KVCache]
    h: mx.array  # layer-K states of [header][screen][mid]
    pos: mx.array
    nb: int  # tokens before the screen
    nk: int  # screen blocks
    total: int
    offsets: list[int]
    tail: list[int]
    next_pos: int
    size: tuple[int, int] = (0, 0)  # the image as read (w, h)


class Engine:
    def __init__(self, model_dir: str | Path, adapters: str | Path | None, heads: np.ndarray,
                 prune_after: int = LAYERS_BEFORE_PRUNE, heads_used: int = HEADS_USED) -> None:
        _capture.install()
        self.model, self.processor = load(str(model_dir))
        vt = self.model.vision_tower
        vt.update(tree_map(lambda x: x.astype(mx.bfloat16) if x.dtype == mx.float32 else x, vt.parameters()))
        if adapters is not None:
            self._attach_adapters(Path(adapters))
        self.tok = self.processor.tokenizer
        self.cfg = self.model.config
        self.inner = self.model.language_model.model
        for i, layer in enumerate(self.inner.layers):
            layer.self_attn._li = i
        self.K = prune_after
        self.weights = self._head_weights(heads, heads_used)
        self.layers_with_heads = {int(i) for i in np.where(self.weights.sum(axis=1) > 0)[0]}
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\x00"}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        self.prefix_text = text.split("\x00")[0]
        # the prompt around an item, tokenised once: [header][image][mid] item [tail]
        tmpl = self._prompt("\x00")
        before, after = tmpl.split("<|image_pad|>", 1)
        after = after.replace("<|image_pad|>", "")
        self.header = self.tok.encode(before, add_special_tokens=False)
        # the item carries its own leading space
        self.mid = self.tok.encode(after.split("\x00")[0].rstrip(" "), add_special_tokens=False)
        self.tail = self.tok.encode(after.split("\x00")[1], add_special_tokens=False)
        comma, space = (self.tok.encode(t, add_special_tokens=False)[0] for t in (",", " "))
        self.comma, self.space = comma, space
        self.digits = {self.tok.encode(str(d), add_special_tokens=False)[0]: str(d) for d in range(10)}

    # -- construction ------------------------------------------------------------------------

    def _prompt(self, item: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TRIGGER.format(item=item)}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "("

    def _attach_adapters(self, path: Path) -> None:
        from mlx_vlm.trainer.utils import _to_lora, set_module_by_name

        file = path / "lora.safetensors" if path.is_dir() else path
        lw = mx.load(str(file))
        trained = {int(k.split(".layers.")[1].split(".")[0]) for k in lw}
        for li, layer in enumerate(self.model.language_model.model.layers):
            if li not in trained:
                continue
            for owner, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                 ("mlp", ("gate_proj", "up_proj", "down_proj"))):
                for nm in names:
                    set_module_by_name(layer, f"{owner}.{nm}", _to_lora(getattr(getattr(layer, owner), nm), LORA))
        self.model.load_weights(list(lw.items()), strict=False)
        mx.eval(self.model.parameters())

    def _head_weights(self, mass: np.ndarray, used: int) -> np.ndarray:
        """Normalised weights over the ``used`` strongest (layer, head) pairs below layer K."""
        K = self.K
        keep = np.zeros(mass[:K].size, dtype=bool)
        keep[np.argsort(-mass[:K].ravel())[:used]] = True
        w = np.zeros_like(mass)
        w[:K] = np.where(keep.reshape(mass[:K].shape), mass[:K], 0.0)
        return w / w.sum()

    # -- step 1: the screen ------------------------------------------------------------------

    def read(self, image: Image.Image) -> Reading:
        """Vision encoder and layers 0..K-1 over [header][screen][mid]. Item-independent."""
        inp = prepare_inputs(self.processor, images=[image], prompts=[self.prefix_text])
        px, grid = inp["pixel_values"], inp["image_grid_thw"]
        gh, gw = int(grid[0, 1]) // 2, int(grid[0, 2]) // 2
        feats, deep = self._vit(px, grid, np.arange(gh * gw))
        full = self.header + [self.cfg.image_token_index] * (gh * gw) + self.mid
        ids = mx.array([full])
        pos, _ = self.model.language_model.get_rope_index(ids, grid, None, None)
        nb = len(self.header)
        embeds = self.inner.embed_tokens(ids)
        idx = mx.arange(nb, nb + gh * gw).astype(mx.uint32)
        embeds = embeds.at[0, idx].add(feats - embeds[0, idx])
        rot = self.inner.layers[0].self_attn.rotary_emb
        pe = None if rot.fused_apply else rot(embeds, pos)
        caches = [KVCache() for _ in range(self.K)]
        h = embeds
        for i in range(self.K):
            h = self.inner.layers[i](h, "causal", caches[i], pos, pe)
            if i < len(deep):
                h = h.at[0, idx].add(deep[i])
        mx.eval(h, *[c.keys for c in caches])
        return Reading(page={"grid": (gh, gw), "kept": np.arange(gh * gw)}, caches=caches, h=h, pos=pos,
                       nb=nb, nk=gh * gw, total=h.shape[1], offsets=[c.offset for c in caches],
                       tail=self.tail, next_pos=int(pos.max().item()) + 1, size=image.size)

    def _vit(self, pixel_values: mx.array, grid: mx.array, keep_blocks: np.ndarray):
        """The vision encoder over the kept 32x32 blocks (rows of pixel_values are ordered block by block)."""
        vt = self.model.vision_tower
        rows = mx.array((keep_blocks[:, None] * 4 + np.arange(4)[None]).reshape(-1), dtype=mx.uint32)
        x = vt.patch_embed(pixel_values[rows].astype(vt.patch_embed.proj.weight.dtype))
        x = x + vt.fast_pos_embed_interpolate(grid)[rows]
        rot = vt.rot_pos_emb(grid)
        rot = rot.reshape(rot.shape[0], -1)[rows]
        n = x.shape[0]
        x = x.reshape(n, -1)
        cu = mx.array([0, n], dtype=mx.int32)
        deep = []
        for i, blk in enumerate(vt.blocks):
            x = blk(x, cu_seqlens=cu, rotary_pos_emb=rot)
            if i in vt.deepstack_visual_indexes:
                deep.append(vt.deepstack_merger_list[vt.deepstack_visual_indexes.index(i)](x))
        return vt.merger(x), deep

    # -- step 2: an item ---------------------------------------------------------------------

    def locate(self, reading: Reading, item: str, blocks: int = 64) -> tuple[float, float, float] | None:
        """(x, y, confidence) for ``item``, x and y as fractions 0-1 of the image; None if the
        model's answer holds no two numbers. Confidence is the lowest probability among the
        leading two digits of x and of y."""
        K, layers = self.K, self.inner.layers
        rows = self.tok.encode(" " + item, add_special_tokens=False) + reading.tail
        n = len(rows)
        rot = layers[0].self_attn.rotary_emb
        pos_i = mx.broadcast_to((reading.next_pos + mx.arange(n))[None, None, :], (3, 1, n))
        h = self.inner.embed_tokens(mx.array([rows]))
        pe = None if rot.fused_apply else rot(h, pos_i)
        for c, o in zip(reading.caches, reading.offsets, strict=True):
            c.offset = o  # forget any previous item on this screen
        _capture.STATE.on, _capture.STATE.layers, _capture.STATE.queries = True, self.layers_with_heads, {}
        try:
            for i in range(K):
                h = layers[i](h, "causal", reading.caches[i], pos_i, pe)
        finally:
            _capture.STATE.on = False
        score = self._screen_scores(reading)
        r = min(blocks, reading.nk)
        top = mx.sort(mx.argpartition(-score, r - 1)[:r])
        nb = reading.nb
        keep = mx.concatenate([mx.arange(nb), nb + top, mx.arange(nb + reading.nk, reading.total)]).astype(mx.uint32)
        hk = mx.concatenate([reading.h[:, keep], h], axis=1)
        pk = mx.concatenate([reading.pos[:, :, keep], pos_i], axis=2)
        pek = None if rot.fused_apply else rot(hk, pk)
        late = [KVCache() for _ in range(K, len(layers))]
        for j, i in enumerate(range(K, len(layers))):
            hk = layers[i](hk, "causal", late[j], pk, pek)
        return self._write_answer(reading, hk, late, reading.next_pos + n)

    def _screen_scores(self, reading: Reading) -> mx.array:
        """The pointing heads' attention from the item's last token over the screen blocks,
        recomputed from the cached keys (the hook only sees the new tokens)."""
        out = 0
        for li in sorted(self.layers_with_heads):
            attn = self.inner.layers[li].self_attn
            q = _capture.STATE.queries[li]  # (heads, d)
            keys = reading.caches[li].keys[0, :, : reading.total, :]  # (kv_heads, P, d)
            k = mx.repeat(keys[:, reading.nb: reading.nb + reading.nk], attn.n_heads // attn.n_kv_heads, axis=0)
            s = (k @ q[:, :, None])[..., 0] * attn.scale  # (heads, nk)
            out = out + (mx.softmax(s.astype(mx.float32), axis=-1) * mx.array(self.weights[li][:, None])).sum(axis=0)
        return out

    def _run(self, toks: list[int], reading: Reading, late: list[KVCache], pos0: int) -> mx.array:
        layers = self.inner.layers
        rot = layers[0].self_attn.rotary_emb
        h = self.inner.embed_tokens(mx.array([toks]))
        n = len(toks)
        p = mx.broadcast_to((pos0 + mx.arange(n))[None, None, :], (3, 1, n))
        pe = None if rot.fused_apply else rot(h, p)
        mask = None if n == 1 else "causal"
        for i, layer in enumerate(layers):
            h = layer(h, mask, reading.caches[i] if i < self.K else late[i - self.K], p, pe)
        return h

    def _pick(self, h_last: mx.array) -> tuple[int, float]:
        lm = self.model.language_model
        hs = self.inner.norm(h_last)
        logits = self.inner.embed_tokens.as_linear(hs) if lm.args.tie_word_embeddings else lm.lm_head(hs)
        p = mx.softmax(logits[0, -1].astype(mx.float32))
        t = int(mx.argmax(p).item())
        return t, float(p[t].item())

    def _write_answer(self, reading: Reading, hk: mx.array, late: list[KVCache], pos: int):
        """Greedy-write "xxx, yyy)" with the predictable steps merged: after x's third digit the
        ", " is fed with it, and the answer stops after y's third digit."""
        tok, pr = self._pick(hk[:, -1:])
        x = y = ""
        in_y = False
        probs: list[float] = []
        for _ in range(10):
            if tok in self.digits:
                if (len(y) if in_y else len(x)) < 2:
                    probs.append(pr)
                if in_y:
                    y += self.digits[tok]
                    if len(y) == 3:
                        break
                    feed = [tok]
                else:
                    x += self.digits[tok]
                    feed = [tok, self.comma, self.space] if len(x) == 3 else [tok]
                    if len(x) == 3:
                        in_y = True
            elif tok == self.comma and not in_y:
                in_y = True
                feed = [self.comma, self.space]
            else:
                break  # ")" or anything unexpected ends the answer
            hs = self._run(feed, reading, late, pos)
            pos += len(feed)
            tok, pr = self._pick(hs[:, -1:])
        if not x or not y:
            return None
        return int(x) / 1000, int(y) / 1000, (min(probs) if probs else 0.0)
