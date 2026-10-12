"""The pointer: a screenshot and the name of an item in, a click position out.

    from anypointer import Pointer

    pointer = Pointer()                      # loads lazily, on first use
    pointer.prime("screen.png")              # optional: do the screen's read while a planner thinks
    p = pointer.point("screen.png", "the wifi toggle")
    p.x, p.y, p.confidence

One screen is held at a time. Asking about the same pixels again reuses the read, and so does
a different item on that screen, so only the first item pays for the screen.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from . import grid as G
from .ocr import OcrBackend, default_backend
from .types import Point

log = logging.getLogger("anypointer")

DEFAULT_PIXELS = 409_600  # the read size (~0.4 MP): the speed/accuracy setting
RESELECT = 64  # screen blocks kept after the pointing heads' pick


def load_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as im:
            return im.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if hasattr(image, "read"):
        return Image.open(image).convert("RGB")
    raise TypeError(f"cannot read an image from {type(image).__name__}")


class Pointer:
    """
    Args:
        model: local directory or Hugging Face repo id of GUI-Owl-1.5-4B. Default: the original
            MIT-licensed weights, converted to MLX 4-bit on first use (see ``weights``).
        adapters: optional ``lora.safetensors`` (file, directory or repo id).
        pixels: area the screenshot is read at. Larger reads small icons better and costs time.
        grid: snap coded references ("C27", "e4") to cells of a measured grid.
        ocr: label reader for grid snapping; default Apple Vision when available.
    """

    def __init__(self, model: str | os.PathLike[str] | None = None, adapters: str | os.PathLike[str] | None = None,
                 pixels: int = DEFAULT_PIXELS, grid: bool = True, ocr: OcrBackend | None = None) -> None:
        self._model_spec, self._adapter_spec = model, adapters
        self.pixels = pixels
        self.grid = grid
        self._ocr_backend = ocr if ocr is not None else (default_backend() if grid else None)
        self._engine: Any = None
        self._lock = threading.RLock()
        self._key: str | None = None
        self._screen: dict[str, Any] = {}
        self._ocr_cache: dict[Any, list] = {}
        self._grids: list = []

    # -- loading -----------------------------------------------------------------------------

    def load(self) -> Pointer:
        """Load the model now (otherwise the first call does)."""
        with self._lock:
            if self._engine is None:
                import numpy as _np

                from . import weights
                from .engine import Engine

                model_dir = weights.resolve_model(self._model_spec)
                adapters = weights.resolve_adapters(self._adapter_spec)
                heads = _np.load(weights.heads_file())["mass"]
                self._engine = Engine(model_dir, adapters, heads)
        return self

    # -- public ------------------------------------------------------------------------------

    def prime(self, image: Any) -> None:
        """Do an image's item-independent work (the screen read, and the grid search with its
        labels) so the first item only pays for itself. Safe to call and then ``point``; call it
        as soon as the screenshot exists, while a planner still decides what to click."""
        im = load_image(image)
        with self._lock:
            self.load()
            key = self._digest(im)
            if key == self._key:
                return
            self._new_screen(key)
            th = None
            if self.grid and self._ocr_backend is not None:
                # grid search and OCR run on the CPU / Neural Engine: do them while the GPU reads
                th = threading.Thread(target=self._find_grids, args=(im, key), daemon=True)
                th.start()
            self._read(im)
            if th is not None:
                th.join()

    def point(self, image: Any, item: str) -> Point:
        """Where to click for ``item`` ("the wifi toggle", "cell C27", "square e4")."""
        t0 = time.perf_counter()
        im = load_image(image)
        with self._lock:
            self.load()
            key = self._digest(im)
            if key != self._key:
                self._new_screen(key)
                self._read(im)
            W, H = im.size
            got = self._engine.locate(self._screen["reading"], item, RESELECT)
            if got is None:  # no coordinates in the model's answer: say so rather than guess
                return Point(W / 2, H / 2, 0.0, "model", "model (no answer)", (W / 2, H / 2), (W, H),
                             1000 * (time.perf_counter() - t0))
            fx, fy, conf = got
            px, py = fx * W, fy * H
            how, x, y = "model", px, py
            if self.grid and self._ocr_backend is not None:
                s = self._snap(im, item, px, py)
                x, y, how = s.x, s.y, s.how
            return Point(x, y, conf, "grid" if how.startswith("grid") else "model", how, (px, py), (W, H),
                         1000 * (time.perf_counter() - t0))

    # -- screen state ------------------------------------------------------------------------

    def _digest(self, im: Image.Image) -> str:
        return f"{hashlib.sha1(im.tobytes()).hexdigest()}-{im.size}-{self.pixels}"

    def _new_screen(self, key: str) -> None:
        import mlx.core as mx

        self._key, self._screen, self._ocr_cache, self._grids = key, {}, {}, []
        mx.clear_cache()  # the previous screen's caches are large

    def _read(self, im: Image.Image) -> None:
        from .engine import fit_pixels

        rw, rh = fit_pixels(im.size, self.pixels)
        small = im.resize((rw, rh), Image.LANCZOS)
        self._screen["reading"] = self._engine.read(small)

    # -- grids -------------------------------------------------------------------------------

    def _ocr(self, crop: Image.Image) -> list:
        """OCR on a label strip, enlarged and padded (single small labels are otherwise skipped);
        words in crop coordinates. Cached per screen."""
        key = (crop.size, hashlib.sha1(crop.convert("RGB").tobytes()).hexdigest())
        if key not in self._ocr_cache:
            words: list = []
            for sc in (3, 5):
                big = crop.convert("RGB").resize((crop.width * sc, crop.height * sc), Image.LANCZOS)
                pad = Image.new("RGB", (big.width + 40, big.height + 40), (255, 255, 255))
                pad.paste(big, (20, 20))
                got = [(t, ((x0 - 20) / sc, (y0 - 20) / sc, (x1 - 20) / sc, (y1 - 20) / sc))
                       for t, (x0, y0, x1, y1) in self._ocr_backend(pad)]
                if len(got) > len(words):
                    words = got
                if len(words) >= 4:
                    break
            self._ocr_cache[key] = words
        return self._ocr_cache[key]

    def _snap(self, im: Image.Image, item: str, px: float, py: float) -> G.Snap:
        coded = G.parse_ref(item) is not None
        known = next((g for g in self._grids if g[0][0] <= px <= g[0][-1] and g[1][0] <= py <= g[1][-1]), None)
        if known is not None and coded:
            # a grid seen from this point may cover more of the same table than the cached one
            # (a popup or the window around an earlier point can cut a grid short): keep the larger
            fresh = G.find_grid(im, px, py)
            area = lambda g: (g[0][-1] - g[0][0]) * (g[1][-1] - g[1][0])
            if fresh is not None and area(fresh) > 1.1 * area(known):
                self._grids = [g for g in self._grids if g is not known] + [fresh]
                known = fresh
        if known is None and self._grids and coded:
            # a coded reference ("seat B16") is answered from the labels, not from where the model
            # pointed: when the point misses every grid, use the nearest one found in prep
            def gap(g):
                dx = max(g[0][0] - px, 0, px - g[0][-1])
                dy = max(g[1][0] - py, 0, py - g[1][-1])
                return dx * dx + dy * dy
            known = min(self._grids, key=gap)
        s = G.snap(im, item, px, py, self._ocr, grid=known)
        if known is None and s.grid is not None:
            self._grids.append(s.grid)  # the same grid (and its label reads) for the next items
        return s

    def _find_grids(self, im: Image.Image, key: str) -> None:
        """Find the main grid on a screen and read its labels now, as a question about a cell would."""
        try:
            W, H = im.size
            found = []
            for fx in (0.2, 0.4, 0.6, 0.8):
                for fy in (0.25, 0.5, 0.75):
                    x, y = fx * W, fy * H
                    if any(g[0][0] <= x <= g[0][-1] and g[1][0] <= y <= g[1][-1] for g in found):
                        continue
                    g = G.find_grid(im, x, y)
                    # a real grid of cells: at least ~20 x 16 px (finer "grids" are lines of text)
                    if g is None or (len(g[0]) - 1) * (len(g[1]) - 1) < 12 or \
                            np.median(np.diff(g[0])) < 20 or np.median(np.diff(g[1])) < 16:
                        continue
                    found.append(g)
            if found and key == self._key:  # only the main grid is prepared ahead
                g = max(found, key=lambda gr: (gr[0][-1] - gr[0][0]) * (gr[1][-1] - gr[1][0]))
                self._grids.append(g)
                G.snap(im, "A1", (g[0][0] + g[0][1]) / 2, (g[1][0] + g[1][1]) / 2, self._ocr, grid=g)
        except Exception:  # prep is an optimisation: a failure must not break pointing
            log.exception("grid preparation failed")


__all__ = ["Point", "Pointer", "load_image"]
