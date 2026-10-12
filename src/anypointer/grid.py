"""Exact cells in any grid: find the grid by its repetition, read its labels, count.

A pointing model lands near the right cell of a chess board or a spreadsheet and is
often one row off: rows are thinner than the 32 px block it sees, and a label like
"27" is hard to tell from "7". For coded references ("C27", "e4", "seat B16") the
cell is not a matter of seeing, it is arithmetic on the grid, so:

  1. Grid: around the model's point, the image's edge energy is summed along each axis;
     its autocorrelation gives the band period (lines, alternating fills, stripes and
     repeated items all repeat), the phase that lines up with the strongest edges gives
     the band boundaries, and the bands are extended outward while the edges keep
     repeating.
  2. Labels: strips just outside the grid (left, right, top, bottom) and the outermost
     bands inside it (labels drawn inside edge cells) are read with OCR; the +-1 sequence
     most labels agree on tells which band holds the wanted label.
  3. Fallback: the usual chess convention for an 8x8 board without readable coordinates,
     else the band the model pointed at.

    parse_ref(item) -> (col_label, row_label) or None
    find_grid(image, px, py) -> (column boundaries, row boundaries) or None
    snap(image, item, px, py, ocr) -> Snap
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

REFS = [
    re.compile(r"^\s*(?:the\s+)?(?:cell\s+|square\s+|seat\s+)?([A-Za-z]{1,3})\s*(\d{1,6})(?:\s+(?:cell|square|seat))?\s*$", re.I),
]
ROWCOL = re.compile(r"^\s*(?:the\s+)?(?:cell|option|box|item)?\s*(?:in|at)?\s*row\s+['\"]?(.+?)['\"]?\s*,?\s*(?:and\s+)?col(?:umn)?\s+['\"]?(.+?)['\"]?\s*$", re.I)



def parse_ref(item: str):
    """(column label, row label) for 'C27', 'e4', 'square e4', 'row 3, column Good'; else None."""
    for rx in REFS:
        m = rx.match(item)
        if m:
            return m.group(1), m.group(2)
    m = ROWCOL.match(item)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    return None


def _edge_profiles(g: np.ndarray):
    gx = np.abs(np.diff(g, axis=1))  # vertical edges -> column boundaries
    gy = np.abs(np.diff(g, axis=0))  # horizontal edges -> row boundaries
    return gx.mean(axis=0), gy.mean(axis=1)


def _period(p: np.ndarray, lo: int = 12, hi: int = 260):
    """Band period: the shortest lag whose autocorrelation peak is nearly as strong
    as the strongest (a checkerboard also repeats every 2 squares; we want 1)."""
    p = p - p.mean()
    if len(p) < 3 * lo:
        return None
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(p) // 3)
    if hi <= lo + 2:
        return None
    seg = ac[lo:hi]
    peaks = [lo + i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None
    top = max(ac[k] for k in peaks)
    if top < 0.2:
        return None
    return min(k for k in peaks if ac[k] >= 0.6 * top)


def _bands(p: np.ndarray, period: int, near: int, min_bands: int = 3):
    """Boundary positions: the phase whose comb hits the strongest edges, extended
    from ``near`` while boundary edges stay strong."""
    phase = max(range(period), key=lambda ph: p[ph::period].sum())
    pos = np.arange(phase, len(p), period)
    # refine each boundary to the local edge maximum (+-8% of the period: real grid lines
    # repeat almost exactly, a page bar or a text column a little further away does not)
    w = max(2, round(period * 0.08))
    ref = []
    for x in pos:
        a, b = max(0, x - w), min(len(p), x + w + 1)
        ref.append(a + int(np.argmax(p[a:b])))
    ref = np.array(ref)
    strength = p[ref]
    thr = 0.35 * np.median(strength[strength > 0]) if (strength > 0).any() else 0
    k = int(np.searchsorted(ref, near))
    lo = hi = min(max(k, 0), len(ref) - 1)
    while lo - 1 >= 0 and strength[lo - 1] > thr:
        lo -= 1
    while hi + 1 < len(ref) and strength[hi + 1] > thr:
        hi += 1
    out = ref[lo: hi + 1]
    return out if len(out) >= min_bands + 1 else None


def _extend(full: np.ndarray, start: list[float], period: int, thr: float) -> list[float]:
    """Follow evenly spaced boundaries outward across the whole image while the
    edges keep repeating (each step re-centred on the local edge maximum)."""
    w = max(2, round(period * 0.08))  # real grid lines repeat almost exactly; page edges and text do not
    out = list(start)

    def step(x):
        a, b = int(max(0, x - w)), int(min(len(full), x + w + 1))
        if b - a < 2:
            return None
        i = a + int(np.argmax(full[a:b]))
        return float(i) if full[i] > thr else None

    while True:
        nxt = step(out[-1] + period)
        if nxt is None or nxt <= out[-1]:
            break
        out.append(nxt)
    while True:
        prv = step(out[0] - period)
        if prv is None or prv >= out[0]:
            break
        out.insert(0, prv)
    # tiles with gaps between them (seat maps): the comb locks onto one side of every tile, so
    # the outermost tile's own edge sits a gap away from where the next boundary would be.
    # One looser step at each end, placed at the expected position; _trim drops the band
    # again if the grid's crossing lines do not run through it
    loose = max(3, round(period * 0.3))
    for end, sign in ((-1, 1), (0, -1)):
        x = out[end] + sign * period
        a, b = int(max(0, x - loose)), int(min(len(full), x + loose + 1))
        if b - a >= 2 and 0 <= x < len(full) and full[a:b].max() > thr:
            out.insert(len(out) if end == -1 else 0, float(x))
    return out


def _crossing(g: np.ndarray, lines: list[float], a: float, b: float, axis: int) -> float:
    """How strongly the boundaries ``lines`` (rows if axis=0, columns if 1) run through
    the band [a, b] across them: mean edge coverage at those boundaries within the band."""
    a, b = int(a) + 2, int(b) - 2
    if b - a < 3:
        return 0.0
    vals = []
    for v in lines:
        v = round(v)
        if axis == 0 and 1 <= v < g.shape[0] - 1:
            seg = np.abs(g[v + 1, a:b] - g[v - 1, a:b])
        elif axis == 1 and 1 <= v < g.shape[1] - 1:
            seg = np.abs(g[a:b, v + 1] - g[a:b, v - 1])
        else:
            continue
        vals.append(float((seg > 6).mean()))
    return float(np.mean(vals)) if vals else 0.0


def _trim(g: np.ndarray, cols: list[float], rows: list[float]):
    """Drop outer bands the grid's crossing lines do not run through (a text column or a
    page bar that happens to sit one period away is not part of the grid)."""
    cols, rows = list(cols), list(rows)
    for _ in range(4):
        changed = False
        if len(cols) > 4:
            inner = np.median([_crossing(g, rows, cols[i], cols[i + 1], 0) for i in range(1, len(cols) - 2)])
            if _crossing(g, rows, cols[0], cols[1], 0) < 0.5 * inner:
                cols.pop(0)
                changed = True
            if _crossing(g, rows, cols[-2], cols[-1], 0) < 0.5 * inner:
                cols.pop()
                changed = True
        if len(rows) > 4:
            inner = np.median([_crossing(g, cols, rows[i], rows[i + 1], 1) for i in range(1, len(rows) - 2)])
            if _crossing(g, cols, rows[0], rows[1], 1) < 0.5 * inner:
                rows.pop(0)
                changed = True
            if _crossing(g, cols, rows[-2], rows[-1], 1) < 0.5 * inner:
                rows.pop()
                changed = True
        if not changed:
            break
    return cols, rows


def find_grid(image: Image.Image, px: float, py: float):
    """(column boundaries, row boundaries) of a grid around (px, py), or None. The
    spacing is measured in a window around the point (several windows, so a popup
    covering part of the grid does not hide it), then followed across the image."""
    g = np.asarray(image.convert("L"), dtype=np.float32)
    H, W = g.shape
    clip = lambda a, b, n: (int(max(0, a)), int(min(n, b)))
    found = []
    for (ax, bx), (ay, by) in (((px - 500, px + 500), (py - 500, py + 500)),
                               ((px - 250, px + 250), (py - 250, py + 250)),
                               ((px - 700, px + 60), (py - 60, py + 700)),
                               ((px - 700, px + 60), (py - 700, py + 60)),
                               ((px - 60, px + 700), (py - 60, py + 700))):
        x0, x1 = clip(ax, bx, W)
        y0, y1 = clip(ay, by, H)
        if x1 - x0 < 80 or y1 - y0 < 80:
            continue
        prof_x, prof_y = _edge_profiles(g[y0:y1, x0:x1])
        per_x, per_y = _period(prof_x), _period(prof_y)
        if per_x is None or per_y is None:
            continue
        cols = _bands(prof_x, per_x, int(px - x0))
        rows = _bands(prof_y, per_y, int(py - y0))
        if cols is None or rows is None:
            continue
        cols = [c + x0 for c in cols]
        rows = [r + y0 for r in rows]
        # follow the grid beyond the window: column edges measured over the grid's rows, and vice versa
        ry0, ry1 = clip(rows[0], rows[-1], H)
        cx0, cx1 = clip(cols[0], cols[-1], W)
        # coverage: the share of the grid's length along which an edge is present; a real
        # grid boundary runs along nearly all of it, label text or a frame does not
        fx = (np.abs(np.diff(g[ry0:ry1, :], axis=1)) > 6).mean(axis=0)
        fy = (np.abs(np.diff(g[:, cx0:cx1], axis=0)) > 6).mean(axis=1)
        tx = 0.6 * float(np.median(fx[np.array(cols, dtype=int)]))
        ty = 0.6 * float(np.median(fy[np.array(rows, dtype=int)]))
        cols = _extend(fx, [float(c) for c in cols], per_x, tx)
        rows = _extend(fy, [float(r) for r in rows], per_y, ty)
        cols, rows = _trim(g, cols, rows)
        if len(cols) < 4 or len(rows) < 4:
            continue
        cand = ([c + 0.5 for c in cols], [r + 0.5 for r in rows])
        if cand[0][0] <= px <= cand[0][-1] and cand[1][0] <= py <= cand[1][-1]:
            found.append(cand)
    # the largest grid any window found (a popup or side panel can truncate a window's view), by
    # area: texture inside a picture can repeat finely too, but only ever over a small area
    return max(found, key=lambda gr: (gr[0][-1] - gr[0][0]) * (gr[1][-1] - gr[1][0])) if found else None


def seq_value(label: str):
    """Position of an ordered label: numbers as themselves, letters as spreadsheet
    columns (A=1 .. Z=26, AA=27; chess files a=1 .. h=8); else None (free text)."""
    t = label.strip()
    if re.fullmatch(r"\d{1,6}", t):
        return int(t)
    if re.fullmatch(r"[A-Za-z]{1,3}", t):
        v = 0
        for ch in t.upper():
            v = v * 26 + (ord(ch) - 64)
        return v
    return None


def _band_of(v: float, edges: list[float]):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return None


OFF = -1  # a label axis that was read, but the wanted label lies outside the grid


OcrFn = Callable[[Image.Image], list[tuple[str, tuple[float, float, float, float]]]]


@dataclass
class Snap:
    x: float
    y: float
    how: str  # "grid", "grid (...)" when labels were partly guessed, "model (...)" when not snapped
    grid: tuple[list[float], list[float]] | None = None  # the grid used, for reuse on the next item

    @property
    def snapped(self) -> bool:
        return self.how.startswith("grid")


def snap(image: Image.Image, item: str, px: float, py: float, ocr: OcrFn | None,
         grid: tuple[list[float], list[float]] | None = None) -> Snap:
    """The centre of the cell ``item`` names ("C27", "e4", "seat B16"), or the model's own point
    (px, py) when the item is not a coded reference or the grid's labels cannot be read.
    ``ocr(image) -> [(text, (x0, y0, x1, y1))]`` reads text; without it only the chess
    convention and the model's band are available."""
    ref = parse_ref(item)
    if ref is None:
        return Snap(px, py, "model")
    # only coded references need the grid: cells named by words ("row Customer service,
    # column Agree") the model finds by reading them, and better
    if seq_value(ref[0]) is None or seq_value(ref[1]) is None:
        return Snap(px, py, "model (labels are words)")
    grid = grid or find_grid(image, px, py)
    if grid is None:
        return Snap(px, py, "model (no grid)")
    cols, rows = grid
    W, H = image.size
    cw, rh = float(np.median(np.diff(cols))), float(np.median(np.diff(rows)))
    col_label, row_label = ref
    steps: dict[str, int] = {}  # per axis: +1 if labels count up along the band order, -1 if down

    def crop(box):
        x0, y0, x1, y1 = (round(v) for v in box)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        return None if (x1 - x0 < 6 or y1 - y0 < 6) else ((x0, y0), image.crop((x0, y0, x1, y1)))

    def has_ink(img) -> bool:
        """Anything written in this crop? (a blank strip needs no reading)"""
        a = np.asarray(img.convert("L"), dtype=np.float32)
        if a.size == 0:
            return False
        a = a[1:-1, 1:-1] if min(a.shape) > 4 else a  # ignore grid lines on the crop's border
        return float((np.abs(a - np.median(a)) > 40).mean()) > 0.004

    def strips(axis):
        """Where an axis's labels may sit: (label-cell box for band i, whole strip box)."""
        if axis == "row":
            n = len(rows) - 1
            return [(lambda i, a=a, b=b: (a, rows[i], b, rows[i + 1]), (a, rows[0], b, rows[-1]))
                    for a, b in ((cols[0] - 1.6 * cw, cols[0]), (cols[-1], cols[-1] + 1.6 * cw),
                                 (cols[0], cols[1]), (cols[-2], cols[-1]))], n
        n = len(cols) - 1
        return [(lambda i, a=a, b=b: (cols[i], a, cols[i + 1], b), (cols[0], a, cols[-1], b))
                for a, b in ((rows[0] - 1.6 * rh, rows[0]), (rows[-1], rows[-1] + 1.6 * rh),
                             (rows[-2], rows[-1]), (rows[0], rows[1]))], n

    def ocr_tiles(tiles, kind, per_line=12, gap=14):
        """OCR on crops laid side by side like words on a line (text recognisers read lines of
        text; labels stacked in a column, or scattered among pieces, are easily missed):
        [(tile index, value)] for single ordered labels of the wanted kind."""
        tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
        lines = (len(tiles) + per_line - 1) // per_line
        sheet = Image.new("RGB", (min(len(tiles), per_line) * (tw + gap) + gap, lines * (th + 2 * gap) + gap),
                          (255, 255, 255))
        bg = None
        for i, t in enumerate(tiles):
            t = t.convert("RGB")
            if bg is None:  # dark screens: lay the tiles on their own background colour
                a = np.asarray(t).reshape(-1, 3)
                bg = tuple(int(v) for v in np.median(a, axis=0))
                sheet.paste(bg, (0, 0, sheet.width, sheet.height))
            sheet.paste(t, (gap + (i % per_line) * (tw + gap), gap + (i // per_line) * (th + 2 * gap)))
        out = []
        for text, (x0, y0, x1, y1) in ocr(sheet):
            t = text.strip().strip(".:")
            if not t or t.isdigit() != kind or (not kind and len(t) > 2):
                continue
            col = int(((x0 + x1) / 2 - gap) // (tw + gap))
            line = int(((y0 + y1) / 2 - gap) // (th + 2 * gap))
            i = line * per_line + col
            v = seq_value(t.lower() if not kind else t)
            if v is not None and 0 <= col < per_line and 0 <= i < len(tiles):
                out.append((i, v))
        return out

    def by_ocr(axis, label):
        """The band of ``label`` on an axis, from labels read by OCR on the edge strips, each placed
        in its band; the +-1-per-band sequence most labels agree on wins. None: not read;
        OFF: read, but the wanted label is not on this grid."""
        if ocr is None:
            return None
        want = seq_value(label)
        if want is None:
            return None
        kind = label.strip().isdigit()
        cells, n = strips(axis)
        edges = rows if axis == "row" else cols
        best = None

        def fit(pts):
            """The band -> value line (step +-1) with the most labels on it."""
            nonlocal best
            for step in (1, -1):
                offs = {}
                for b, v in pts:
                    offs.setdefault(v - step * b, set()).add(b)
                for off, bands in offs.items():
                    if len(bands) >= 2 and (best is None or len(bands) > best[0]):
                        best = (len(bands), step, off)

        def finish():
            if best is None or best[0] < 3:  # three labels on one line; two can line up by chance
                return None
            _, step, off = best
            steps[axis] = step
            k = (want - off) * step  # value = off + step * band
            return k if 0 <= k < n else OFF

        for _, box in cells:
            c = crop(box)
            if c is None:
                continue
            (ox, oy), img = c
            pts = []
            for text, (x0, y0, x1, y1) in ocr(img):
                t = text.strip().strip(".:")
                if t.isdigit() != kind:
                    continue
                v = seq_value(t.lower() if not kind else t)
                if v is None:
                    continue
                centre = (oy + (y0 + y1) / 2) if axis == "row" else (ox + (x0 + x1) / 2)
                b = _band_of(centre, edges)
                if b is not None:
                    pts.append((b, v))
            fit(pts)
        if best is None or best[0] < 3:
            for cell_box, _ in cells:
                tiles = [crop(cell_box(i)) for i in range(n)]
                if any(t is None for t in tiles) or sum(has_ink(t[1]) for t in tiles) < 3:
                    continue
                fit(ocr_tiles([t[1] for t in tiles], kind))
                if best is not None and best[0] >= 3:
                    break
        if best is not None and best[0] >= 3:
            return finish()  # found on the edge strips; no need to look inside the cells
        # labels drawn inside the edge cells (e.g. lichess boards): the same corner of every
        # edge cell, cut out and lined up in one strip, so the label is not lost among pieces
        for inside in (0, 1):
            for corner in ("tl", "tr", "bl", "br"):
                tiles = []
                for i in range(n):
                    if axis == "row":
                        cx0, cx1 = (cols[0], cols[1]) if inside == 0 else (cols[-2], cols[-1])
                        cy0, cy1 = rows[i], rows[i + 1]
                    else:
                        cx0, cx1 = cols[i], cols[i + 1]
                        cy0, cy1 = (rows[-2], rows[-1]) if inside == 0 else (rows[0], rows[1])
                    w_, h_ = (cx1 - cx0) * 0.4, (cy1 - cy0) * 0.4
                    x0 = cx0 if corner[1] == "l" else cx1 - w_
                    y0 = cy0 if corner[0] == "t" else cy1 - h_
                    c = crop((x0, y0, x0 + w_, y0 + h_))
                    tiles.append(None if c is None else c[1])
                if any(t is None for t in tiles) or sum(has_ink(t) for t in tiles) < 3:
                    continue
                fit(ocr_tiles(tiles, kind))
        return finish()

    # OCR both ways round (cheap): "C12" is column C, row 12 on a spreadsheet but row C,
    # seat 12 on a seat map, so unless both labels read on the assumed axes, try them swapped
    r, c = by_ocr("row", row_label), by_ocr("col", col_label)
    if r is None and c is None:
        kept = dict(steps)
        r2, c2 = by_ocr("row", col_label), by_ocr("col", row_label)
        if OFF in (r2, c2) and None not in (r2, c2):
            return Snap(px, py, "model (cell not on the grid)", grid)
        if r2 is not None and c2 is not None:
            r, c = r2, c2
        else:
            steps.clear()
            steps.update(kept)
    if OFF in (r, c) and None not in (r, c):
        return Snap(px, py, "model (cell not on the grid)", grid)
    r = None if r is OFF else r
    c = None if c is OFF else c
    how = "grid"
    # an 8x8 board whose coordinates cannot be read: the usual convention, white at the bottom
    # (files a..h left to right, rank 8 at the top)
    chess = (len(cols) - 1 == 8 and len(rows) - 1 == 8 and re.fullmatch(r"[a-hA-H]", col_label or "")
             and re.fullmatch(r"[1-8]", row_label or ""))
    # a board turns as a whole: if the readable axis shows it flipped (black at the bottom:
    # ranks counting up downwards, or files counting down left to right), so is the other
    flipped = steps.get("row") == 1 or steps.get("col") == -1
    if chess and r is None:
        r, how = (int(row_label) - 1 if flipped else 8 - int(row_label)), "grid (rank by board convention)"
    if chess and c is None:
        f = "abcdefgh".index(col_label.lower())
        c, how = (7 - f if flipped else f), "grid (file by board convention)"
    if r is None:
        r, how = _band_of(py, rows), "grid (row from the model)"
    if c is None:
        c, how = _band_of(px, cols), "grid (column from the model)"
    if r is None or c is None:
        return Snap(px, py, "model (outside the grid)", grid)
    return Snap((cols[c] + cols[c + 1]) / 2, (rows[r] + rows[r + 1]) / 2, how, grid)
