# How it works

## 1. Read the screen once

GUI-Owl reads `[screen]["Locate the element: "][item]`. Everything before the item is the same
for any item on that screen, so `Engine.read` computes it once: the vision encoder and language
layers 0-23, keeping their KV caches. The screenshot is resized to about 0.4 megapixels (sides in
multiples of 32; one image token per 32x32 block).

## 2. Answer an item

`Engine.locate` runs only the item's tokens through layers 0-23 against the cache. A fixed set of
32 attention heads (layers 17-23, chosen once on training pages and shipped in `data/`) score every
screen block by how much the item's last token attends to it. Only the 64 best blocks are kept for
layers 24-35, which then write the coordinate as `x, y` in the model's 0-1000 space. The answer is
written greedily with predictable tokens merged (the `", "` after x, the stop after y's third
digit). Confidence is the lowest probability among the first two digits of x and of y.

## 3. Snap to the grid (coded references only)

When the item is `e4`, `C27` or `seat B16` the model's job is only to say *which grid*; the cell is
arithmetic:

1. **Find the grid.** Around the model's point, edge energy is summed along each axis, its
   autocorrelation gives the band period (lines, alternating fills, stripes and repeated items all
   repeat), the best-aligned phase gives the boundaries, and the bands are followed across the
   image while the edges keep repeating. Several windows are tried so a popup does not hide the grid.
2. **Read the labels** on the strips outside the grid and in its edge cells with OCR. The +-1
   sequence most labels agree on says which band holds the wanted label; ordered labels may run
   either way (flipped boards, scrolled sheets) and letters/numbers may label either axis (seat maps).
3. **Fall back** to the usual board convention for an 8x8 board with unreadable coordinates, else
   the band the model pointed at.

`Pointer.prime` runs the grid search and OCR on the CPU/Neural Engine while the GPU does the read.

## Numbers during development

On M4 / 16 GB, GUI-Owl-1.5-4B 4-bit at 0.4 MP, 64 blocks:

| | |
|---|---|
| screen read | ~1.2 s once per screenshot (can overlap the planner) |
| item to click | ~0.3 s |
| generated chess squares (180), with snapping | 179/180 (model alone: ~55%) |
| generated sheet cells (200), with snapping | 198-200/200 (model alone: ~46%) |
| ScreenSpot-v2 icons, 0.4 MP | ~83% on a 277-item sample; 85% with a 0.8 MP re-read when unsure |

Treat these as development measurements on one machine, not a published benchmark.
