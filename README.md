# anypointer

Name a thing on a screenshot, get the pixel to click.

```python
from anypointer import Pointer

pointer = Pointer()                          # first use downloads + converts the model (one time)
p = pointer.point("screen.png", "the wifi toggle")
print(p.x, p.y, p.confidence)               # pixels of screen.png
```

`anypointer` is the pointing half of a GUI agent: something else (a planner, an LLM, you)
decides *what* to act on, this finds *where* it is. It runs on your machine, on the GPU of an
Apple-silicon Mac, with no API calls.

- **Item in, point out.** Ask for the item ("the wifi toggle", "Add to cart under Wool rug"),
  not an instruction ("turn off wifi"). Resolving intent into an item is the planner's job.
- **Fast.** A screenshot is read once (about 1.2 s on an M4 Mac); every further item on that screenshot
  answers in about 0.3 s. `prime()` does the read ahead of time, while whatever decides what to click is
  still thinking.
- **Exact on grids.** Chess boards, spreadsheets, seat maps and survey matrices addressed by code
  (`e4`, `C27`, `seat B16`) are found by measuring the grid and reading its labels, not by guessing.
- **A confidence you can use.** 0-1, from the model's own certainty in the digits it writes.

Under the hood: [GUI-Owl-1.5-4B](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct) (MIT),
4-bit on MLX, read at ~0.4 megapixels, with its pointing attention heads used to keep only the 64
screen blocks that matter before it writes the coordinate.

## Install

Needs an Apple-silicon Mac (MLX). Python 3.10+.

```sh
pip install "anypointer[macos] @ git+https://github.com/christopher-inegbedion/anypointer"
# [macos] adds Apple Vision OCR, used to read grid labels
anypointer fetch                       # optional: download + convert now (a multi-GB download, ~3 GB on disk after conversion)
```

Without `[macos]` everything works except label reading: grid items fall back to the band the
model pointed at. Bring your own reader with `Pointer(ocr=fn)`, where
`fn(PIL.Image) -> [(text, (x0, y0, x1, y1)), ...]`.

## Hook up to it

### Python

```python
pointer = Pointer()                          # Pointer(model=..., adapters=..., pixels=..., grid=True, ocr=...)
pointer.prime(screenshot)                    # optional, do the screen's read early
p = pointer.point(screenshot, "cell C27")    # screenshot: path, bytes, PIL image or numpy array
p.x, p.y                                     # click here (pixels of the screenshot)
p.confidence, p.method, p.detail             # method: "model" or "grid"
p.normalized                                 # (x, y) as fractions of the image
```

### HTTP (any language)

```sh
anypointer serve --port 8765
curl -s localhost:8765/point -d "{\"image\": \"$(base64 -i screen.png)\", \"item\": \"the wifi toggle\"}"
```

`POST /point`, `POST /prime`, `GET /health`. Binds to localhost; `--allow-paths` lets clients send
`{"path": "/abs/file.png"}` instead of base64.

### MCP (Claude Code, Cursor, any MCP client)

```sh
claude mcp add anypointer -- anypointer mcp
```

Tools: `point_at(image_path, item)` and `prime(image_path)`.

### CLI

```sh
anypointer point screen.png "the wifi toggle"
anypointer point screen.png "e4" --json
```

## Models and weights

The package ships code and 10 KB of head-selection data; weights are fetched on first use from
`mPLUG/GUI-Owl-1.5-4B-Instruct` and converted to MLX 4-bit into `~/.cache/anypointer`
(`$ANYPOINTER_HOME` to move it). `Pointer(model=...)` takes a local directory or a Hugging Face
repo id; a repo that already holds an MLX-quantised copy is used as is.

`adapters=` takes an optional LoRA (`lora.safetensors`, a file, directory or repo id) for the
language layers. It is not needed for the grid-snapping results above.

## Benchmarks

Measured on an M4 MacBook, 16 GB, default settings (`pixels=409600`, no adapters).

**Icons on real screenshots** (ScreenSpot-v2, public benchmark of web, desktop and mobile screens;
icon targets only; every second item, 277 of 554):

| `pixels=` | accuracy | screen read | per item |
|---|---|---|---|
| 409600 (default) | ~83% | ~1.1 s | ~0.3 s |
| 786432 | ~87% | ~2.3 s | ~0.35 s |

**Grid cells by code** (generated screens: chess boards with and without coordinates, spreadsheets,
theatre seat maps, survey matrices, striped tables; item named like `e4` or `C27`):

| | model alone | with grid snapping |
|---|---|---|
| chess squares (180) | ~57% | 99-100% |
| spreadsheet cells (200) | ~46% | 99-100% |
| seats (125) | - | 100% |

These are our own measurements on one machine and our own generated grids, not a published
leaderboard. `docs/how-it-works.md` says how each number was taken.

## Limits

- Apple-silicon Macs only for now: the model runs on MLX. The grid code and the HTTP/MCP layers
  are portable.
- Small icons are the hard case: at the default read size an icon can be a few pixels wide.
  Raise `pixels=` for dense screens (about 2x slower per screen).
- It points at the *item*, so relational wording ("the cheapest one", "the third row") should be turned
  into something visible ("Add to cart under Wool rug") before asking.
- Grid snapping applies to coded references only (letter/number labels such as `C27`, `e4`). Cells named
  by words ("row Sales, column Q3") are left to the model.
- One screenshot is held at a time, and one request is served at a time.

## Develop

```sh
pip install -e ".[dev,macos]"
pytest                                        # fast, no weights needed
ANYPOINTER_TEST_MODEL=/path/to/converted pytest -m model
```

Apache-2.0. See NOTICE for the model's licence.
