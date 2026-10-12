"""Command line: ``anypointer point|serve|mcp|fetch``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .pointer import Pointer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anypointer", description="Name a thing on a screenshot, get the pixel to click.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--model", help="local directory or Hugging Face repo id of GUI-Owl-1.5-4B (default: convert the original)")
    ap.add_argument("--adapters", help="lora.safetensors (file, directory or Hugging Face repo id)")
    ap.add_argument("--pixels", type=int, default=409_600, help="area the screenshot is read at (default 409600)")
    ap.add_argument("--no-grid", action="store_true", help="never snap to grid cells")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point", help="print the click point for an item on a screenshot")
    p.add_argument("image")
    p.add_argument("item")
    p.add_argument("--json", action="store_true")
    s = sub.add_parser("serve", help="local HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-paths", action="store_true", help="let clients name files on this machine")
    sub.add_parser("mcp", help="MCP server on stdio")
    sub.add_parser("fetch", help="download and convert the model now")
    a = ap.parse_args(argv)

    pointer = Pointer(model=a.model, adapters=a.adapters, pixels=a.pixels, grid=not a.no_grid)
    if a.cmd == "fetch":
        from . import weights
        print(weights.resolve_model(a.model))
        if a.adapters:
            print(weights.resolve_adapters(a.adapters))
        return 0
    if a.cmd == "point":
        pt = pointer.point(a.image, a.item)
        if a.json:
            print(json.dumps(pt.to_dict()))
        else:
            print(f"{pt.x:.0f} {pt.y:.0f}  (confidence {pt.confidence:.2f}, {pt.detail}, {pt.elapsed_ms:.0f} ms)")
        return 0
    if a.cmd == "serve":
        from .server import serve
        serve(pointer, a.host, a.port, a.allow_paths)
        return 0
    if a.cmd == "mcp":
        from .mcp import run
        run(pointer)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
