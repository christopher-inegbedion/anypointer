"""MCP server (stdio), no SDK needed: tools for agents that can see screenshots.

    anypointer mcp          # add to an MCP client as: command "anypointer", args ["mcp"]

Tools:
  point_at(image_path, item)   where to click for ``item`` on the screenshot at image_path
  prime(image_path)            warm up a screenshot so the next point_at is fast

Speaks JSON-RPC 2.0, one message per line, MCP protocol 2024-11-05 (tools only).
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .pointer import Pointer

PROTOCOL = "2024-11-05"
TOOLS = [
    {
        "name": "point_at",
        "description": ("Find the pixel to click for a named item on a screenshot. Give the ITEM as it looks "
                        "('the wifi toggle', 'Add to cart under Wool rug'), not an instruction. Grid cells may "
                        "be named by code ('C27', 'e4'). Returns x, y in image pixels, a confidence 0-1, and "
                        "whether the point was snapped to a grid cell."),
        "inputSchema": {"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Absolute path of a PNG/JPEG screenshot"},
            "item": {"type": "string", "description": "What to point at"}},
            "required": ["image_path", "item"]},
    },
    {
        "name": "prime",
        "description": "Pre-read a screenshot so later point_at calls on it answer in a fraction of a second.",
        "inputSchema": {"type": "object", "properties": {"image_path": {"type": "string"}},
                        "required": ["image_path"]},
    },
]


def _call(pointer: Pointer, name: str, args: dict[str, Any]) -> str:
    if name == "point_at":
        p = pointer.point(args["image_path"], args["item"])
        return json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.to_dict().items()})
    if name == "prime":
        pointer.prime(args["image_path"])
        return json.dumps({"ok": True})
    raise KeyError(name)


def handle(pointer: Pointer, msg: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON-RPC message in, the response out (None for notifications)."""
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None  # notifications/initialized etc.

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code: int, text: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}

    if method == "initialize":
        return ok({"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                   "serverInfo": {"name": "anypointer", "version": _version()}})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        try:
            return ok({"content": [{"type": "text", "text": _call(pointer, params.get("name", ""),
                                                                   params.get("arguments") or {})}]})
        except KeyError as e:
            return err(-32602, f"unknown tool or missing argument: {e}")
        except Exception as e:
            return ok({"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return err(-32601, f"method not found: {method}")


def _version() -> str:
    from . import __version__
    return __version__


def run(pointer: Pointer, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(pointer, json.loads(line))
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
