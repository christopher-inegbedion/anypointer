"""Local HTTP server: any language or agent framework can ask for a point.

    anypointer serve [--host 127.0.0.1] [--port 8765]

    POST /point   {"image": "<base64 png/jpg>" | "path": "/abs/file.png", "item": "the wifi toggle"}
                  -> {"x", "y", "confidence", "method", "detail", "normalized", "image_size", ...}
    POST /prime   {"image": ... | "path": ...}     do a screen's read ahead of its first item
    GET  /health  -> {"ok": true, "loaded": bool}

Stdlib only. Requests are served one at a time (one GPU). ``path`` reads files from the server's
machine, so it is off unless ``--allow-paths`` is given; the server binds to localhost by default.
"""

from __future__ import annotations

import base64
import binascii
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .pointer import Pointer

MAX_BODY = 64 * 1024 * 1024


def _image_from(body: dict[str, Any], allow_paths: bool) -> Any:
    if "image" in body:
        try:
            return base64.b64decode(body["image"].split(",", 1)[-1], validate=False)
        except (binascii.Error, AttributeError) as e:
            raise ValueError("image must be base64 text") from e
    if "path" in body:
        if not allow_paths:
            raise PermissionError("path requests are disabled; send base64 'image' or start with --allow-paths")
        return body["path"]
    raise ValueError("send 'image' (base64) or 'path'")


def make_server(pointer: Pointer, host: str = "127.0.0.1", port: int = 8765,
                allow_paths: bool = False) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "loaded": pointer._engine is not None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._send(413, {"error": "request too large"})
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/point":
                    item = body.get("item")
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError("send 'item', the name of what to point at")
                    self._send(200, pointer.point(_image_from(body, allow_paths), item).to_dict())
                elif self.path == "/prime":
                    pointer.prime(_image_from(body, allow_paths))
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except PermissionError as e:
                self._send(403, {"error": str(e)})
            except (ValueError, json.JSONDecodeError, OSError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args: Any) -> None:  # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(pointer: Pointer, host: str = "127.0.0.1", port: int = 8765, allow_paths: bool = False) -> None:
    pointer.load()
    srv = make_server(pointer, host, port, allow_paths)
    print(f"anypointer listening on http://{host}:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
