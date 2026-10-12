import base64
import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from anypointer import Point, Pointer, load_image, mcp, weights
from anypointer.engine import fit_pixels
from anypointer.server import make_server


class FakePointer:
    _engine = object()

    def point(self, image, item):
        im = load_image(image)
        return Point(10.0, 20.0, 0.9, "model", "model", (10.0, 20.0), im.size, 1.0)

    def prime(self, image):
        load_image(image)


def png_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


def test_load_image_accepts_the_usual_shapes(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(png_bytes())
    for src in (p, str(p), png_bytes(), Image.open(p), np.zeros((8, 8, 3), dtype=np.uint8), io.BytesIO(png_bytes())):
        assert load_image(src).mode == "RGB"
    with pytest.raises(TypeError):
        load_image(3)


def test_fit_pixels_is_a_multiple_of_32_near_the_budget():
    w, h = fit_pixels((1920, 1080), 409_600)
    assert w % 32 == 0 and h % 32 == 0 and abs(w * h - 409_600) / 409_600 < 0.1
    assert fit_pixels((100, 100), 409_600, enlarge=False) == (96, 96)


def test_point_dict_has_normalized():
    d = Point(50, 25, 0.5, "model", "model", (50, 25), (100, 100), 3.0).to_dict()
    assert d["normalized"] == (0.5, 0.25)


def test_mcp_handshake_and_call():
    fake = FakePointer()
    init = mcp.handle(fake, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "anypointer"
    assert mcp.handle(fake, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in mcp.handle(fake, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
    assert names == ["point_at", "prime"]
    r = mcp.handle(fake, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "point_at", "arguments": {"image_path": "/nonexistent.png", "item": "x"}}})
    assert r["result"]["isError"] is True  # tool errors are results the agent can read
    assert mcp.handle(fake, {"jsonrpc": "2.0", "id": 4, "method": "nope"})["error"]["code"] == -32601


def test_mcp_stdio_loop(tmp_path):
    p = tmp_path / "s.png"
    p.write_bytes(png_bytes())
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "point_at", "arguments": {"image_path": str(p), "item": "button"}}}]
    out = io.StringIO()
    mcp.run(FakePointer(), io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\nnot json\n"), out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert json.loads(lines[0]["result"]["content"][0]["text"])["x"] == 10.0
    assert lines[1]["error"]["code"] == -32700


@pytest.fixture
def server():
    srv = make_server(FakePointer(), port=0)  # port 0: any free port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_point(server):
    img = base64.b64encode(png_bytes()).decode()
    code, body = post(server + "/point", {"image": img, "item": "ok button"})
    assert code == 200 and body["x"] == 10.0 and body["normalized"][0] == 10 / 64
    assert post(server + "/point", {"image": img})[0] == 400  # no item
    assert post(server + "/point", {"path": "/etc/hosts", "item": "x"})[0] == 403  # paths are off by default
    assert post(server + "/prime", {"image": img})[0] == 200
    with urllib.request.urlopen(server + "/health") as r:
        assert json.loads(r.read())["ok"] is True


def test_weights_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYPOINTER_HOME", str(tmp_path / "home"))
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    assert weights.resolve_model(model) == model  # already quantised: used as is
    assert weights.resolve_adapters(None) is None
    d = tmp_path / "ad"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        weights.resolve_adapters(d)
    (d / "lora.safetensors").write_bytes(b"x")
    assert weights.resolve_adapters(d) == d / "lora.safetensors"
    assert weights.heads_file().is_file()


def test_pointer_is_lazy():
    p = Pointer(model="/does/not/exist")  # nothing is loaded until the first call
    assert p._engine is None
