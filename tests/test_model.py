"""Needs the weights: ANYPOINTER_TEST_MODEL=/path/to/guiowl15-4b-4bit pytest -m model"""
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from anypointer import Pointer

MODEL = os.environ.get("ANYPOINTER_TEST_MODEL")
pytestmark = [pytest.mark.model, pytest.mark.skipif(not MODEL, reason="ANYPOINTER_TEST_MODEL not set")]


def test_points_at_a_labelled_button(tmp_path: Path):
    im = Image.new("RGB", (1280, 800), (245, 245, 245))
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=28)
    for label, x in (("Cancel", 300), ("Save", 600), ("Delete", 900)):
        d.rounded_rectangle([x, 500, x + 200, 570], 12, fill=(40, 100, 220), outline=(20, 60, 160))
        d.text((x + 100, 535), label, fill="white", font=f, anchor="mm")
    p = Pointer(model=MODEL).point(im, "the Save button")
    assert 600 <= p.x <= 800 and 500 <= p.y <= 570
    assert 0 < p.confidence <= 1
