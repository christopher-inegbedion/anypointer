import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

from anypointer import grid as G


def test_parse_ref():
    assert G.parse_ref("C27") == ("C", "27")
    assert G.parse_ref("square e4") == ("e", "4")
    assert G.parse_ref("cell B16") == ("B", "16")
    assert G.parse_ref("row 3, column Good") == ("Good", "3")
    assert G.parse_ref("the wifi toggle") is None
    assert G.parse_ref("settings") is None


def test_seq_value():
    assert G.seq_value("27") == 27
    assert G.seq_value("c") == 3 and G.seq_value("AA") == 27
    assert G.seq_value("Agree") is None


def sheet(cols=8, rows=18, cw=96, rh=28, ox=70, oy=40):
    im = Image.new("RGB", (ox + cols * cw + 20, oy + rows * rh + 20), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=15)
    for c in range(cols + 1):
        d.line([(ox + c * cw, oy), (ox + c * cw, oy + rows * rh)], fill=(170, 170, 170))
    for r in range(rows + 1):
        d.line([(ox, oy + r * rh), (ox + cols * cw, oy + r * rh)], fill=(170, 170, 170))
    for c in range(cols):
        d.text((ox + c * cw + cw // 2 - 4, oy - 24), "ABCDEFGH"[c], fill="black", font=f)
    for r in range(rows):
        d.text((ox - 40, oy + r * rh + 6), str(r + 1), fill="black", font=f)
    return im, (ox, oy, cw, rh)


def test_find_grid_measures_a_sheet():
    im, (ox, oy, cw, rh) = sheet()
    g = G.find_grid(im, ox + 3.5 * cw, oy + 5.5 * rh)
    assert g is not None
    cols, rows = g
    assert abs(cols[1] - cols[0] - cw) <= 2 and abs(rows[1] - rows[0] - rh) <= 2
    assert len(cols) - 1 >= 8 and len(rows) - 1 >= 18


def test_snap_without_ocr_keeps_the_models_band():
    im, (ox, oy, cw, rh) = sheet()
    # no label reader: a spreadsheet cell falls back to the band the model pointed at, centred
    s = G.snap(im, "C7", ox + 2.4 * cw, oy + 6.9 * rh, None)
    assert s.snapped and s.how.endswith("from the model)")
    assert abs(s.x - (ox + 2.5 * cw)) <= 2 and abs(s.y - (oy + 6.5 * rh)) <= 2


def test_snap_leaves_plain_items_alone():
    im, _ = sheet()
    s = G.snap(im, "the save button", 100, 100, None)
    assert (s.x, s.y, s.how) == (100, 100, "model")


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision OCR")
def test_snap_reads_labels_with_ocr():
    pytest.importorskip("Vision")
    from anypointer.ocr import apple_vision

    im, (ox, oy, cw, rh) = sheet()
    # the model is a row off (it pointed at row 6 for "C7"): the labels fix it
    s = G.snap(im, "C7", ox + 2.5 * cw, oy + 5.5 * rh, apple_vision)
    assert s.how == "grid"
    assert abs(s.x - (ox + 2.5 * cw)) <= 2 and abs(s.y - (oy + 6.5 * rh)) <= 2
