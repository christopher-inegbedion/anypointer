"""Text recognition for grid labels.

Grid snapping reads row and column labels ("27", "C", "e") with OCR. Any callable
``ocr(image: PIL.Image) -> [(text, (x0, y0, x1, y1)), ...]`` works (boxes in image
pixels); ``apple_vision`` is the built-in one and needs ``pip install anypointer[macos]``.
"""

from __future__ import annotations

import io
from collections.abc import Callable

from PIL import Image

Box = tuple[float, float, float, float]
OcrBackend = Callable[[Image.Image], list[tuple[str, Box]]]


def apple_vision(image: Image.Image, fast: bool = False) -> list[tuple[str, Box]]:
    """Every word on ``image`` with its pixel box, from macOS's on-device text recognition."""
    import Quartz
    import Vision
    from Foundation import NSData, NSMakeRange

    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    W, H = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                             else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)  # labels like "C27" are not words
    req.setMinimumTextHeight_(0.0)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, _err = handler.performRequests_error_([req], None)
    out: list[tuple[str, Box]] = []
    if not ok:
        return out
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        rec = cand[0]
        text = str(rec.string())
        pos = 0
        for word in text.split():  # the line, split into words each with its own box
            start = text.index(word, pos)
            pos = start + len(word)
            box, _e = rec.boundingBoxForRange_error_(NSMakeRange(start, len(word)), None)
            if box is None:
                continue
            bb = box.boundingBox()
            x0 = bb.origin.x * W
            y1 = (1 - bb.origin.y) * H
            out.append((word, (x0, y1 - bb.size.height * H, x0 + bb.size.width * W, y1)))
    return out


def default_backend() -> OcrBackend | None:
    """Apple Vision where available, else None (grid snapping then falls back to the model)."""
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except ImportError:
        return None
    return apple_vision
