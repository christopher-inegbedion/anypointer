from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Point:
    """Where to click for an item, in pixels of the image that was given."""

    x: float
    y: float
    confidence: float  # 0-1: the model's own certainty in its leading coordinate digits
    method: str  # "model": the model's point; "grid": snapped to a cell of a measured grid
    detail: str  # how the point was reached ("grid (rank by board convention)", "model (no grid)", ...)
    model_xy: tuple[float, float]  # the model's own point, before any snapping
    image_size: tuple[int, int]  # (width, height) of the image
    elapsed_ms: float  # time for this call (the first call on a screen includes its read)

    @property
    def normalized(self) -> tuple[float, float]:
        """(x, y) as fractions 0-1 of the image width and height."""
        return self.x / self.image_size[0], self.y / self.image_size[1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["normalized"] = self.normalized
        return d
