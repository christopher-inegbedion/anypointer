"""anypointer: name a thing on a screenshot, get the pixel to click.

    from anypointer import Pointer
    Pointer().point("screen.png", "the wifi toggle")
"""

from .pointer import Pointer, load_image
from .types import Point

__version__ = "0.1.0"
__all__ = ["Point", "Pointer", "__version__", "load_image"]
