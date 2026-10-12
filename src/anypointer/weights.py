"""Finding and fetching the model.

Base model: GUI-Owl-1.5-4B-Instruct (MIT, mPLUG). It is converted once to MLX 4-bit
(group size 64) into ``$ANYPOINTER_HOME`` (default ``~/.cache/anypointer``). A model
argument may be a local directory or a Hugging Face repo id; a repo that already holds
an MLX-quantised copy is used as is, anything else is converted.

Adapters (optional): a ``lora.safetensors`` fine-tune of the language layers that helps on
dense grids. A local file/directory or a Hugging Face repo id holding that file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_REPO = "mPLUG/GUI-Owl-1.5-4B-Instruct"


def home() -> Path:
    return Path(os.environ.get("ANYPOINTER_HOME", Path.home() / ".cache" / "anypointer"))


def is_quantised(model_dir: Path) -> bool:
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return "quantization" in cfg


def resolve_model(spec: str | os.PathLike[str] | None) -> Path:
    """A local MLX-quantised model directory, downloading / converting if needed."""
    spec = str(spec) if spec else BASE_REPO
    local = Path(spec).expanduser()
    if local.is_dir():
        if is_quantised(local):
            return local
        return _convert(str(local), home() / "models" / (local.name + "-4bit"))
    target = home() / "models" / (spec.split("/")[-1] + "-4bit")
    if is_quantised(target):
        return target
    from huggingface_hub import snapshot_download

    got = Path(snapshot_download(spec))
    if is_quantised(got):
        return got
    return _convert(str(got), target)


def _convert(src: str, dest: Path) -> Path:
    from mlx_vlm import convert

    if is_quantised(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    convert(src, mlx_path=str(dest), quantize=True, q_bits=4, q_group_size=64)
    return dest


def resolve_adapters(spec: str | os.PathLike[str] | None) -> Path | None:
    """Path of ``lora.safetensors`` for a local file/directory or a Hugging Face repo id."""
    if not spec:
        return None
    local = Path(str(spec)).expanduser()
    if local.is_file():
        return local
    if local.is_dir():
        if not (local / "lora.safetensors").is_file():
            raise FileNotFoundError(f"no lora.safetensors in {local}")
        return local / "lora.safetensors"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(spec), "lora.safetensors"))


def heads_file() -> Path:
    return Path(__file__).parent / "data" / "guiowl4b-heads.npz"
