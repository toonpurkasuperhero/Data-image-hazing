# src/models/dehazeformer_wrapper.py
import os
import sys
import importlib
from collections import OrderedDict
import numpy as np
import torch


def _find_dehazeformer_root() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    candidates = [
        os.path.join(project_root, "third_party", "DehazeFormer"),
        os.path.join(project_root, "third_party", "third_party", "DehazeFormer"),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "models")):
            return p
    raise RuntimeError(f"DehazeFormer repo not found. Tried: {candidates}")


DEHAZEFORMER_ROOT = _find_dehazeformer_root()
if DEHAZEFORMER_ROOT not in sys.path:
    sys.path.insert(0, DEHAZEFORMER_ROOT)

# Import DehazeFormer constructors from its own package
dehazeformer_module = importlib.import_module("models")
dehazeformer_t = getattr(dehazeformer_module, "dehazeformer_t")
dehazeformer_s = getattr(dehazeformer_module, "dehazeformer_s")
dehazeformer_b = getattr(dehazeformer_module, "dehazeformer_b")


def _pick_state_dict(ckpt: object) -> dict:
    """
    Supports common checkpoint formats:
      - {'state_dict': ...}
      - {'model': ...}
      - {'net': ...}
      - raw tensor-valued mapping
    """
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # raw state dict (tensor values)
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

        raise ValueError(f"Unrecognized checkpoint dict keys: {list(ckpt.keys())}")

    if isinstance(ckpt, (OrderedDict,)):
        return ckpt

    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def _load_state_dict_compat(ckpt_path: str) -> OrderedDict:
    """
    Loads .pth via torch.load and returns cleaned state_dict.
    - Strips 'module.' prefix if present.
    - If on torch>=2.0, tries weights_only=True first.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state = _pick_state_dict(ckpt)

    clean = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v
    return clean


class DehazeFormerWrapper:
    """
    VRAM-safe DehazeFormer wrapper.
    Input/Output:
      - input tile: HxWx3 uint8 RGB
      - output tile: HxWx3 uint8 RGB

    Normalization matches DehazeFormer test pipeline:
      x in [0,1] -> [-1,1]
      y clamped [-1,1] -> [0,1]
    """
    def __init__(self, variant: str, ckpt_path: str):
        self.variant = variant.lower()
        self.ckpt_path = ckpt_path
        self.net = None
        self.device = "cpu"

    @property
    def name(self) -> str:
        return f"dehazeformer-{self.variant}"

    def load(self, device: str = "cuda"):
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        if self.variant == "t":
            self.net = dehazeformer_t()
        elif self.variant == "s":
            self.net = dehazeformer_s()
        elif self.variant == "b":
            self.net = dehazeformer_b()
        else:
            raise ValueError("variant must be one of: t, s, b")

        state = _load_state_dict_compat(self.ckpt_path)
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(self.device)
        return self

    @torch.no_grad()
    def predict_tile(self, tile_rgb_u8: np.ndarray) -> np.ndarray:
        if tile_rgb_u8.ndim != 3 or tile_rgb_u8.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB tile")

        x = torch.from_numpy(tile_rgb_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x * 2.0 - 1.0  # [-1, 1]
        x = x.to(self.device, non_blocking=True)

        use_amp = False
        with torch.cuda.amp.autocast(enabled=use_amp):
            y = self.net(x)
            y = y.clamp(-1, 1)
            y = (y * 0.5 + 0.5).clamp(0, 1)

        out = (y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
        return np.clip(out, 0, 255).astype(np.uint8)