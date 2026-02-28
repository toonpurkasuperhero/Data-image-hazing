# src/models/unetpp_wrapper.py
import os
from collections import OrderedDict
import numpy as np
import torch

from src.models.unetpp_model import UNetPP


def _pick_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
        raise ValueError(f"Unrecognized checkpoint dict keys: {list(ckpt.keys())}")
    if isinstance(ckpt, (OrderedDict,)):
        return ckpt
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def _load_state_dict(path: str) -> OrderedDict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    state = _pick_state_dict(ckpt)
    clean = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v
    return clean


class UNetPPWrapper:
    """
    Input tile: HxWx3 uint8 RGB
    Output tile: HxWx3 uint8 RGB

    Normalization: [0,1] -> [-1,1], output -> [-1,1] then back to [0,1]
    """
    def __init__(self, ckpt_path: str, base_ch: int = 32):
        self.ckpt_path = ckpt_path
        self.base_ch = base_ch
        self.net = None
        self.device = "cpu"

    @property
    def name(self):
        return f"unetpp-b{self.base_ch}"

    def load(self, device="cuda"):
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.net = UNetPP(base_ch=self.base_ch)
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            state = _load_state_dict(self.ckpt_path)
            self.net.load_state_dict(state, strict=False)
        self.net.eval().to(self.device)
        return self

    @torch.no_grad()
    def predict_tile(self, tile_rgb_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(tile_rgb_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x * 2.0 - 1.0
        x = x.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(self.device != "cpu")):
            y = self.net(x)
            y = y.clamp(-1, 1)
            y = (y * 0.5 + 0.5).clamp(0, 1)

        out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        return np.clip(out, 0, 255).astype(np.uint8)