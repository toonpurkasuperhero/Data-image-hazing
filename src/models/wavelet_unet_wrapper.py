# src/models/wavelet_unet_wrapper.py
import os
from collections import OrderedDict
import numpy as np
import torch
from src.models.wavelet_unet_model import ACT


def _pick_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
        raise ValueError(f"Unrecognized checkpoint keys: {list(ckpt.keys())}")
    if isinstance(ckpt, OrderedDict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def _load_state_dict(path: str) -> OrderedDict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", encoding="latin1")

    state = _pick_state_dict(ckpt)

    clean = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v
    return clean


class WaveletUNetWrapper:
    def __init__(self, ckpt_path: str, debug: bool = True):
        self.ckpt_path = ckpt_path
        self.debug = debug
        self.net = None
        self.device = "cpu"

    @property
    def name(self):
        return "wavelet-unet"

    def load(self, device="cuda"):
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.net = ACT().to(self.device).eval()

        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(self.ckpt_path)

        state = _load_state_dict(self.ckpt_path)

        # Drop BN entries that don't match (yours has 320 channels)
        drop = [k for k in state.keys() if ".bn." in k]
        for k in drop:
            state.pop(k, None)

        missing, unexpected = self.net.load_state_dict(state, strict=False)

        if self.debug:
            print(f"[WaveletUNet] Dropped BN keys: {len(drop)}")
            print(f"[WaveletUNet] Missing keys: {len(missing)}")
            print(f"[WaveletUNet] Unexpected keys: {len(unexpected)}")
            if len(unexpected) > 0:
                print("  Unexpected (first 20):", unexpected[:20])
            if len(missing) > 0:
                print("  Missing (first 20):", missing[:20])

        return self

    @torch.no_grad()
    def predict_tile(self, tile_rgb_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(tile_rgb_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            y = self.net(x)

        y = y.clamp(0, 1)
        out = (y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
        return np.clip(out, 0, 255).astype(np.uint8)