import numpy as np
from PIL import Image

def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def save_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)

def to_float01(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32) / 255.0

def to_u8(rgb_f01: np.ndarray) -> np.ndarray:
    return np.clip(rgb_f01 * 255.0, 0, 255).astype(np.uint8)