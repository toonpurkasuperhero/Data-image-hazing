import numpy as np
import cv2

def gray_world_white_balance(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Gray-world white balance.
    strength in [0,1]: 0 = off, 1 = full correction.
    """
    x = rgb_u8.astype(np.float32)
    mean = x.reshape(-1, 3).mean(axis=0)  # R,G,B means
    gray = mean.mean()
    scale = gray / (mean + 1e-6)
    wb = x * scale[None, None, :]
    wb = np.clip(wb, 0, 255)

    # blend with original to avoid over-correction
    out = (strength * wb + (1.0 - strength) * x)
    return np.clip(out, 0, 255).astype(np.uint8)

def gamma_correction(rgb_u8: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """
    gamma > 1 darkens slightly; gamma < 1 brightens.
    For dehazing output, mild brightening often helps: gamma ~ 0.9 to 1.0.
    """
    x = rgb_u8.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), 1.0 / max(gamma, 1e-6))
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def mild_contrast(rgb_u8: np.ndarray, clip_limit: float = 1.5, tile_grid_size=(8, 8), strength: float = 0.35) -> np.ndarray:
    """
    Very mild CLAHE on L channel in LAB space. Low strength to avoid artifacts.
    strength in [0,1].
    """
    x = rgb_u8.astype(np.uint8)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    y = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    out = (strength * y.astype(np.float32) + (1.0 - strength) * x.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)

def postprocess(rgb_u8: np.ndarray, wb_strength: float = 0.6, gamma: float = 0.95, clahe_strength: float = 0.25) -> np.ndarray:
    """
    Safe default pipeline:
      1) mild gray-world WB (prevents weird color cast)
      2) mild gamma (slight brightening)
      3) mild CLAHE on luminance (small contrast boost)
    """
    out = gray_world_white_balance(rgb_u8, strength=wb_strength)
    out = gamma_correction(out, gamma=gamma)
    out = mild_contrast(out, strength=clahe_strength)
    return out