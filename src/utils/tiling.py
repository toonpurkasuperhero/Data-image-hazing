import numpy as np


# this script is for vram bounding by limiting the tile of the images of input.
def _weight_map(h: int, w: int, border: int) -> np.ndarray:
    """
    Smooth weight map to blend overlapping tiles and avoid seams.
    border: overlap size (px)
    """
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    if border > 0:
        ramp = np.linspace(0.0, 1.0, border, dtype=np.float32)
        wy[:border] = ramp
        wy[-border:] = ramp[::-1]
        wx[:border] = ramp
        wx[-border:] = ramp[::-1]

    return wy[:, None] * wx[None, :]

def tiled_process(
    img: np.ndarray,
    fn_tile,
    tile: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    img: HxWxC uint8 or float32 (0..1)
    fn_tile: function(tile_img)->tile_out with same shape
    Returns: stitched output with same HxWxC
    """
    assert img.ndim == 3 and img.shape[2] in (3, 4), "Expected HxWxC"
    H, W, C = img.shape
    step = tile - overlap
    if step <= 0:
        raise ValueError("tile must be > overlap")

    out = np.zeros((H, W, C), dtype=np.float32)
    acc = np.zeros((H, W, 1), dtype=np.float32)

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)

            # shift tile window to keep size consistent near borders
            y0s = max(0, y1 - tile)
            x0s = max(0, x1 - tile)
            y1s = min(y0s + tile, H)
            x1s = min(x0s + tile, W)

            tile_in = img[y0s:y1s, x0s:x1s, :]
            tile_out = fn_tile(tile_in)

            th, tw = tile_out.shape[:2]
            wmap = _weight_map(th, tw, border=min(overlap // 2, th // 4, tw // 4)).astype(np.float32)
            wmap = wmap[..., None]  # th x tw x 1

            out[y0s:y1s, x0s:x1s, :] += tile_out.astype(np.float32) * wmap
            acc[y0s:y1s, x0s:x1s, :] += wmap

    out = out / np.clip(acc, 1e-6, None)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)