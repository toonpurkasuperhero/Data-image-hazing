import torch
import pywt

# db1 == Haar
_w = pywt.Wavelet("db1")

dec_hi = torch.tensor(_w.dec_hi[::-1], dtype=torch.float32)
dec_lo = torch.tensor(_w.dec_lo[::-1], dtype=torch.float32)
rec_hi = torch.tensor(_w.rec_hi, dtype=torch.float32)
rec_lo = torch.tensor(_w.rec_lo, dtype=torch.float32)

# Filters from the official repo (same math), but device-safe. :contentReference[oaicite:3]{index=3}
_filters = torch.stack([
    dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
], dim=0)

_inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
], dim=0)


def wt(vimg: torch.Tensor) -> torch.Tensor:
    """
    vimg: [B, C, H, W] on any device
    returns: [B, 4C, H/2, W/2]
    """
    device = vimg.device
    B, C, H, W = vimg.shape
    filters = _filters.to(device=device, dtype=vimg.dtype)

    res = torch.zeros((B, 4 * C, H // 2, W // 2), device=device, dtype=vimg.dtype)
    for i in range(C):
        out = torch.nn.functional.conv2d(vimg[:, i:i+1], filters[:, None], stride=2)
        res[:, 4*i:4*i+4] = out

        # match repo scaling for high-freq bands  :contentReference[oaicite:4]{index=4}
        res[:, 4*i+1:4*i+4] = (res[:, 4*i+1:4*i+4] + 1) / 2.0
    return res


def iwt(vres: torch.Tensor) -> torch.Tensor:
    """
    vres: [B, 4C, H, W] on any device
    returns: [B, C, 2H, 2W]
    """
    device = vres.device
    B, C4, H, W = vres.shape
    C = C4 // 4
    inv_filters = _inv_filters.to(device=device, dtype=vres.dtype)

    res = torch.zeros((B, C, H * 2, W * 2), device=device, dtype=vres.dtype)

    vres = vres.clone()
    for i in range(C):
        # invert repo scaling for high-freq bands  :contentReference[oaicite:5]{index=5}
        vres[:, 4*i+1:4*i+4] = 2 * vres[:, 4*i+1:4*i+4] - 1
        temp = torch.nn.functional.conv_transpose2d(vres[:, 4*i:4*i+4], inv_filters[:, None], stride=2)
        res[:, i:i+1] = temp
    return res