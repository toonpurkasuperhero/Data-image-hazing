# src/infer.py
import argparse
import os
import torch

from src.utils.io import load_rgb, save_rgb
from src.utils.tiling import tiled_process
from src.utils.postprocess import postprocess

from src.models.dehazeformer_wrapper import DehazeFormerWrapper
from src.models.wavelet_unet_wrapper import WaveletUNetWrapper


def run_with_fallback_tiles(img, fn_tile, overlap: int, tiles):
    """
    Try tiles from larger to smaller. If CUDA OOM, reduce tile size.
    """
    last_err = None
    for t in tiles:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return tiled_process(img, fn_tile=fn_tile, tile=t, overlap=overlap)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                last_err = e
                continue
            raise
    raise RuntimeError(f"All tile sizes failed due to OOM. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--model",
        default="dehazeformer",
        choices=["dehazeformer", "wavelet-unet"],
        help="Select model backend",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tiles", default="640,512,384,320")

    # DehazeFormer-specific
    ap.add_argument("--variant", default="t", choices=["t", "s", "b"], help="DehazeFormer variant (t/s/b)")

    # Shared: checkpoint path for selected model
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pth) for chosen model")

    # Post-processing
    ap.add_argument("--post", action="store_true", help="Enable safe post-processing (WB + gamma + mild CLAHE)")
    ap.add_argument("--wb_strength", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--clahe_strength", type=float, default=0.25)

    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    img = load_rgb(args.input)

    if args.model == "dehazeformer":
        model = DehazeFormerWrapper(variant=args.variant, ckpt_path=args.ckpt).load(device=args.device)
    elif args.model == "wavelet-unet":
        model = WaveletUNetWrapper(ckpt_path=args.ckpt).load(device=args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    def fn(tile_img):
        return model.predict_tile(tile_img)

    tiles = [int(x.strip()) for x in args.tiles.split(",") if x.strip()]
    out = run_with_fallback_tiles(img, fn_tile=fn, overlap=args.overlap, tiles=tiles)

    if args.post:
        out = postprocess(
            out,
            wb_strength=float(args.wb_strength),
            gamma=float(args.gamma),
            clahe_strength=float(args.clahe_strength),
        )

    save_rgb(args.output, out)
    print(f"Saved: {args.output} | model={model.name} | device={args.device} | post={args.post}")


if __name__ == "__main__":
    main()