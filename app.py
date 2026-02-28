import os
import cv2
import numpy as np
import torch
import gradio as gr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.models.dehazeformer_wrapper import DehazeFormerWrapper
from src.models.wavelet_unet_wrapper import WaveletUNetWrapper
from src.models.dummy import DummyDehaze
from src.utils.tiling import tiled_process
from src.utils.postprocess import postprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_dir = "weights"
indoor_path = os.path.join(weights_dir, "dehazeformer_indoor_s.pth")
outdoor_path = os.path.join(weights_dir, "dehazeformer_outdoor_s.pth")
unetpp_path = os.path.join(weights_dir, "unetpp.pth")

print("Loading models to CPU initially...")
model_indoor = DehazeFormerWrapper(variant="s", ckpt_path=indoor_path).load(device="cpu")
model_outdoor = DehazeFormerWrapper(variant="s", ckpt_path=outdoor_path).load(device="cpu")
model_unetpp = WaveletUNetWrapper(ckpt_path=unetpp_path).load(device="cpu")
model_dummy = DummyDehaze().load(device="cpu")
print("Models loaded successfully.")

def run_with_fallback_tiles(img, fn_tile, overlap=16, tiles=(64, 32)):
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

def compute_metrics(pred_u8, gt_u8):
    if gt_u8 is None:
        return None, None
    if pred_u8.shape != gt_u8.shape:
        gt_u8 = cv2.resize(gt_u8, (pred_u8.shape[1], pred_u8.shape[0]))
    
    psnr = peak_signal_noise_ratio(gt_u8, pred_u8)
    ssim = structural_similarity(gt_u8, pred_u8, channel_axis=2)
    return psnr, ssim

def process_image(model, img_u8):
    if hasattr(model, 'net') and model.net is not None:
        model.net.to(DEVICE)
        model.device = DEVICE
    try:
        res = run_with_fallback_tiles(img_u8, fn_tile=model.predict_tile)
        res = postprocess(res, wb_strength=0.6, gamma=0.95, clahe_strength=0.25)
    finally:
        if hasattr(model, 'net') and model.net is not None:
            model.net.to("cpu")
            model.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return res

def haze_to_clean(hazy_img, gt_img):
    if hazy_img is None:
        return None, None, None, "Please upload a hazy image."

    res_indoor = process_image(model_indoor, hazy_img)
    res_outdoor = process_image(model_outdoor, hazy_img)
    res_unetpp = process_image(model_unetpp, hazy_img)
    
    results = []
    
    for name, pred in [("DehazeFormer Indoor", res_indoor), 
                       ("DehazeFormer Outdoor", res_outdoor), 
                       ("UNetPP", res_unetpp)]:
        if gt_img is not None:
            psnr, ssim = compute_metrics(pred, gt_img)
            metrics_str = f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
        else:
            metrics_str = "Metrics: N/A (No GT provided)"
        results.append((pred, metrics_str))
        
    metric_text = "\n".join([f"{name} -> {res[1]}" for name, res in zip(["DehazeFormer Indoor", "DehazeFormer Outdoor", "UNetPP"], results)])
    
    return results[0][0], results[1][0], results[2][0], metric_text

def add_haze(img, beta=2.0, A=None):
    img_f = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    gradient = np.tile(np.linspace(0, 1.5, h), (w,1)).T
    noise = cv2.GaussianBlur(np.random.rand(h, w).astype(np.float32), (51,51), 0)
    depth = 0.6 * gradient + 0.4 * noise

    t = np.exp(-beta * depth)
    t = np.expand_dims(t, axis=2)

    if A is None:
        A = np.array([0.95, 0.9, 0.85], dtype=np.float32)

    hazy = img_f * t + A * (1 - t)
    hazy = np.clip(hazy, 0, 1)

    return (hazy * 255).astype(np.uint8)

def clean_to_haze_to_clean(clean_img, beta):
    if clean_img is None:
        return None, None, None, None, None, "Please upload a clean image."
        
    hazy_img = add_haze(clean_img, beta=beta)
    
    res_indoor = process_image(model_indoor, hazy_img)
    res_outdoor = process_image(model_outdoor, hazy_img)
    res_unetpp = process_image(model_unetpp, hazy_img)
    res_dummy = process_image(model_dummy, hazy_img)
    
    results = []
    for name, pred in [("DehazeFormer Indoor", res_indoor), 
                       ("DehazeFormer Outdoor", res_outdoor), 
                       ("UNetPP", res_unetpp),
                       ("Dummy (CLAHE)", res_dummy)]:
        psnr, ssim = compute_metrics(pred, clean_img) 
        metrics_str = f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
        results.append((pred, metrics_str))
        
    metric_text = "\n".join([f"{name} -> {res[1]}" for name, res in zip(["DehazeFormer Indoor", "DehazeFormer Outdoor", "UNetPP", "Dummy (CLAHE)"], results)])
    
    return hazy_img, results[0][0], results[1][0], results[2][0], results[3][0], metric_text

with gr.Blocks(title="Image Dehazing - Team The Outliers") as app:
    gr.Markdown("# Image Dehazing Pipeline")
    gr.Markdown("Team: **The outliers**")
    
    with gr.Tabs():
        with gr.Tab("Mode 1: Haze to Clean"):
            with gr.Row():
                with gr.Column():
                    h1_in = gr.Image(label="Hazy Image", type="numpy")
                    h1_gt = gr.Image(label="Ground Truth (Optional, for metrics)", type="numpy")
                    h1_btn = gr.Button("Dehaze")
                with gr.Column():
                    h1_out1 = gr.Image(label="DehazeFormer (Indoor)")
                    h1_out2 = gr.Image(label="DehazeFormer (Outdoor)")
                    h1_out3 = gr.Image(label="UNetPP")
                    h1_metrics = gr.Textbox(label="Metrics", lines=4)
                    
            h1_btn.click(fn=haze_to_clean, inputs=[h1_in, h1_gt], outputs=[h1_out1, h1_out2, h1_out3, h1_metrics])
            
        with gr.Tab("Mode 2: Clean to Haze => Dehaze"):
            with gr.Row():
                with gr.Column():
                    h2_in = gr.Image(label="Clean Image (GT)", type="numpy")
                    h2_beta = gr.Slider(minimum=0.5, maximum=4.0, value=2.0, step=0.1, label="Haze Density (Beta)")
                    h2_btn = gr.Button("Haze and Dehaze")
                with gr.Column():
                    h2_hazy = gr.Image(label="Synthetic Hazy Image")
                    h2_out1 = gr.Image(label="DehazeFormer (Indoor)")
                    h2_out2 = gr.Image(label="DehazeFormer (Outdoor)")
                    h2_out3 = gr.Image(label="UNetPP")
                    h2_out4 = gr.Image(label="Dummy (CLAHE)")
                    h2_metrics = gr.Textbox(label="Metrics", lines=5)
                    
            h2_btn.click(fn=clean_to_haze_to_clean, inputs=[h2_in, h2_beta], outputs=[h2_hazy, h2_out1, h2_out2, h2_out3, h2_out4, h2_metrics])

if __name__ == "__main__":
    app.launch()
