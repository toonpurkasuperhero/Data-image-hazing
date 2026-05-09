# Image Dehazing Pipeline - Team The Outliers

## Overview
This repository contains a comprehensive Image Dehazing Pipeline developed by **Team The Outliers**. The project provides tools for removing haze from images using state-of-the-art deep learning architectures. It includes a user-friendly web interface powered by Gradio, as well as robust command-line scripts for batch evaluation and single-image inference. 

The pipeline supports synthetic haze generation, performance metric calculation, and robust handling of high-resolution images to avoid out-of-memory (OOM) errors.

## Key Features
- **Multiple Dehazing Models**: Includes wrappers and implementations for `DehazeFormer` (Indoor & Outdoor variants), `UNetPP` (UNet++), and `Wavelet-UNet`. 
- **Interactive Gradio App**: A fully functional web interface (`app.py`) with two intuitive modes:
  - *Haze to Clean*: Upload a hazy image (and optionally a Ground Truth) to generate clean outputs using multiple models, and compare PSNR/SSIM metrics.
  - *Clean to Haze => Dehaze*: Upload a clean image, apply synthetic haze with a tunable density slider (Beta), and instantly run dehazing models to benchmark their restoration capabilities.
- **Tiled Processing for High-Res Images**: Integrates fallback tile-processing logic (`run_with_fallback_tiles`) to dynamically adjust tile sizes for inference, preventing CUDA Out-Of-Memory crashes on large inputs.
- **Advanced Post-Processing**: Includes tools for mild Contrast Limited Adaptive Histogram Equalization (CLAHE), gamma correction, and white balance adjustment to improve final output visual quality.
- **CLI Tools for Inference and Evaluation**: Dedicated scripts (`src/eval.py` and `src/infer.py`) to easily benchmark models over datasets or process single files directly from the terminal.

## Directory Structure
```
.
├── app.py                # Main Gradio application script
├── test_app.py           # Quick verification script for the app functions
├── requirements.txt      # Python dependencies
├── src/                  # Core source code
│   ├── models/           # Architectures and model wrapper classes
│   │   ├── dehazeformer_wrapper.py
│   │   ├── unetpp_wrapper.py & unetpp_model.py
│   │   ├── wavelet_unet_wrapper.py & wavelet_unet_model.py
│   │   ├── wavelet_ops.py
│   │   └── dummy.py      # Baseline/Dummy (CLAHE-based) approach
│   └── utils/            # Utility functions for processing
│       ├── io.py         # Image load/save utilities
│       ├── postprocess.py# Color correction, CLAHE, and gamma adjustment
│       └── tiling.py     # Tiled processing functions for memory safety
├── weights/              # Pre-trained model checkpoints (e.g., .pth files)
├── data/                 # Directory for storing input datasets
├── outputs/              # Directory for generated results
└── hazing.ipynb          # Jupyter notebook for exploratory data analysis
```

## Installation

1. **Clone the Repository** (or navigate to the project directory):
   ```bash
   cd Data-image-hazing-main
   ```

2. **Create a Virtual Environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have compatible PyTorch/CUDA versions installed if you intend to run models on GPU.*

## Usage

### 1. Web Application (Gradio)
To launch the interactive UI, simply run:
```bash
python app.py
```
This will open a local server (usually accessible at `http://127.0.0.1:7860/`). 
- **Mode 1**: Upload a hazy image to see predictions from DehazeFormer and UNetPP.
- **Mode 2**: Upload a clean image to add synthetic haze and evaluate model performance dynamically.

### 2. Command Line Inference
Process a single image via `src/infer.py`:
```bash
python -m src.infer --input path/to/hazy.png --output path/to/clean.png --model dehazeformer --variant s --ckpt weights/dehazeformer_indoor_s.pth --device cuda --post
```
*Options:* 
- `--model`: `dehazeformer` or `wavelet-unet`
- `--post`: Enables post-processing (White balance, Gamma, CLAHE)

### 3. Batch Evaluation
Evaluate a model on a dataset using `src/eval.py`:
```bash
python -m src.eval --hazy_dir path/to/hazy/ --gt_dir path/to/clear/ --out_dir outputs_eval/ --model dehazeformer --variant s --ckpt weights/dehazeformer_indoor_s.pth --device cuda
```
This script will process all images, save the outputs, and generate a `metrics.csv` containing PSNR and SSIM scores.

---
**Team**: The Outliers
