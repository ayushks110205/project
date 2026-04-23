# =============================================================================
# pipeline.py  –  End-to-End Two-Stage Road Extraction + Inpainting Pipeline
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  FULL PIPELINE                                                           │
# │                                                                          │
# │  INPUT  : Raw satellite image (B, 3, H, W) RGB                          │
# │                                                                          │
# │  STAGE 1: DeepLabV3+ ResNet34                                            │
# │           Satellite Image → Binary Road Mask                             │
# │           (B, 3, 512, 512) → (B, 1, 512, 512)                           │
# │                                                                          │
# │  [Optional]: Provide or auto-detect hole mask                            │
# │                                                                          │
# │  STAGE 2: Partial Conv U-Net                                             │
# │           Incomplete Road Mask → Complete Road Mask                      │
# │           (B, 2, 512, 512) → (B, 1, 512, 512)                           │
# │                                                                          │
# │  OUTPUT : Completed road mask + 4-panel visualisation                   │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   from pipeline import RoadInpaintingPipeline
#   pipe   = RoadInpaintingPipeline(stage1_path, stage2_path)
#   result = pipe.run(image_path, hole_mask_path=None)
#
# Or from CLI:
#   python pipeline.py --image sat.jpg --s1 stage1.pth --s2 stage2.pth
# =============================================================================

import os
import argparse
import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stage 1 imports
from models  import get_road_model
from dataset import val_transform      # Stage 1 image pre-processing

# Stage 2 imports
from inpainting_model  import get_inpainting_model
from infer_inpainting  import auto_detect_holes

# ImageNet normalisation (must match dataset.py val_transform)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Image Pre-processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_satellite(image_path: str, size: int = 512) -> tuple:
    """
    Load and pre-process a satellite image for Stage 1.

    Returns:
        tensor     : (1, 3, 512, 512) float32 — normalised, ready for model
        vis_img    : (512, 512, 3) float32 in [0,1] — for visualisation
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0

    vis_img = img_rgb.copy()

    # Apply ImageNet normalisation
    normalised = (img_rgb - _MEAN) / _STD   # (512, 512, 3)

    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.float(), vis_img


def load_hole_mask(mask_path: str, size: int = 512) -> np.ndarray:
    """
    Load an explicit hole mask from file.
    Convention: white (255) = valid/known, black (0) = hole.

    Returns:
        hole_mask : (H, W) float32 — 1=valid, 0=hole
    """
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.where(raw >= 128, 1.0, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_figure(vis_img:      np.ndarray,
                         stage1_mask:  np.ndarray,
                         corrupted:    np.ndarray,
                         stage2_mask:  np.ndarray,
                         save_path:    str):
    """
    4-panel pipeline visualisation:
        1. Input Satellite
        2. Stage 1 Road Mask  (raw extraction)
        3. Corrupted Mask     (with holes)
        4. Stage 2 Completed  (inpainted)
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("End-to-End Road Extraction + Inpainting Pipeline",
                 fontsize=14, fontweight='bold', y=1.01)

    axes[0].imshow(vis_img)
    axes[0].set_title("Input Satellite", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(stage1_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Stage 1: Road Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(corrupted, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Corrupted (with holes)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(stage2_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Stage 2: Completed Mask", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼️  Pipeline figure saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Class
# ─────────────────────────────────────────────────────────────────────────────

class RoadInpaintingPipeline:
    """
    End-to-end two-stage pipeline: satellite image → complete road mask.

    Args:
        stage1_path : path to Stage 1 DeepLabV3+ weights (.pth)
        stage2_path : path to Stage 2 Partial Conv U-Net weights (.pth)
        device      : torch.device, or None for auto-detect
        threshold1  : Stage 1 binarisation threshold (default 0.5)
        threshold2  : Stage 2 binarisation threshold (default 0.5)
    """

    def __init__(self,
                 stage1_path: str,
                 stage2_path: str,
                 device:      torch.device = None,
                 threshold1:  float = 0.5,
                 threshold2:  float = 0.5):

        self.device     = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold1 = threshold1
        self.threshold2 = threshold2

        print(f"🚀 Loading pipeline on: {self.device}")

        # ── Stage 1: DeepLabV3+ ResNet34 ──────────────────────────────────────
        self.stage1 = get_road_model().to(self.device)
        self.stage1.load_state_dict(
            torch.load(stage1_path, map_location=self.device))
        self.stage1.eval()
        print(f"  ✅ Stage 1 loaded: {stage1_path}")

        # ── Stage 2: Partial Conv U-Net ────────────────────────────────────────
        self.stage2 = get_inpainting_model(base_channels=64).to(self.device)
        self.stage2.load_state_dict(
            torch.load(stage2_path, map_location=self.device))
        self.stage2.eval()
        print(f"  ✅ Stage 2 loaded: {stage2_path}")

    def run(self,
            image_path:      str,
            hole_mask_path:  str   = None,
            min_hole_frac:   float = 0.05,
            save_dir:        str   = '/kaggle/working/results/pipeline',
            save_figure:     bool  = True) -> dict:
        """
        Run the full two-stage pipeline on a satellite image.

        Args:
            image_path     : path to satellite image (JPG/PNG)
            hole_mask_path : optional path to explicit hole mask
                             (white=valid, black=hole).
                             If None, auto-detect from Stage 1 output.
            min_hole_frac  : minimum area fraction for auto-detection
            save_dir       : output directory for results
            save_figure    : whether to save the 4-panel figure

        Returns:
            dict with keys:
                'stage1_mask'   — (512, 512) float32 Stage 1 road mask
                'hole_mask'     — (512, 512) float32 hole mask used
                'corrupted'     — (512, 512) float32 corrupted mask
                'stage2_mask'   — (512, 512) float32 completed road mask
                'output_path'   — path of saved completed mask PNG
        """
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # ── Step 1: Load satellite image ──────────────────────────────────────
        img_tensor, vis_img = load_satellite(image_path, size=512)
        # img_tensor : (1, 3, 512, 512)
        # vis_img    : (512, 512, 3) float32

        img_tensor = img_tensor.to(self.device)

        # ── Step 2: Stage 1 — Extract road mask ───────────────────────────────
        with torch.no_grad():
            with autocast():
                stage1_logits = self.stage1(img_tensor)   # (1, 1, 512, 512)

        stage1_prob = torch.sigmoid(stage1_logits).squeeze().cpu().numpy()
        # (512, 512) float32

        stage1_mask = (stage1_prob > self.threshold1).astype(np.float32)
        # (512, 512) float32 binary road mask

        print(f"📍 Stage 1 complete | Road coverage: "
              f"{stage1_mask.mean()*100:.1f}%")

        # ── Step 3: Determine hole mask ───────────────────────────────────────
        if hole_mask_path is not None:
            hole_mask = load_hole_mask(hole_mask_path, size=512)
            print(f"🗺️  Hole mask loaded from: {hole_mask_path}")
        else:
            hole_mask = auto_detect_holes(stage1_mask, min_frac=min_hole_frac)

        # ── Step 4: Build corrupted mask ──────────────────────────────────────
        corrupted = stage1_mask * hole_mask   # (512, 512)

        # ── Step 5: Stage 2 — Inpaint holes ───────────────────────────────────
        # Model input: (1, 2, 512, 512) = [corrupted_mask, hole_mask]
        c_t = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).to(self.device)
        h_t = torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        # c_t : (1, 1, 512, 512)
        # h_t : (1, 1, 512, 512)

        with torch.no_grad():
            with autocast():
                stage2_pred = self.stage2(c_t, h_t)   # (1, 1, 512, 512)

        stage2_prob = stage2_pred.squeeze().cpu().numpy()   # (512, 512)
        stage2_mask = (stage2_prob > self.threshold2).astype(np.float32)

        # Preserve known-good regions from Stage 1 in final output
        final_mask = stage2_mask * (1 - hole_mask) + stage1_mask * hole_mask
        # (512, 512) float32

        print(f"📍 Stage 2 complete | Final coverage: "
              f"{final_mask.mean()*100:.1f}%")

        # ── Step 6: Save outputs ──────────────────────────────────────────────
        out_name    = f"pipeline_{ts}_completed.png"
        output_path = os.path.join(save_dir, out_name)
        cv2.imwrite(output_path, (final_mask * 255).astype(np.uint8))
        print(f"💾 Complete road mask saved: {output_path}")

        if save_figure:
            fig_path = os.path.join(save_dir, f"pipeline_{ts}_figure.png")
            save_pipeline_figure(vis_img, stage1_mask, corrupted, final_mask,
                                 save_path=fig_path)

        return {
            'stage1_mask':  stage1_mask,
            'hole_mask':    hole_mask,
            'corrupted':    corrupted,
            'stage2_mask':  final_mask,
            'output_path':  output_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='End-to-End Road Extraction + Inpainting Pipeline')
    parser.add_argument('--image',  required=True, help='Satellite image path')
    parser.add_argument('--s1',     required=True, help='Stage 1 model .pth path')
    parser.add_argument('--s2',     required=True, help='Stage 2 model .pth path')
    parser.add_argument('--hole',   default=None,  help='Optional hole mask path')
    parser.add_argument('--outdir', default='/kaggle/working/results/pipeline')
    parser.add_argument('--t1',     type=float, default=0.5, help='Stage 1 threshold')
    parser.add_argument('--t2',     type=float, default=0.5, help='Stage 2 threshold')
    args = parser.parse_args()

    pipe   = RoadInpaintingPipeline(args.s1, args.s2,
                                    threshold1=args.t1, threshold2=args.t2)
    result = pipe.run(args.image, hole_mask_path=args.hole, save_dir=args.outdir)
    print(f"\n✅ Pipeline complete → {result['output_path']}")
