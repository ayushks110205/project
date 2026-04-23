# =============================================================================
# vizualize_road.py  –  5-Panel Road Prediction Visualiser
# =============================================================================
# KEY CHANGES vs v1:
#   • 5th panel added: TP/FP/FN colour overlay on satellite image
#       – TP (true positive roads)  → green
#       – FP (false positive roads) → red
#       – FN (false negative roads) → blue
#       – TN (background correct)   → original image (translucent)
#   • IoU and Dice score printed in each figure's suptitle
#   • Loops through 5 RANDOM val images (not hardcoded indices)
#   • Saves each result with ISO timestamp in filename
#   • Reads from the same val split used in training and evaluation
# =============================================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # Prevents OpenMP crashes on some systems

import random
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — safe for Colab & headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import autocast

# ── Local imports ─────────────────────────────────────────────────────────────
from dataset import get_road_splits, val_transform
from models import get_road_model


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Helper: un-normalise an ImageNet-normalised tensor to RGB array
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised image tensor [3, H, W] → uint8 RGB numpy [H, W, 3].
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()   # [H, W, 3]
    img = (_STD * img + _MEAN)                         # un-normalise
    img = np.clip(img, 0.0, 1.0)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Helper: build TP/FP/FN colour overlay
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay(rgb_img: np.ndarray, pred_binary: np.ndarray,
                  gt_binary: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend TP/FP/FN colour masks onto the satellite image.

    Colour coding:
        Green  (0, 200, 0)   → True Positive  (model correctly found road)
        Red    (220, 0, 0)   → False Positive (model hallucinated a road)
        Blue   (0, 0, 220)   → False Negative (model missed a real road)

    Args:
        rgb_img    : float [H, W, 3] in [0, 1]
        pred_binary: uint8 [H, W] in {0, 1}
        gt_binary  : uint8 [H, W] in {0, 1}
        alpha      : opacity of the colour mask [0=invisible, 1=opaque]

    Returns:
        float [H, W, 3] composite image, clipped to [0, 1]
    """
    overlay = rgb_img.copy()
    p = pred_binary.astype(bool)
    g = gt_binary.astype(bool)

    tp = p & g
    fp = p & ~g
    fn = ~p & g

    # Apply semi-transparent colours
    overlay[tp] = (1 - alpha) * overlay[tp] + alpha * np.array([0.0, 0.78, 0.0])
    overlay[fp] = (1 - alpha) * overlay[fp] + alpha * np.array([0.86, 0.0, 0.0])
    overlay[fn] = (1 - alpha) * overlay[fn] + alpha * np.array([0.0, 0.0, 0.86])

    return np.clip(overlay, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Metric helpers (self-contained so this file is importable alone)
# ─────────────────────────────────────────────────────────────────────────────

def _iou_dice(pred_binary: np.ndarray, gt_binary: np.ndarray):
    eps = 1e-6
    tp  = np.logical_and(pred_binary, gt_binary).sum()
    fp  = np.logical_and(pred_binary, ~gt_binary.astype(bool)).sum()
    fn  = np.logical_and(~pred_binary.astype(bool), gt_binary).sum()
    iou  = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(iou), float(dice)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Main Visualisation Function
# ─────────────────────────────────────────────────────────────────────────────

def visualise_road(model_path: str,
                   image_dir:  str = 'datasets/train',
                   mask_dir:   str = 'datasets/train',
                   val_ratio:  float = 0.20,
                   n_samples:  int   = 5,
                   threshold:  float = 0.50,
                   save_dir:   str   = 'results'):
    """
    Render 5-panel diagnostic visualisations for n_samples random val images.

    Panels:
        1. Input Satellite Image
        2. Ground Truth Mask
        3. Confidence Heatmap  (magma colormap)
        4. Binary Prediction   (thresholded)
        5. TP/FP/FN Overlay    (green/red/blue on satellite)

    Args:
        model_path : path to trained model weights
        image_dir  : dataset folder
        mask_dir   : dataset folder
        val_ratio  : must match value used during training
        n_samples  : number of random val images to visualise
        threshold  : binarisation threshold
        save_dir   : output folder for saved figures
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Visualisation | Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = get_road_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    # ── Load val split ────────────────────────────────────────────────────────
    _, val_ds = get_road_splits(image_dir, mask_dir, val_ratio=val_ratio)

    n_available = len(val_ds)
    n_samples   = min(n_samples, n_available)
    selected    = random.sample(range(n_available), n_samples)
    print(f"🎲 Randomly selected {n_samples} val images: indices {selected}\n")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Per-image loop ────────────────────────────────────────────────────────
    for sample_idx, val_idx in enumerate(selected, start=1):
        img_tensor, mask_tensor = val_ds[val_idx]

        # Inference
        input_batch = img_tensor.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            with autocast():
                output = model(input_batch)
        prob_map    = torch.sigmoid(output).squeeze().cpu().numpy()   # [H, W]
        binary_pred = (prob_map > threshold).astype(np.uint8)
        gt_binary   = mask_tensor.cpu().numpy().astype(np.uint8)

        # Metrics for title
        iou, dice = _iou_dice(binary_pred, gt_binary)

        # Visual prep
        rgb_img = tensor_to_rgb(img_tensor)            # [H, W, 3] float
        overlay = build_overlay(rgb_img, binary_pred, gt_binary, alpha=0.45)

        # ── 5-panel figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(
            f"Val image #{val_idx}  |  IoU = {iou:.4f}  |  Dice = {dice:.4f}",
            fontsize=15, fontweight='bold', y=1.01
        )

        # Panel 1: Satellite input
        axes[0].imshow(rgb_img)
        axes[0].set_title("Input Satellite", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Ground truth
        axes[1].imshow(gt_binary, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Confidence heatmap
        im = axes[2].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
        axes[2].set_title("Confidence Heatmap", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')

        # Panel 4: Binary prediction
        axes[3].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title("Binary Prediction", fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # Panel 5: TP/FP/FN overlay with colour legend
        axes[4].imshow(overlay)
        axes[4].set_title("TP / FP / FN Overlay", fontsize=12, fontweight='bold')
        axes[4].axis('off')

        # Legend patches for panel 5
        legend_handles = [
            mpatches.Patch(color=(0.0, 0.78, 0.0), label='TP (correct road)'),
            mpatches.Patch(color=(0.86, 0.0, 0.0), label='FP (false road)'),
            mpatches.Patch(color=(0.0, 0.0, 0.86), label='FN (missed road)'),
        ]
        axes[4].legend(
            handles=legend_handles,
            loc='lower left',
            fontsize=8,
            framealpha=0.75,
        )

        plt.tight_layout()

        # ── Save ──────────────────────────────────────────────────────────────
        fname    = f"road_viz_{timestamp}_sample{sample_idx:02d}_val{val_idx}.png"
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  [{sample_idx}/{n_samples}] IoU={iou:.4f} | Dice={dice:.4f} → {out_path}")

    print(f"\n✅ All {n_samples} visualisations saved to '{save_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DRIVE_PATH = '/content/drive/MyDrive/datasets/road_model_latest.pth'
    LOCAL_PATH = 'best_model.pth'

    model_path = DRIVE_PATH if os.path.exists(DRIVE_PATH) else LOCAL_PATH

    visualise_road(
        model_path=model_path,
        n_samples=5,
        threshold=0.5,
        save_dir='results',
    )