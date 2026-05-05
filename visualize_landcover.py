# =============================================================================
# visualize_landcover.py  –  Stage 3: Visual Inspection of Val Predictions
# =============================================================================
# Loads 6 random val images and produces a 4-panel figure per image:
#   Panel 1: Input satellite image (un-normalized)
#   Panel 2: Ground-truth mask  (colour-coded + legend)
#   Panel 3: Predicted mask     (same colour palette + legend)
#   Panel 4: Error map  (white=correct, red=wrong, tinted by mistake type)
# Saves to results/landcover_visual_<timestamp>.png
# =============================================================================

import os
import random
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Subset
import torch.amp

from dataset import get_landcover_splits, DeepGlobeLandCoverDataset
from models import get_landcover_model

# ─────────────────────────────────────────────────────────────────────────────
# Config  — Kaggle paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_BASE = '/kaggle/input/datasets/balraj98/deepglobe-land-cover-classification-dataset'
IMAGE_DIR    = f'{DATASET_BASE}/train'
MASK_DIR     = f'{DATASET_BASE}/train'
# Model weights — try the uploaded 'best-path' dataset first,
# then fall back to /kaggle/working/ if evaluating right after training.
_MODEL_CANDIDATES = [
    '/kaggle/input/datasets/ayushks07/best-path/landcover_best.pth',
    '/kaggle/working/landcover_best.pth',
]
MODEL_PATH   = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])
RESULTS_DIR  = '/kaggle/working/results/landcover_eval'
NUM_SAMPLES  = 6
SEED         = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥  Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# DeepGlobe Standard Colour Palette
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (class_name, R/255, G/255, B/255)
CLASS_NAMES  = DeepGlobeLandCoverDataset.CLASS_NAMES
PALETTE_NORM = np.array([
    [0,   1,   1  ],   # 0 Urban      → #00FFFF  cyan
    [1,   1,   0  ],   # 1 Agriculture → #FFFF00 yellow
    [1,   0,   1  ],   # 2 Rangeland  → #FF00FF  magenta
    [0,   1,   0  ],   # 3 Forest     → #00FF00  green
    [0,   0,   1  ],   # 4 Water      → #0000FF  blue
    [1,   1,   1  ],   # 5 Barren     → #FFFFFF  white
    [0,   0,   0  ],   # 6 Unknown    → #000000  black
], dtype=np.float32)   # (7, 3)  in [0,1]

NUM_CLASSES = 7

def class_ids_to_rgb(id_map: np.ndarray) -> np.ndarray:
    """Convert (H, W) int class-ID map → (H, W, 3) float32 RGB in [0,1]."""
    H, W = id_map.shape
    rgb  = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        rgb[id_map == c] = PALETTE_NORM[c]
    return rgb


def make_legend_patches():
    """Return matplotlib patches for the colour legend."""
    patches = []
    for name, colour in zip(CLASS_NAMES, PALETTE_NORM):
        edge = 'gray' if colour.sum() > 2.5 else 'none'  # white needs border
        patches.append(
            mpatches.Patch(facecolor=colour, edgecolor=edge,
                           linewidth=0.7, label=name)
        )
    return patches


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Build error map (H, W, 3) float32.

    Correct pixels → white (1,1,1).
    Wrong pixels   → colour of the PREDICTED class (tinted red).
    Tinting formula: 0.6 * red + 0.4 * pred_colour  →  stays reddish.
    """
    H, W = gt.shape
    err  = np.ones((H, W, 3), dtype=np.float32)  # start white
    wrong = gt != pred
    if wrong.any():
        pred_colours     = PALETTE_NORM[pred[wrong]]          # (N, 3)
        RED              = np.array([1.0, 0.0, 0.0])
        err[wrong]       = 0.6 * RED + 0.4 * pred_colours
    return err

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet denormalisation (reverses Normalize in val_transform)
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised float tensor → (H, W, 3) uint8 RGB."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * _STD + _MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Data  (val split, no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
_, val_ds = get_landcover_splits(IMAGE_DIR, MASK_DIR, val_ratio=0.2)

random.seed(SEED)
sample_indices = random.sample(range(len(val_ds)), min(NUM_SAMPLES, len(val_ds)))
subset = Subset(val_ds, sample_indices)
loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
model = get_landcover_model().to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# Handle both full checkpoint ('model_state_dict') and plain state_dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
elif isinstance(ckpt, dict) and 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"✅ Model loaded  (best mIoU={ckpt.get('best_miou', '?') if isinstance(ckpt, dict) else '?'})")

# ─────────────────────────────────────────────────────────────────────────────
# Inference pass  (collect results before plotting)
# ─────────────────────────────────────────────────────────────────────────────
results = []   # list of dicts: {img_rgb, gt_id, pred_id, sample_miou}

with torch.no_grad():
    for images, masks in loader:
        # images: (1, 3, 512, 512)   masks: (1, 512, 512) int64
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()

        with torch.amp.autocast('cuda'):
            logits = model(images)
            # logits: (1, 7, 512, 512)

        preds = logits.argmax(dim=1)   # (1, 512, 512)

        # Per-sample mIoU
        gt_flat   = masks[0].cpu().numpy().flatten()
        pred_flat = preds[0].cpu().numpy().flatten()

        iou_per_class = []
        for c in range(NUM_CLASSES):
            tp  = ((gt_flat == c) & (pred_flat == c)).sum()
            fp  = ((gt_flat != c) & (pred_flat == c)).sum()
            fn  = ((gt_flat == c) & (pred_flat != c)).sum()
            den = tp + fp + fn
            if den > 0:
                iou_per_class.append(tp / den)
        sample_miou = float(np.mean(iou_per_class)) if iou_per_class else 0.0

        results.append({
            'img_rgb':     denormalize(images[0]),        # (H, W, 3) uint8
            'gt_id':       masks[0].cpu().numpy(),         # (H, W) int
            'pred_id':     preds[0].cpu().numpy(),         # (H, W) int
            'sample_miou': sample_miou,
        })

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
n      = len(results)
fig, axes = plt.subplots(n, 4, figsize=(22, 5 * n))
if n == 1:
    axes = axes[np.newaxis, :]  # keep 2-D indexing consistent

legend_patches = make_legend_patches()

# Overall mIoU from checkpoint
ckpt_miou = ckpt.get('best_miou', None)
sup_title  = (f"Land Cover Predictions  |  Best val mIoU = {ckpt_miou:.4f}"
              if ckpt_miou is not None else "Land Cover Predictions")
fig.suptitle(sup_title, fontsize=16, fontweight='bold', y=1.005)

PANEL_TITLES = ['Satellite Image', 'Ground Truth', 'Prediction', 'Error Map']
PANEL_AXES_OFF = [False, False, False, False]

for row, res in enumerate(results):
    img_rgb    = res['img_rgb']                              # (512,512,3) uint8
    gt_rgb     = class_ids_to_rgb(res['gt_id'])             # (512,512,3) float
    pred_rgb   = class_ids_to_rgb(res['pred_id'])           # (512,512,3) float
    err_rgb    = make_error_map(res['gt_id'], res['pred_id'])# (512,512,3) float
    miou_str   = f"mIoU={res['sample_miou']:.3f}"

    panels = [img_rgb, gt_rgb, pred_rgb, err_rgb]
    for col, (ax, panel, title) in enumerate(
            zip(axes[row], panels, PANEL_TITLES)):
        ax.imshow(panel)
        ax.set_title(title if col != 2
                     else f'Prediction  ({miou_str})',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

        # Legend on GT and Prediction panels
        if col in (1, 2):
            ax.legend(
                handles=legend_patches,
                loc='lower right',
                fontsize=7,
                framealpha=0.75,
                ncol=2,
            )

plt.tight_layout()
timestamp   = time.strftime('%Y%m%d_%H%M%S')
save_path   = os.path.join(RESULTS_DIR,
                           f'landcover_visual_{timestamp}.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"💾 Visualization saved to: {save_path}")
