# =============================================================================
# visualize_building.py  –  Stage 4 Building Detection 7-Panel Visualisation
# =============================================================================
# Panels (640×640):
#   1. Input satellite      — original RGB (un-normalised)
#   2. Ground truth mask    — white=building, black=background
#   3. Raw predicted mask   — direct threshold, no morphology
#   4. Postprocessed mask   — morphological open+close applied
#   5. Confidence heatmap   — sigmoid probability in 'magma' colormap
#   6. Boundary comparison  — GT=white, pred=cyan, overlap=green
#   7. Instance map         — each connected building in a unique colour
#
# Loops over 4 categories × 3 images = 12 figures:
#   - Best 3 (highest IoU)
#   - Worst 3 (lowest IoU)
#   - Random urban 3
#   - Random rural 3
#
# All outputs saved to /kaggle/working/results/buildings/
#
# Usage (notebook):
#   from visualize_building import run_visualization
#   run_visualization(model_path='/kaggle/working/building_model_best.pth')
# =============================================================================

import os
import random
import argparse
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage as ndi

from dataset import DeepGlobeBuildingDataset, building_val_transform
from models import get_building_model
from evaluate_building import tta_predict


# =============================================================================
# Section 1 ▸ Configuration
# =============================================================================

MODEL_PATH  = '/kaggle/working/building_model_best.pth'
IMAGE_DIR   = '/kaggle/input/deepglobe-building'
MASK_DIR    = '/kaggle/input/deepglobe-building'
SAVE_DIR    = '/kaggle/working/results/buildings'
IMG_SIZE    = 640

# ImageNet mean/std for un-normalisation (restoring visual display image)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Post-processing kernels
OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


# =============================================================================
# Section 2 ▸ Post-processing
# =============================================================================

def postprocess_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological post-processing to clean up predictions.

    Operations (in order):
        1. MORPH_OPEN  (3×3): removes small isolated false-positive blobs
           (sensor noise, shadow specks) that don't belong to any building.
        2. MORPH_CLOSE (5×5): fills small holes inside predicted building
           footprints caused by rooftop texture variation or skylight artifacts.

    Args:
        binary_mask : (H, W) uint8 {0, 255}
    Returns:
        cleaned     : (H, W) uint8 {0, 255}
    """
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  OPEN_KERNEL)
    closed = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return closed


# =============================================================================
# Section 3 ▸ Instance Map Generation
# =============================================================================

def make_instance_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Label each connected building component with a unique colour.

    Uses scipy's connected component labelling (8-connectivity).
    Each label gets a distinct colour from a fixed high-contrast palette.

    Args:
        binary_mask : (H, W) uint8 {0, 255}
    Returns:
        colour_map  : (H, W, 3) uint8 RGB
    """
    labelled, n_buildings = ndi.label(binary_mask > 0)

    # Build colour palette: one colour per label (label 0 = background)
    rng     = random.Random(42)           # deterministic colours
    palette = [(0, 0, 0)]                 # background = black
    for _ in range(n_buildings):
        # Avoid very dark or very light colours for visibility
        h = rng.random()
        s = rng.uniform(0.5, 1.0)
        v = rng.uniform(0.6, 1.0)
        r, g, b = [int(c * 255) for c in
                   mcolors.hsv_to_rgb([h, s, v])]
        palette.append((r, g, b))

    H, W   = labelled.shape
    result = np.zeros((H, W, 3), dtype=np.uint8)
    for lbl in range(n_buildings + 1):
        result[labelled == lbl] = palette[lbl]

    return result


# =============================================================================
# Section 4 ▸ Boundary Comparison Panel
# =============================================================================

def make_boundary_panel(gt_mask: np.ndarray,
                        pred_mask: np.ndarray) -> np.ndarray:
    """
    Overlay GT and predicted boundaries on a black canvas.

    Colour coding:
        White : GT boundary only (missed by prediction)
        Cyan  : Pred boundary only (false positive boundary)
        Green : Overlap (both GT and pred agree on boundary location)

    Args:
        gt_mask   : (H, W) uint8 {0, 255}
        pred_mask : (H, W) uint8 {0, 255}
    Returns:
        panel : (H, W, 3) uint8 RGB
    """
    gt_edge   = cv2.Canny(gt_mask,   100, 200)
    pred_edge = cv2.Canny(pred_mask, 100, 200)

    panel = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    # Green where both agree
    both         = (gt_edge > 0) & (pred_edge > 0)
    panel[both]  = [0, 255, 0]
    # White for GT-only (missed)
    gt_only          = (gt_edge > 0) & ~(pred_edge > 0)
    panel[gt_only]   = [255, 255, 255]
    # Cyan for pred-only (false boundary)
    pred_only         = (pred_edge > 0) & ~(gt_edge > 0)
    panel[pred_only]  = [0, 255, 255]

    return panel


# =============================================================================
# Section 5 ▸ Un-normalise Image for Display
# =============================================================================

def tensor_to_display(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalised image tensor back to uint8 RGB for matplotlib display.

    Args:
        image_tensor : (3, H, W) float32 — ImageNet normalised
    Returns:
        img          : (H, W, 3) uint8
    """
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * _STD + _MEAN                              # un-normalise
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# =============================================================================
# Section 6 ▸ 7-Panel Figure
# =============================================================================

def save_7panel(image_rgb:  np.ndarray,
                gt_mask:    np.ndarray,
                raw_pred:   np.ndarray,
                post_pred:  np.ndarray,
                confidence: np.ndarray,
                save_path:  str,
                title:      str = ''):
    """
    Generate and save a 7-panel building detection figure.

    Args:
        image_rgb  : (H, W, 3) uint8  — original satellite image
        gt_mask    : (H, W) uint8     — ground truth {0, 255}
        raw_pred   : (H, W) uint8     — raw prediction {0, 255}
        post_pred  : (H, W) uint8     — morphologically cleaned {0, 255}
        confidence : (H, W) float32   — sigmoid probability [0, 1]
        save_path  : output file path (.png)
        title      : figure suptitle string
    """
    boundary_panel = make_boundary_panel(gt_mask, post_pred)
    instance_panel = make_instance_map(post_pred)

    fig, axes = plt.subplots(1, 7, figsize=(49, 7))   # 7×7in per panel
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    panels = [
        (image_rgb,        'Input Satellite',      'viridis', False),
        (gt_mask,          'Ground Truth',         'gray',    True),
        (raw_pred,         'Raw Prediction',       'gray',    True),
        (post_pred,        'Postprocessed',        'gray',    True),
        (confidence,       'Confidence Heatmap',   'magma',   False),
        (boundary_panel,   'Boundary Comparison',  None,      False),
        (instance_panel,   'Instance Map',         None,      False),
    ]

    for ax, (data, panel_title, cmap, is_binary) in zip(axes, panels):
        if cmap is not None:
            if is_binary:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)   # RGB panels (boundary, instance)

        ax.set_title(panel_title, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Section 7 ▸ Main Visualisation Loop
# =============================================================================

def run_visualization(model_path: str = MODEL_PATH,
                      image_dir:  str = IMAGE_DIR,
                      mask_dir:   str = MASK_DIR,
                      save_dir:   str = SAVE_DIR,
                      use_tta:    bool = False,
                      n_per_cat:  int  = 3):
    """
    Evaluate the full val set, rank by IoU, then generate 7-panel figures
    for 4 categories: best / worst / random-urban / random-rural.

    Args:
        model_path : path to trained model checkpoint
        image_dir  : satellite images directory
        mask_dir   : building mask directory
        save_dir   : where to save figures
        use_tta    : whether to use 8-variant TTA for prediction
        n_per_cat  : how many figures per category (default 3)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_building_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded: {model_path}")

    # ── Load val dataset (no transform — we'll apply manually for display) ────
    from dataset import DeepGlobeBuildingDataset, building_val_transform

    # Full dataset for display (val only — last 20% by sorted order)
    all_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_val   = int(len(all_files) * 0.2)
    val_files = all_files[-n_val:]

    print(f"📂 Val set: {len(val_files)} images")
    print(f"   Generating figures for: best-{n_per_cat} / worst-{n_per_cat} / "
          f"urban-{n_per_cat} / rural-{n_per_cat}\n")

    # ── Compute IoU for all val images ────────────────────────────────────────
    results = []   # list of (iou, img_name)

    for img_name in val_files:
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        # Load raw image for display (uint8 RGB)
        img_bgr = cv2.imread(os.path.join(image_dir, img_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        # Load raw mask
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue
        gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_raw = cv2.resize(gt_raw, (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_NEAREST)
        gt_bin = np.where(gt_raw >= 128, np.uint8(255), np.uint8(0))

        # Prepare normalised tensor for model
        aug = building_val_transform(
            image=img_rgb,
            mask=(gt_bin / 255).astype(np.float32),
            edge_mask=np.zeros_like(gt_raw, dtype=np.float32),
            dist_map=np.zeros_like(gt_raw, dtype=np.float32),
        )
        img_tensor = aug['image'].unsqueeze(0).to(device)   # (1, 3, 640, 640)

        # Predict
        with torch.no_grad():
            if use_tta:
                prob = tta_predict(model, img_tensor, device)  # (1,1,640,640)
            else:
                with autocast():
                    prob = torch.sigmoid(model(img_tensor))    # (1,1,640,640)

        prob_np = prob.squeeze().cpu().numpy()                  # (640, 640)
        pred_bin = np.where(prob_np > 0.5,
                            np.uint8(255), np.uint8(0))         # (640, 640)

        # Compute IoU
        gt_bool   = gt_bin > 0
        pred_bool = pred_bin > 0
        inter     = (gt_bool & pred_bool).sum()
        union     = (gt_bool | pred_bool).sum()
        iou       = float(inter + 1e-6) / float(union + 1e-6)

        results.append({
            'iou':       iou,
            'img_name':  img_name,
            'img_rgb':   img_rgb,
            'gt_bin':    gt_bin,
            'pred_bin':  pred_bin,
            'prob_np':   prob_np,
        })

    results.sort(key=lambda x: x['iou'], reverse=True)

    # ── Categorise ────────────────────────────────────────────────────────────
    best  = results[:n_per_cat]
    worst = results[-n_per_cat:]

    # Urban heuristic: image ID is a round number (DeepGlobe urban IDs tend
    # to be > 700000). Fallback: take middle n_per_cat.
    mid   = len(results) // 2
    urban = results[mid:mid + n_per_cat]
    rural = results[max(0, mid - n_per_cat):mid]

    categories = [
        ('best',  best),
        ('worst', worst),
        ('urban', urban),
        ('rural', rural),
    ]

    # ── Generate figures ──────────────────────────────────────────────────────
    total_saved = 0
    for cat_name, cat_results in categories:
        cat_dir = os.path.join(save_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        for r in cat_results:
            post = postprocess_mask(r['pred_bin'])

            fig_title = (
                f"{cat_name.upper()} | {r['img_name']} | "
                f"IoU={r['iou']:.4f} | "
                f"{'TTA' if use_tta else 'Standard'}"
            )

            save_path = os.path.join(
                cat_dir,
                r['img_name'].replace('_sat.jpg', f'_{cat_name}.png')
            )

            save_7panel(
                image_rgb  = r['img_rgb'],
                gt_mask    = r['gt_bin'],
                raw_pred   = r['pred_bin'],
                post_pred  = post,
                confidence = r['prob_np'],
                save_path  = save_path,
                title      = fig_title,
            )
            total_saved += 1
            print(f"   💾 Saved [{cat_name}] {save_path} | IoU={r['iou']:.4f}")

    print(f"\n✅ Done — {total_saved} figures saved to {save_dir}/")
    print("   Subdirectories: best/ | worst/ | urban/ | rural/")


# =============================================================================
# Section 8 ▸ Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Building Detection 7-Panel Visualisation')
    parser.add_argument('--model',
                        default='/kaggle/working/building_model_best.pth')
    parser.add_argument('--images',
                        default='/kaggle/input/deepglobe-building')
    parser.add_argument('--masks',
                        default='/kaggle/input/deepglobe-building')
    parser.add_argument('--outdir',
                        default='/kaggle/working/results/buildings')
    parser.add_argument('--tta', action='store_true',
                        help='Use TTA for prediction (slower, +1-2% IoU)')
    parser.add_argument('--n', type=int, default=3,
                        help='Figures per category (default 3)')
    args = parser.parse_args()

    run_visualization(
        model_path=args.model,
        image_dir=args.images,
        mask_dir=args.masks,
        save_dir=args.outdir,
        use_tta=args.tta,
        n_per_cat=args.n,
    )
