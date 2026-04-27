# =============================================================================
# visualize_building.py  –  Stage 4 Building Detection 7-Panel Visualisation
# =============================================================================
# Panels (640×640):
#   1. Input satellite      — original RGB tile
#   2. Ground truth mask    — white=building, black=background
#   3. Raw predicted mask   — direct threshold
#   4. Postprocessed mask   — morphological open+close
#   5. Confidence heatmap   — sigmoid probability in 'magma' colormap
#   6. Boundary comparison  — GT=white, pred=cyan, overlap=green
#   7. Instance map         — each connected building in a unique colour
#
# Saves 4 categories × n figures each:
#   best / worst / mid-high / mid-low
#
# Output: /kaggle/working/results/buildings/{best,worst,mid_high,mid_low}/
#
# Usage:
#   from visualize_building import run_visualization
#   run_visualization()
# =============================================================================

import os
import random
import numpy as np
import torch
import torch.amp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from scipy import ndimage as ndi
import cv2

from dataset import MassachusettsBuildingDataset, building_val_transform
from models import get_building_model
from evaluate_building import tta_predict

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATASET_BASE = '/kaggle/input/datasets/balraj98/massachusetts-buildings-dataset'
IMAGE_DIR    = f'{DATASET_BASE}/tiff/val'
MASK_DIR     = f'{DATASET_BASE}/tiff/val_labels'
MODEL_PATH   = '/kaggle/working/building_model_best.pth'
SAVE_DIR     = '/kaggle/working/results/buildings'
IMG_SIZE     = 640

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Section 1 ▸ Post-processing
# =============================================================================

def postprocess_mask(binary_mask):
    """Morphological open+close to clean small FP blobs and fill interior holes."""
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  OPEN_KERNEL)
    closed = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return closed


# =============================================================================
# Section 2 ▸ Instance Map
# =============================================================================

def make_instance_map(binary_mask):
    """Label each connected building with a unique colour."""
    labelled, n = ndi.label(binary_mask > 0)
    rng     = random.Random(42)
    palette = [(0, 0, 0)]
    for _ in range(n):
        h, s, v = rng.random(), rng.uniform(0.5, 1.0), rng.uniform(0.6, 1.0)
        palette.append(tuple(int(c * 255) for c in mcolors.hsv_to_rgb([h, s, v])))
    H, W   = labelled.shape
    result = np.zeros((H, W, 3), dtype=np.uint8)
    for lbl in range(n + 1):
        result[labelled == lbl] = palette[lbl]
    return result


# =============================================================================
# Section 3 ▸ Boundary Comparison Panel
# =============================================================================

def make_boundary_panel(gt_mask, pred_mask):
    """White=GT only, Cyan=Pred only, Green=overlap."""
    gt_edge   = cv2.Canny(gt_mask,   100, 200)
    pred_edge = cv2.Canny(pred_mask, 100, 200)
    panel = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    panel[(gt_edge > 0) & (pred_edge > 0)]              = [0, 255, 0]
    panel[(gt_edge > 0) & ~(pred_edge > 0)]             = [255, 255, 255]
    panel[(pred_edge > 0) & ~(gt_edge > 0)]             = [0, 255, 255]
    return panel


# =============================================================================
# Section 4 ▸ 7-Panel Figure
# =============================================================================

def save_7panel(image_rgb, gt_mask, raw_pred, post_pred, confidence,
                save_path, title=''):
    boundary_panel = make_boundary_panel(gt_mask, post_pred)
    instance_panel = make_instance_map(post_pred)

    fig, axes = plt.subplots(1, 7, figsize=(49, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    panels = [
        (image_rgb,      'Input Satellite',    'viridis', False),
        (gt_mask,        'Ground Truth',       'gray',    True),
        (raw_pred,       'Raw Prediction',     'gray',    True),
        (post_pred,      'Postprocessed',      'gray',    True),
        (confidence,     'Confidence Heatmap', 'magma',   False),
        (boundary_panel, 'Boundary Comparison', None,     False),
        (instance_panel, 'Instance Map',        None,     False),
    ]

    for ax, (data, ptitle, cmap, is_binary) in zip(axes, panels):
        if cmap is not None:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=255 if is_binary else None)
        else:
            ax.imshow(data)
        ax.set_title(ptitle, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Section 5 ▸ Main Visualisation Loop
# =============================================================================

def run_visualization(model_path=MODEL_PATH, image_dir=IMAGE_DIR,
                      mask_dir=MASK_DIR, save_dir=SAVE_DIR,
                      use_tta=False, n_per_cat=3):
    """
    Evaluate all val images, rank by IoU, generate 7-panel figures for
    4 categories: best / worst / mid-high / mid-low.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_building_model().to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"✅ Model loaded: {model_path}")

    # ── Val image list ─────────────────────────────────────────────────────────
    exts = ('.tiff', '.tif', '.png', '.jpg')
    val_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"📂 Val images: {len(val_files)}")
    print(f"   Generating {n_per_cat} figures per category "
          f"(best/worst/mid-high/mid-low) | TTA: {use_tta}\n")

    # ── Per-image inference ────────────────────────────────────────────────────
    results = []
    for img_name in val_files:
        # Load via PIL (no GeoTIFF warnings)
        pil_img   = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
        img_rgb   = np.array(pil_img, dtype=np.uint8)    # (H, W, 3)

        mask_path = os.path.join(mask_dir, img_name)
        if not os.path.exists(mask_path):
            continue
        pil_mask  = Image.open(mask_path).convert('L')
        mask_raw  = np.array(pil_mask, dtype=np.uint8)
        gt_bin    = np.where(mask_raw >= 128, np.uint8(255), np.uint8(0))

        # Resize display image to IMG_SIZE
        img_disp  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
        gt_disp   = cv2.resize(gt_bin, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST)

        # Build tensor for model
        aug = building_val_transform(
            image=img_rgb,
            mask=(gt_bin / 255).astype(np.float32),
            edge_mask=np.zeros(mask_raw.shape, dtype=np.float32),
            dist_map=np.zeros(mask_raw.shape, dtype=np.float32),
        )
        img_tensor = aug['image'].unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            if use_tta:
                prob = tta_predict(model, img_tensor)
            else:
                with torch.amp.autocast('cuda'):
                    prob = torch.sigmoid(model(img_tensor))

        prob_np  = prob.squeeze().cpu().numpy()                     # (640, 640)
        pred_bin = np.where(prob_np > 0.5, np.uint8(255), np.uint8(0))

        inter = ((gt_disp > 0) & (pred_bin > 0)).sum()
        union = ((gt_disp > 0) | (pred_bin > 0)).sum()
        iou   = float(inter + 1e-6) / float(union + 1e-6)

        results.append({
            'iou':      iou,
            'img_name': img_name,
            'img_rgb':  img_disp,
            'gt_bin':   gt_disp,
            'pred_bin': pred_bin,
            'prob_np':  prob_np,
        })
        print(f"  {img_name} → IoU={iou:.4f}")

    results.sort(key=lambda x: x['iou'], reverse=True)
    N = len(results)

    categories = {
        'best':     results[:n_per_cat],
        'worst':    results[max(0, N - n_per_cat):],
        'mid_high': results[N//4 : N//4 + n_per_cat],
        'mid_low':  results[3*N//4 : 3*N//4 + n_per_cat],
    }

    # ── Save figures ───────────────────────────────────────────────────────────
    total = 0
    for cat_name, cat_results in categories.items():
        cat_dir = os.path.join(save_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        for r in cat_results:
            post      = postprocess_mask(r['pred_bin'])
            base_name = os.path.splitext(r['img_name'])[0]
            save_path = os.path.join(cat_dir, f"{base_name}_{cat_name}.png")
            title     = (f"{cat_name.upper()} | {r['img_name']} | "
                         f"IoU={r['iou']:.4f} | "
                         f"{'TTA' if use_tta else 'Standard'}")

            save_7panel(r['img_rgb'], r['gt_bin'], r['pred_bin'],
                        post, r['prob_np'], save_path, title)
            total += 1
            print(f"   💾 [{cat_name}] {save_path}  IoU={r['iou']:.4f}")

    print(f"\n✅ Done — {total} figures saved to {save_dir}/")
    print("   Subdirs: best/ | worst/ | mid_high/ | mid_low/")


if __name__ == '__main__':
    run_visualization()
