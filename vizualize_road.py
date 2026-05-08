# =============================================================================
# vizualize_road.py  –  8-Panel Road Prediction Visualiser  (Tier 1 integrated)
# =============================================================================
# Panels:
#   1. Input Satellite           4. Binary Prediction
#   2. Ground Truth Mask         5. TP/FP/FN Overlay
#   3. Confidence Heatmap        6. Width Heatmap  (Module 1)
#                                7. Surface Overlay (Module 2)
#                                8. Route Overlay   (Module 3)
#
# CHANGE HISTORY
#   v1 → 5-panel (satellite / GT / heatmap / binary / TP-FP-FN)
#   v2 → 8-panel: adds Tier 1 width, surface, route (with graceful fallback)
#        Title extended: + mean road width (m) + dominant surface type
# =============================================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.amp import autocast

# ── Local imports ─────────────────────────────────────────────────────────────
from dataset import get_road_splits, val_transform
from models  import get_road_model

# ── Tier 1 modules (optional — graceful fallback if unavailable) ──────────────
try:
    from road_width          import RoadWidthEstimator
    from road_type_classifier import RoadTypeClassifier
    from road_router          import RoadRouter, _pick_route_endpoints
    _TIER1_OK = True
except ImportError as _tier1_err:
    _TIER1_OK = False
    print(f"[Tier1 WARNING] Tier 1 modules not importable: {_tier1_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Helper: un-normalise an ImageNet-normalised tensor to RGB array
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor [3, H, W] → float32 RGB [H, W, 3] in [0,1]."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (_STD * img + _MEAN)
    return np.clip(img, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Helper: build TP/FP/FN colour overlay
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay(rgb_img: np.ndarray, pred_binary: np.ndarray,
                  gt_binary: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend TP/FP/FN colour masks onto satellite image (float [0,1])."""
    overlay = rgb_img.copy()
    p = pred_binary.astype(bool)
    g = gt_binary.astype(bool)

    tp = p & g
    fp = p & ~g
    fn = ~p & g

    overlay[tp] = (1 - alpha) * overlay[tp] + alpha * np.array([0.0, 0.78, 0.0])
    overlay[fp] = (1 - alpha) * overlay[fp] + alpha * np.array([0.86, 0.0, 0.0])
    overlay[fn] = (1 - alpha) * overlay[fn] + alpha * np.array([0.0, 0.0, 0.86])

    return np.clip(overlay, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Metric helpers
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
# Section 4 ▸ Tier 1 panel helpers (each returns a uint8 RGB or float image)
# ─────────────────────────────────────────────────────────────────────────────

def _black_panel(H: int, W: int, msg: str, ax) -> None:
    """Fill *ax* with a black panel and centred white text."""
    ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    ax.text(W / 2, H / 2, msg,
            color='white', fontsize=9, ha='center', va='center',
            wrap=True, bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='#333333', alpha=0.8))
    ax.axis('off')


def _run_tier1(pred_mask_u8: np.ndarray,
               sat_rgb_u8:   np.ndarray,
               H: int, W: int):
    """
    Run all three Tier 1 modules on *pred_mask_u8*.

    Returns:
        width_result  : RoadWidthResult or None
        type_result   : dict or None
        route_overlay : (H, W, 3) uint8 or None
        mean_width_m  : float
        dominant_surf : str
    """
    width_result  = None
    type_result   = None
    route_overlay = None
    mean_width_m  = 0.0
    dominant_surf = 'n/a'

    # Module 1 — Width
    try:
        est          = RoadWidthEstimator()
        width_result = est.analyse(pred_mask_u8)
        if not width_result.is_empty:
            mean_width_m = width_result.summary_stats['mean_m']
    except Exception as e:
        print(f"[Tier1 WARNING] Module 1 (width): {e}")

    # Module 2 — Surface
    try:
        clf         = RoadTypeClassifier()
        type_result = clf.predict(sat_rgb_u8, pred_mask_u8,
                                  width_result=width_result)
        if type_result and not type_result.get('is_empty', True):
            dominant_surf = type_result['summary']['dominant_type']
    except Exception as e:
        print(f"[Tier1 WARNING] Module 2 (surface): {e}")

    # Module 3 — Route
    try:
        if (width_result is not None and not width_result.is_empty
                and type_result is not None):
            src, dst = _pick_route_endpoints(width_result.skeleton)
            if src is not None and dst is not None:
                router = RoadRouter(pred_mask_u8, width_result, type_result)
                route  = router.find_route(src, dst, vehicle_type='car',
                                           satellite_rgb=sat_rgb_u8)
                route_overlay = route.route_overlay_rgb
    except Exception as e:
        print(f"[Tier1 WARNING] Module 3 (router): {e}")

    return width_result, type_result, route_overlay, mean_width_m, dominant_surf


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Main Visualisation Function
# ─────────────────────────────────────────────────────────────────────────────

def visualise_road(model_path: str,
                   image_dir:  str   = '/kaggle/input/datasets/balraj98/deepglobe-road-extraction-dataset/train',
                   mask_dir:   str   = '/kaggle/input/datasets/balraj98/deepglobe-road-extraction-dataset/train',
                   val_ratio:  float = 0.20,
                   n_samples:  int   = 5,
                   threshold:  float = 0.50,
                   save_dir:   str   = '/kaggle/working/results'):
    """
    Render 8-panel diagnostic visualisations for n_samples random val images.

    Panels:
        1. Input Satellite          5. TP/FP/FN Overlay
        2. Ground Truth Mask        6. Width Heatmap      [Tier 1 – M1]
        3. Confidence Heatmap       7. Surface Overlay    [Tier 1 – M2]
        4. Binary Prediction        8. Route Overlay      [Tier 1 – M3]

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
    print(f"🔍 Visualisation | Device: {device} | Tier1={'ON' if _TIER1_OK else 'OFF'}")

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
    print(f"🎲 Selected {n_samples} val images: indices {selected}\n")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Per-image loop ────────────────────────────────────────────────────────
    for sample_idx, val_idx in enumerate(selected, start=1):
        img_tensor, mask_tensor = val_ds[val_idx]

        # Inference
        input_batch = img_tensor.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                output = model(input_batch)

        prob_map    = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_pred = (prob_map > threshold).astype(np.uint8)
        gt_binary   = mask_tensor.cpu().numpy().astype(np.uint8)

        # Metrics
        iou, dice = _iou_dice(binary_pred, gt_binary)

        # Visual prep
        rgb_float  = tensor_to_rgb(img_tensor)           # [H, W, 3] float [0,1]
        rgb_u8     = (rgb_float * 255).astype(np.uint8)  # for Tier 1 modules
        overlay_fp = build_overlay(rgb_float, binary_pred, gt_binary)

        # Tier 1 mask needs 0/255 uint8
        pred_mask_u8 = (binary_pred * 255).astype(np.uint8)

        H, W = binary_pred.shape

        # ── Run Tier 1 ────────────────────────────────────────────────────────
        mean_width_m  = 0.0
        dominant_surf = 'n/a'
        width_result  = None
        type_result   = None
        route_overlay = None

        if _TIER1_OK:
            (width_result, type_result,
             route_overlay, mean_width_m, dominant_surf) = _run_tier1(
                pred_mask_u8, rgb_u8, H, W)

        # ── 8-panel figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 8, figsize=(32, 4))
        fig.patch.set_facecolor('#1a1a2e')

        title = (f"Val #{val_idx}  |  IoU={iou:.4f}  Dice={dice:.4f}"
                 f"  |  Width={mean_width_m:.1f}m  Surface={dominant_surf}")
        fig.suptitle(title, fontsize=11, fontweight='bold',
                     color='white', y=1.01)

        def _style(ax, ttl):
            ax.set_title(ttl, fontsize=9, fontweight='bold',
                         color='#e0e0e0', pad=4)
            ax.axis('off')

        # Panel 1 – Satellite
        axes[0].imshow(rgb_float)
        _style(axes[0], 'Input Satellite')

        # Panel 2 – Ground truth
        axes[1].imshow(gt_binary, cmap='gray', vmin=0, vmax=1)
        _style(axes[1], 'Ground Truth')

        # Panel 3 – Confidence heatmap
        im = axes[2].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        _style(axes[2], 'Confidence Heatmap')

        # Panel 4 – Binary prediction
        axes[3].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
        _style(axes[3], 'Binary Prediction')

        # Panel 5 – TP/FP/FN overlay
        axes[4].imshow(overlay_fp)
        _style(axes[4], 'TP / FP / FN Overlay')
        legend_handles = [
            mpatches.Patch(color=(0.0, 0.78, 0.0), label='TP'),
            mpatches.Patch(color=(0.86, 0.0, 0.0), label='FP'),
            mpatches.Patch(color=(0.0, 0.0, 0.86), label='FN'),
        ]
        axes[4].legend(handles=legend_handles, loc='lower left',
                       fontsize=7, framealpha=0.7)

        # Panel 6 – Width heatmap (Module 1)
        if width_result is not None and not width_result.is_empty:
            im6 = axes[5].imshow(width_result.width_heatmap_rgb)
            plt.colorbar(im6, ax=axes[5], fraction=0.046, pad=0.04,
                         label='Width (m)')
            _style(axes[5], f'Width Heatmap  μ={mean_width_m:.1f}m')
        else:
            _black_panel(H, W, 'No roads\ndetected', axes[5])
            _style(axes[5], 'Width Heatmap')

        # Panel 7 – Surface type overlay (Module 2)
        if type_result is not None and not type_result.get('is_empty', True):
            axes[6].imshow(type_result['overlay_rgb'])
            _style(axes[6], f'Surface  [{dominant_surf}]')
        else:
            _black_panel(H, W, 'Surface N/A', axes[6])
            _style(axes[6], 'Surface Overlay')

        # Panel 8 – Route overlay (Module 3)
        if route_overlay is not None:
            axes[7].imshow(route_overlay)
            _style(axes[7], 'Route  [car · cyan]')
        else:
            _black_panel(H, W, 'No route\nfound', axes[7])
            _style(axes[7], 'Route Overlay')

        plt.tight_layout()

        # ── Save ──────────────────────────────────────────────────────────────
        fname    = f"road_viz_{timestamp}_sample{sample_idx:02d}_val{val_idx}.png"
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"  [{sample_idx}/{n_samples}] IoU={iou:.4f} | Dice={dice:.4f} "
              f"| Width={mean_width_m:.1f}m | Surface={dominant_surf} → {out_path}")

    print(f"\n✅ All {n_samples} visualisations saved to '{save_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 ▸ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model_path = '/kaggle/working/road_model_best.pth'
    visualise_road(
        model_path=model_path,
        n_samples=5,
        threshold=0.5,
        save_dir='/kaggle/working/results',
    )