# =============================================================================
# evaluate_inpainting.py  –  Stage 2 Inpainting Evaluation
# =============================================================================
# Metrics: Hole IoU, Full IoU, Connectivity Score, Hole MAE
# Visual : 5-panel figures (Complete | Corrupted | Prediction | GT | Error Map)
# =============================================================================

import os
import datetime
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from inpainting_dataset import get_inpainting_splits
from inpainting_model   import get_inpainting_model


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(p: np.ndarray, t: np.ndarray) -> float:
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def hole_iou_score(pred_bin: np.ndarray, target_bin: np.ndarray,
                   hole_map: np.ndarray) -> float:
    """IoU restricted to hole pixels (hole_map=1 means hole)."""
    p = pred_bin   * hole_map
    t = target_bin * hole_map
    return iou_score(p, t)


def hole_mae_score(pred_prob: np.ndarray, target: np.ndarray,
                   hole_map: np.ndarray) -> float:
    mask = hole_map.astype(bool)
    if mask.sum() < 1:
        return 0.0
    return float(np.abs(pred_prob[mask] - target[mask]).mean())


def connectivity_score(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Ratio of connected road components pred/target via cv2.connectedComponents.
    Score ≈ 1.0 → same topology as ground truth.
    """
    _, pl = cv2.connectedComponents(pred_bin.astype(np.uint8))
    _, tl = cv2.connectedComponents(target_bin.astype(np.uint8))
    return float(max(pl.max(), 0) / max(tl.max(), 1))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _error_map(pred_bin: np.ndarray, target_bin: np.ndarray,
               hole_map: np.ndarray) -> np.ndarray:
    H, W = pred_bin.shape
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 200
    h = hole_map.astype(bool)
    p = pred_bin.astype(bool)
    t = target_bin.astype(bool)
    rgb[h & p & t]   = [0,   200, 0]    # TP = green
    rgb[h & p & ~t]  = [220, 0,   0]    # FP = red
    rgb[h & ~p & t]  = [0,   0, 220]    # FN = blue
    rgb[~h] = (rgb[~h] * 0.4).astype(np.uint8)
    return rgb


def save_visual(complete: np.ndarray, corrupted: np.ndarray,
                pred_bin: np.ndarray, hole_map: np.ndarray,
                title: str, path: str):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    for ax, img, lbl in zip(axes[:4],
                             [complete, corrupted, pred_bin, complete],
                             ["Original", "Corrupted", "Prediction", "Ground Truth"]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.axis('off')
    axes[4].imshow(_error_map(pred_bin, (complete > 0.5).astype(np.uint8), hole_map))
    axes[4].set_title("Error Map", fontsize=11, fontweight='bold')
    axes[4].axis('off')
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               [((0,.78,0),'TP'), ((.86,0,0),'FP'), ((0,0,.86),'FN')]]
    axes[4].legend(handles=patches, loc='lower left', fontsize=8, framealpha=.75)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_path: str,
                   mask_dir: str = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset/train',
                   val_ratio: float = 0.20, threshold: float = 0.50,
                   n_visuals: int = 5,
                   save_dir: str = '/kaggle/working/results/inpainting_eval'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Inpainting Evaluation | Device: {device}")

    model = get_inpainting_model(base_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded: {model_path}\n")

    _, val_ds = get_inpainting_splits(mask_dir, val_ratio=val_ratio)
    loader    = DataLoader(val_ds, batch_size=8, shuffle=False,
                           num_workers=0, pin_memory=True)
    print(f"🗂️  Val images: {len(val_ds)}\n")
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    all_m = []

    with torch.no_grad():
        for corrupted, hole_mask, complete in tqdm(loader, desc="Eval", unit='batch'):
            corrupted = corrupted.to(device, non_blocking=True)
            hole_mask = hole_mask.to(device, non_blocking=True)
            with autocast('cuda'):
                pred = model(corrupted, hole_mask)   # (B,1,H,W)

            pp  = pred.squeeze(1).cpu().numpy()            # (B,H,W) prob
            cn  = corrupted.squeeze(1).cpu().numpy()       # (B,H,W)
            hn  = (1 - hole_mask.squeeze(1).cpu().numpy()) # 1=hole
            tn  = complete.squeeze(1).cpu().numpy()        # (B,H,W)

            for p_prob, t, h in zip(pp, tn, hn):
                pb = (p_prob > threshold).astype(np.uint8)
                tb = (t > 0.5).astype(np.uint8)
                hb = (h > 0.5).astype(np.uint8)
                all_m.append({
                    'hole_iou':    hole_iou_score(pb, tb, hb),
                    'full_iou':    iou_score(pb, tb),
                    'hole_mae':    hole_mae_score(p_prob, t, hb),
                    'connectivity': connectivity_score(pb, tb),
                })

    # Aggregate
    mhi  = np.mean([m['hole_iou']     for m in all_m])
    mfi  = np.mean([m['full_iou']     for m in all_m])
    mmae = np.mean([m['hole_mae']     for m in all_m])
    mcon = np.mean([m['connectivity'] for m in all_m])

    sep = '=' * 55
    print(f"\n{sep}")
    print("      STAGE 2 INPAINTING EVALUATION REPORT")
    print(f"{sep}")
    print(f"  Val images          : {len(all_m)}")
    print(f"  Threshold           : {threshold}")
    print(f"{'-'*55}")
    print(f"  ★ Hole IoU (primary): {mhi:.4f}")
    print(f"  Full IoU            : {mfi:.4f}")
    print(f"  Hole MAE            : {mmae:.4f}")
    print(f"  Connectivity Score  : {mcon:.4f}  (1.0=perfect)")
    print(f"{sep}\n")

    # Visual reports
    n_vis = min(n_visuals, len(val_ds))
    vis_idx = random.sample(range(len(val_ds)), n_vis)
    print(f"🎨 Saving {n_vis} visual reports → {save_dir}/")

    for si, vi in enumerate(vis_idx, 1):
        c_t, h_t, t_t = val_ds[vi]
        c_t = c_t.unsqueeze(0).to(device)
        h_t = h_t.unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast('cuda'):
                p_t = model(c_t, h_t)

        pp   = p_t.squeeze().cpu().numpy()
        pb   = (pp > threshold).astype(np.uint8)
        tn   = t_t.squeeze().numpy()
        hn   = (1 - h_t.squeeze().cpu().numpy()).astype(np.uint8)
        hi   = hole_iou_score(pb, (tn > 0.5).astype(np.uint8), hn)
        fi   = iou_score(pb, (tn > 0.5).astype(np.uint8))
        cs   = connectivity_score(pb, (tn > 0.5).astype(np.uint8))
        title = f"Val #{vi}  Hole IoU={hi:.4f}  Full IoU={fi:.4f}  Conn={cs:.3f}"
        save_visual(tn, c_t.squeeze().cpu().numpy(), pb, hn, title,
                    os.path.join(save_dir, f"inpaint_{ts}_{si:02d}_val{vi}.png"))
        print(f"  [{si}/{n_vis}] {title}")

    print(f"\n✅ Done. Figures saved to: {save_dir}/")
    return all_m


if __name__ == '__main__':
    # Check multiple candidate paths — working dir first, then uploaded dataset
    model_candidates = [
        '/kaggle/working/inpainting_best.pth',
        '/kaggle/input/datasets/ayushks07/best-path/inpainting_best.pth',
    ]
    mp = next((p for p in model_candidates if os.path.exists(p)), None)
    if mp is None:
        print("❌ Model not found in any of:")
        for p in model_candidates:
            print(f"   {p}")
        print("Train Stage 2 first, or upload inpainting_best.pth to the 'best path' dataset.")
    else:
        print(f"📦 Loading model from: {mp}")
        run_evaluation(mp)
