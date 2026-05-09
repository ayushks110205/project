# =============================================================================
# evaluate_road.py  –  Honest Evaluation on the Val Split
# =============================================================================
# KEY CHANGES vs v1:
#   • BUG FIX: evaluates on the VAL split only (was full train set — inflated
#     metrics because the model had seen all those images during training).
#   • Extended metrics: mIoU, mean Dice, Precision, Recall, F1 (separately)
#   • Per-image breakdown: worst 5 and best 5 predictions by IoU
#   • Clean, formatted report printed to console
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

# Pipeline imports are LAZY (inside run_evaluation()) — prevents the
# albumentations → scipy → numpy crash on `import evaluate_road`.
# Run as subprocess:  !python evaluate_road.py [--tta]


# ─────────────────────────────────────────────────────────────────────────────
# Section 0 ▸ TTA Helper  [Plan 5]
# ─────────────────────────────────────────────────────────────────────────────

def tta_predict(model: torch.nn.Module,
               image_tensor: torch.Tensor,
               device: torch.device) -> np.ndarray:
    """
    Test-Time Augmentation: average sigmoid probabilities over 4 flip orientations.

    Augmentations:
        0. Original
        1. Horizontal flip  (left ↔ right)
        2. Vertical flip    (up ↔ down)
        3. Both flips       (180° rotation)

    Args:
        model        : road segmentation model (eval mode, on device).
        image_tensor : (3, H, W) float32 tensor — NOT batched.
        device       : torch device.

    Returns:
        prob_map : (H, W) float32 numpy array, averaged probability in [0, 1].
    """
    flip_configs = [
        [],          # original
        [2],         # horizontal flip  (dim 2 = width on a 3D tensor)
        [3],         # vertical flip    (dim 3 = height ... but we unsqueeze below)
        [2, 3],      # both
    ]
    # Note: after unsqueeze(0) the tensor is (1, 3, H, W).
    # Width = dim 3, Height = dim 2 in a (B, C, H, W) batch.
    flip_configs_batch = [
        [],
        [3],   # horizontal (W)
        [2],   # vertical   (H)
        [2, 3],
    ]

    preds = []
    batch = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    for flip_dims in flip_configs_batch:
        x = torch.flip(batch, dims=flip_dims) if flip_dims else batch
        with torch.no_grad():
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logit = model(x)
        prob = torch.sigmoid(logit).squeeze().cpu()   # (H, W)
        if flip_dims:
            prob = torch.flip(prob.unsqueeze(0), dims=[d - 1 for d in flip_dims]).squeeze(0)
        preds.append(prob)

    return torch.stack(preds).mean(0).numpy()   # (H, W) float32


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_prob: np.ndarray, target: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Compute binary segmentation metrics for a single image.

    Args:
        pred_prob : sigmoid probability map  [H, W], values in [0, 1]
        target    : binary ground-truth mask [H, W], values in {0, 1}
        threshold : binarisation cutoff (default 0.5)

    Returns:
        dict with keys: iou, dice, precision, recall, f1
    """
    pred   = (pred_prob > threshold).astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    tn = np.logical_and(pred == 0, target == 0).sum()  # noqa: F841 (unused but kept for clarity)

    eps = 1e-6
    iou       = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)  # = F1 for binary

    return {
        'iou':       float(iou),
        'dice':      float(dice),
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_path: str,
                   image_dir: str = '/kaggle/input/datasets/balraj98/deepglobe-road-extraction-dataset/train',
                   mask_dir:  str = '/kaggle/input/datasets/balraj98/deepglobe-road-extraction-dataset/train',
                   val_ratio: float = 0.20,
                   threshold: float = 0.50,
                   top_k:     int   = 5,
                   use_tta:   bool  = False):   # Plan 5: TTA flag
    """
    Evaluate the road extraction model on the held-out val split.

    Args:
        model_path : path to saved model weights (.pth)
        image_dir  : dataset image directory
        mask_dir   : dataset mask directory
        val_ratio  : must match the value used during training (default 0.20)
        threshold  : binarisation threshold for predictions
        top_k      : number of best / worst images to report in breakdown
        use_tta    : if True, average over 4 flip orientations at inference (+0.5-1 IoU)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Evaluation | Device: {device} | TTA={'ON' if use_tta else 'OFF'}")

    # Lazy imports — deferred to avoid albumentations → scipy → numpy crash
    from dataset import get_road_splits   # noqa: PLC0415
    from models  import get_road_model    # noqa: PLC0415

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_road_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"✅ Model loaded from: {model_path}\n")

    # ── Load val split (same deterministic split as training) ─────────────────
    _, val_ds = get_road_splits(image_dir, mask_dir, val_ratio=val_ratio)
    val_loader = DataLoader(
        val_ds, batch_size=1 if use_tta else 16, shuffle=False,
        num_workers=0, pin_memory=False
    )
    print(f"🗂️  Evaluating on {len(val_ds)} validation images "
          f"({val_ratio*100:.0f}% of total dataset)\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    per_image_metrics = []   # list of dicts, one per image

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating", unit='batch'):
            images = images.to(device, non_blocking=True)

            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)

            preds_np   = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            targets_np = masks.cpu().numpy()

            for pred, target in zip(preds_np, targets_np):
                metrics = compute_metrics(pred, target, threshold=threshold)
                per_image_metrics.append(metrics)

    n = len(per_image_metrics)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mean_iou       = np.mean([m['iou']       for m in per_image_metrics])
    mean_dice      = np.mean([m['dice']      for m in per_image_metrics])
    mean_precision = np.mean([m['precision'] for m in per_image_metrics])
    mean_recall    = np.mean([m['recall']    for m in per_image_metrics])
    mean_f1        = np.mean([m['f1']        for m in per_image_metrics])

    # ── Sort by IoU for best/worst breakdown ──────────────────────────────────
    sorted_by_iou = sorted(enumerate(per_image_metrics),
                            key=lambda x: x[1]['iou'])
    worst_k = sorted_by_iou[:top_k]
    best_k  = sorted_by_iou[-top_k:][::-1]  # highest first

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "=" * 55
    print(f"\n{sep}")
    print("            ROAD EXTRACTION EVALUATION REPORT")
    print(f"{sep}")
    print(f"  Split         : Validation only ({n} images)")
    print(f"  Threshold     : {threshold}")
    print(f"{'-'*55}")
    print(f"  Mean IoU      : {mean_iou:.4f}")
    print(f"  Mean Dice     : {mean_dice:.4f}")
    print(f"  Mean Precision: {mean_precision:.4f}")
    print(f"  Mean Recall   : {mean_recall:.4f}")
    print(f"  Mean F1       : {mean_f1:.4f}")
    print(f"{sep}")

    # -- Worst predictions -----------------------------------------------------
    print(f"\n  ▼ WORST {top_k} PREDICTIONS BY IoU")
    print(f"  {'Val Index':>10} | {'IoU':>6} | {'Dice':>6} | {'Precision':>9} | {'Recall':>6}")
    print(f"  {'-'*55}")
    for idx, m in worst_k:
        print(f"  {idx:>10} | {m['iou']:.4f} | {m['dice']:.4f} | "
              f"{m['precision']:.4f}    | {m['recall']:.4f}")

    # -- Best predictions ------------------------------------------------------
    print(f"\n  ▲ BEST {top_k} PREDICTIONS BY IoU")
    print(f"  {'Val Index':>10} | {'IoU':>6} | {'Dice':>6} | {'Precision':>9} | {'Recall':>6}")
    print(f"  {'-'*55}")
    for idx, m in best_k:
        print(f"  {idx:>10} | {m['iou']:.4f} | {m['dice']:.4f} | "
              f"{m['precision']:.4f}    | {m['recall']:.4f}")

    print(f"\n{sep}\n")

    return {
        'mean_iou':       mean_iou,
        'mean_dice':      mean_dice,
        'mean_precision': mean_precision,
        'mean_recall':    mean_recall,
        'mean_f1':        mean_f1,
        'per_image':      per_image_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate road segmentation model')
    parser.add_argument('--model', default='/kaggle/working/road_model_best.pth',
                        help='Path to model weights')
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Augmentation (4-flip ensemble)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarisation threshold (default 0.5)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found at '{args.model}'. Train the model first.")
    else:
        run_evaluation(args.model, threshold=args.threshold, use_tta=args.tta)