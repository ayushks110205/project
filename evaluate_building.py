# =============================================================================
# evaluate_building.py  –  Stage 4 Building Detection Evaluation  (P100)
# =============================================================================
# Features:
#   • TTA (Test Time Augmentation): 8-variant ensemble for +1-2% IoU
#   • Full metric suite: IoU, Dice, Precision, Recall, F1
#   • Side-by-side standard vs TTA comparison
#   • Single-GPU (P100) — no DDP
# =============================================================================

import os
import numpy as np
import torch
import torch.amp
from torch.utils.data import DataLoader

from dataset import MassachusettsBuildingDataset, building_val_transform
from models import get_building_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATASET_BASE = '/kaggle/input/datasets/balraj98/massachusetts-buildings-dataset'
IMAGE_DIR    = f'{DATASET_BASE}/tiff/val'
MASK_DIR     = f'{DATASET_BASE}/tiff/val_labels'
MODEL_PATH   = '/kaggle/working/building_model_best.pth'
RESULTS_DIR  = '/kaggle/working/results/building_eval'
BATCH_SIZE   = 4
NUM_WORKERS  = 0

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥  Device: {device}")


# =============================================================================
# Section 1 ▸ TTA
# =============================================================================

def tta_predict(model, images):
    """8-variant TTA ensemble — averages sigmoid probs over 8 augmentations."""
    preds = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds.append(torch.sigmoid(model(images)))

            aug = torch.flip(images, [3])
            preds.append(torch.flip(torch.sigmoid(model(aug)), [3]))

            aug = torch.flip(images, [2])
            preds.append(torch.flip(torch.sigmoid(model(aug)), [2]))

            aug = torch.rot90(images, k=1, dims=[2, 3])
            preds.append(torch.rot90(torch.sigmoid(model(aug)), k=-1, dims=[2, 3]))

            aug = torch.rot90(images, k=2, dims=[2, 3])
            preds.append(torch.rot90(torch.sigmoid(model(aug)), k=-2, dims=[2, 3]))

            aug = torch.rot90(images, k=3, dims=[2, 3])
            preds.append(torch.rot90(torch.sigmoid(model(aug)), k=-3, dims=[2, 3]))

            aug = torch.rot90(torch.flip(images, [3]), k=1, dims=[2, 3])
            pred = torch.rot90(torch.sigmoid(model(aug)), k=-1, dims=[2, 3])
            preds.append(torch.flip(pred, [3]))

            aug = torch.rot90(torch.flip(images, [2]), k=1, dims=[2, 3])
            pred = torch.rot90(torch.sigmoid(model(aug)), k=-1, dims=[2, 3])
            preds.append(torch.flip(pred, [2]))

    return torch.stack(preds, dim=0).mean(dim=0)


# =============================================================================
# Section 2 ▸ Metrics
# =============================================================================

def compute_metrics(pred_prob, masks, threshold=0.5):
    pred_bin = (pred_prob > threshold).float()
    smooth   = 1e-6
    tp = (pred_bin * masks).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * masks).sum(dim=(1, 2, 3))

    iou       = ((tp + smooth) / (tp + fp + fn + smooth)).mean().item()
    dice      = ((2*tp + smooth) / (2*tp + fp + fn + smooth)).mean().item()
    precision = ((tp + smooth) / (tp + fp + smooth)).mean().item()
    recall    = ((tp + smooth) / (tp + fn + smooth)).mean().item()
    f1        = (2 * precision * recall + smooth) / (precision + recall + smooth)
    return {'iou': iou, 'dice': dice,
            'precision': precision, 'recall': recall, 'f1': f1}


def avg_metrics(accum, batch_m, n):
    for k, v in batch_m.items():
        accum[k] = accum.get(k, 0.0) + v * n
    accum['_n'] = accum.get('_n', 0) + n
    return accum


def finalise(accum):
    n = max(accum.pop('_n', 1), 1)
    return {k: v / n for k, v in accum.items()}


# =============================================================================
# Section 3 ▸ Main Evaluation
# =============================================================================

def run_evaluation(model_path=MODEL_PATH, use_tta=True):
    # ── Data ──────────────────────────────────────────────────────────────────
    val_ds = MassachusettsBuildingDataset(
        IMAGE_DIR, MASK_DIR, transform=building_val_transform
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)
    print(f"📂 Val samples: {len(val_ds)}")
    print(f"   TTA: {'ON (×8 variants)' if use_tta else 'OFF'}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_building_model().to(device)
    state = torch.load(model_path, map_location=device)
    # Support both plain state_dict and full checkpoint dicts
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
        best_iou = state.get('best_iou', '?')
    else:
        model.load_state_dict(state)
        best_iou = '?'
    model.eval()
    print(f"✅ Loaded: {model_path}  (train best IoU={best_iou})\n")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    std_accum = {}
    tta_accum = {} if use_tta else None

    with torch.no_grad():
        for i, (images, masks, _, _) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()
            b      = images.size(0)

            with torch.amp.autocast('cuda'):
                std_prob = torch.sigmoid(model(images))

            std_accum = avg_metrics(std_accum, compute_metrics(std_prob, masks), b)

            if use_tta:
                tta_prob  = tta_predict(model, images)
                tta_accum = avg_metrics(tta_accum, compute_metrics(tta_prob, masks), b)

            print(f"  Batch {i+1}/{len(val_loader)} done", end='\r')

    std_final = finalise(std_accum)
    tta_final = finalise(tta_accum) if use_tta else None

    # ── Report ────────────────────────────────────────────────────────────────
    SEP = '=' * (55 if not use_tta else 70)
    print('\n' + SEP)
    print('    Building Detection — Evaluation Report (P100)')
    print(SEP)

    header = f"{'Metric':<12} {'Standard':>12}"
    if use_tta:
        header += f" {'TTA (×8)':>12} {'Δ TTA':>10}"
    print(header)
    print('-' * (55 if not use_tta else 70))

    for m in ['iou', 'dice', 'precision', 'recall', 'f1']:
        s = std_final[m]
        row = f"{m.capitalize():<12} {s:>12.4f}"
        if use_tta:
            t     = tta_final[m]
            delta = t - s
            sign  = '+' if delta >= 0 else ''
            row  += f" {t:>12.4f} {sign}{delta*100:>8.2f}%"
        print(row)

    print(SEP)
    if use_tta:
        print(f"\n📈 TTA IoU gain: {(tta_final['iou']-std_final['iou'])*100:+.2f}%")
    print(f"   Model: {model_path}")

    return std_final, tta_final


if __name__ == '__main__':
    run_evaluation(model_path=MODEL_PATH, use_tta=True)