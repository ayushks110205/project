# =============================================================================
# evaluate_landcover.py  –  Stage 3: Full Evaluation on Val Split
# =============================================================================
# Computes per-class IoU, F1, Precision, Recall, mIoU, pixel accuracy,
# and frequency-weighted IoU from the full 7x7 confusion matrix.
# Prints a formatted table and saves a confusion matrix heatmap PNG.
# =============================================================================

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch.amp
from tqdm import tqdm

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

NUM_CLASSES = 7
BATCH_SIZE  = 16
NUM_WORKERS = 0   # P100: keep at 0 to avoid multiprocessing OOM
CLASS_NAMES = DeepGlobeLandCoverDataset.CLASS_NAMES

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥  Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Data  (val split only — honest evaluation, no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
_, val_ds = get_landcover_splits(IMAGE_DIR, MASK_DIR, val_ratio=0.2)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
print(f"📂 Val samples: {len(val_ds)}")

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
model = get_landcover_model().to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# Handle both full checkpoint format ('model_state_dict') and plain state_dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
elif isinstance(ckpt, dict) and 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"✅ Loaded model from: {MODEL_PATH}")
if isinstance(ckpt, dict) and 'best_miou' in ckpt:
    print(f"   Checkpoint best val mIoU: {ckpt['best_miou']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Build 7×7 confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)

with torch.no_grad():
    for images, masks in tqdm(val_loader, desc='Evaluating'):
        # images: (B, 3, 512, 512)  masks: (B, 512, 512) int64
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()

        with torch.amp.autocast('cuda'):
            logits = model(images)
            # logits: (B, 7, 512, 512)

        preds = logits.argmax(dim=1)  # (B, 512, 512) class IDs

        # Flatten and accumulate
        mask_flat = masks.view(-1)    # (B*H*W,)
        pred_flat = preds.view(-1)    # (B*H*W,)
        valid     = (mask_flat >= 0) & (mask_flat < NUM_CLASSES)
        idx       = NUM_CLASSES * mask_flat[valid] + pred_flat[valid]
        conf_mat += torch.bincount(
            idx, minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

conf = conf_mat.cpu().float()  # (7, 7)

# ─────────────────────────────────────────────────────────────────────────────
# Derive metrics from confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
tp  = conf.diagonal()                      # (7,)
fp  = conf.sum(dim=0) - tp                 # false positives per class
fn  = conf.sum(dim=1) - tp                 # false negatives per class
total_pixels = conf.sum()

# Per-class IoU
denom_iou = tp + fp + fn
iou = torch.where(denom_iou > 0, tp / denom_iou,
                  torch.zeros_like(tp))    # 0 for absent classes

# Per-class Precision
denom_prec = tp + fp
precision = torch.where(denom_prec > 0, tp / denom_prec,
                        torch.zeros_like(tp))

# Per-class Recall
denom_rec = tp + fn
recall = torch.where(denom_rec > 0, tp / denom_rec,
                     torch.zeros_like(tp))

# Per-class F1 / Dice
denom_f1 = precision + recall
f1 = torch.where(denom_f1 > 0,
                 2 * precision * recall / denom_f1,
                 torch.zeros_like(tp))

# Class frequencies (fraction of total pixels)
class_freq = conf.sum(dim=1) / total_pixels  # (7,)

# Aggregate metrics
# mIoU: mean over classes that actually appear in val set
present = conf.sum(dim=1) > 0               # (7,) bool
miou    = iou[present].mean().item()

# Overall pixel accuracy
px_acc = tp.sum().item() / total_pixels.item()

# Frequency-weighted IoU
fw_iou = (class_freq * iou).sum().item()

# ─────────────────────────────────────────────────────────────────────────────
# Print formatted table
# ─────────────────────────────────────────────────────────────────────────────
SEP  = '─'
HSEP = '┌' + SEP*15 + '┬' + SEP*7 + '┬' + SEP*7 + '┬' + SEP*7 + '┬' + SEP*7 + '┐'
MSEP = '├' + SEP*15 + '┼' + SEP*7 + '┼' + SEP*7 + '┼' + SEP*7 + '┼' + SEP*7 + '┤'
FSEP = '└' + SEP*15 + '┴' + SEP*7 + '┴' + SEP*7 + '┴' + SEP*7 + '┴' + SEP*7 + '┘'

def fmt(v): return f'{v:.4f}'

print('\n' + '='*60)
print('  Land Cover Evaluation Results')
print('='*60)
print(HSEP)
print(f'│ {"Class":<13s} │ {"IoU":^5s} │ {"F1":^5s} │ {"Prec":^5s} │ {"Recall":^5s}│')
print(MSEP)

iou_list  = iou.tolist()
f1_list   = f1.tolist()
prec_list = precision.tolist()
rec_list  = recall.tolist()

for i, name in enumerate(CLASS_NAMES):
    print(f'│ {name:<13s} │ {fmt(iou_list[i])} │ {fmt(f1_list[i])} │ '
          f'{fmt(prec_list[i])} │ {fmt(rec_list[i])} │')

print(MSEP)
print(f'│ {"MEAN (mIoU)":<13s} │ {fmt(miou)} │ {fmt(float(f1[present].mean()))} │ '
      f'{fmt(float(precision[present].mean()))} │ '
      f'{fmt(float(recall[present].mean()))} │')
print(FSEP)

print(f'\n  Overall pixel accuracy : {px_acc:.4f}')
print(f'  Frequency-weighted IoU : {fw_iou:.4f}')
print(f'  mIoU                   : {miou:.4f}')

# Hardest / easiest class (among present classes)
hardest_idx = torch.where(present, iou, torch.full_like(iou, float('inf'))).argmin().item()
easiest_idx = torch.where(present, iou, torch.full_like(iou, float('-inf'))).argmax().item()
print(f'\n  Hardest class: {CLASS_NAMES[hardest_idx]:<12s} IoU={iou_list[hardest_idx]:.4f}')
print(f'  Easiest class: {CLASS_NAMES[easiest_idx]:<12s} IoU={iou_list[easiest_idx]:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────
# Normalise by row (true class) for readability
conf_norm = conf.numpy()
row_sums  = conf_norm.sum(axis=1, keepdims=True)
conf_norm = np.divide(conf_norm, row_sums,
                      out=np.zeros_like(conf_norm),
                      where=row_sums > 0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    conf_norm,
    annot=True, fmt='.2f', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor='#cccccc',
    ax=ax
)
ax.set_xlabel('Predicted', fontsize=12, labelpad=8)
ax.set_ylabel('Ground Truth', fontsize=12, labelpad=8)
ax.set_title(f'Land Cover Confusion Matrix  (val mIoU = {miou:.4f})',
             fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()

heatmap_path = os.path.join(RESULTS_DIR, 'landcover_confusion_matrix.png')
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n💾 Confusion matrix saved to: {heatmap_path}')
