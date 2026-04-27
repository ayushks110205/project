# =============================================================================
# train_landcover.py  –  Stage 3: Land Cover Classification (7-class)
# =============================================================================
# Architecture : DeepLabV3+ ResNet34  (encoder_output_stride=16)
# Loss         : 0.6 * WeightedCrossEntropy  +  0.4 * DiceLoss (multiclass)
# Optimizer    : AdamW  lr=2e-4  wd=1e-4
# Scheduler    : CosineAnnealingWarmRestarts  T0=10  T_mult=2
# Epochs       : 40   |  Early stopping patience: 7  (based on val mIoU)
# Precision    : Mixed (torch.cuda.amp)  mandatory
# Environment  : Kaggle P100  (16 GB VRAM, single GPU, 30 GB RAM)
#                • Single GPU — no DataParallel overhead
#                • BATCH_SIZE=16 fits comfortably on 16 GB P100 at 512×512 fp16
#                • num_workers=0, pin_memory=False for RAM stability
#                • Dataset at /kaggle/input/, outputs to /kaggle/working/
# =============================================================================

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import get_landcover_splits, DeepGlobeLandCoverDataset
from models import get_landcover_model

# ─────────────────────────────────────────────────────────────────────────────
# Config  — Kaggle paths
# ─────────────────────────────────────────────────────────────────────────────
# ⚠ Update DATASET_NAME to match your Kaggle dataset slug.
# Input datasets are READ-ONLY at /kaggle/input/<slug>/
# All outputs (models, checkpoints) go to /kaggle/working/ (persistent).
# ✔️ Land Cover dataset — balraj98/deepglobe-land-cover-classification-dataset
DATASET_NAME = 'deepglobe-land-cover-classification-dataset'
DATASET_BASE = f'/kaggle/input/datasets/balraj98/{DATASET_NAME}'

IMAGE_DIR    = f'{DATASET_BASE}/train'
MASK_DIR     = f'{DATASET_BASE}/train'

# ✔️ These are correct for saving your work
SAVE_PATH    = '/kaggle/working/landcover_best.pth'
CKPT_DIR     = '/kaggle/working/landcover_ckpts'

# ── Checkpoint Resume ────────────────────────────────────────────────
# Auto-detects latest checkpoint in RESUME_CKPT_DIR.
# • Default : CKPT_DIR (current session checkpoints, if they exist)
# • Override: point at an uploaded Kaggle dataset containing .pth files
# • Disable : set to None to always start from scratch
RESUME_CKPT_DIR = '/kaggle/input/datasets/ayushks07/best-path'
NUM_CLASSES   = 7
EPOCHS        = 40
# P100 16 GB VRAM (single GPU). DeepLabV3+ ResNet34 at 512×512 fp16 with
# 7-class head uses ~1.2 GB activations per sample. Batch=16 → ~19 GB,
# which sits comfortably within 16 GB with AMP halving tensor sizes.
BATCH_SIZE    = 16
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7
CKPT_EVERY    = 5
# num_workers=0: each worker spawns a full Python process (~1.5 GB RAM).
# With P100’s ~15 GB RAM baseline, keeping workers=0 is the safe choice.
NUM_WORKERS   = 0
PIN_MEMORY    = False   # only useful with workers > 0

CLASS_NAMES = DeepGlobeLandCoverDataset.CLASS_NAMES
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Single-GPU setup (P100) ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cuDNN: deterministic mode prevents algorithm-probe caching from leaking RAM
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True

print(f"🖥  Device: {device}")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
train_ds, val_ds = get_landcover_splits(IMAGE_DIR, MASK_DIR, val_ratio=0.2)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# ─────────────────────────────────────────────────────────────────────────────
# Class weights  (inverse frequency — must be computed BEFORE model init
#                 so Colab prints class distribution at startup)
# ─────────────────────────────────────────────────────────────────────────────
class_weights = train_ds.get_class_weights().to(device)
# class_weights: (7,) float32  — higher for rarer classes

# ─────────────────────────────────────────────────────────────────────────────
# Model  (single GPU — P100)
# ─────────────────────────────────────────────────────────────────────────────
model = get_landcover_model().to(device)
# model input:  (B, 3, 512, 512)  float32
# model output: (B, 7, 512, 512)  float32  raw logits

# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
# CrossEntropy handles per-pixel class confusion with class-imbalance weighting.
# DiceLoss handles boundary quality — Forest vs Rangeland boundaries are
# notoriously blurry in satellite imagery; Dice penalises boundary errors more.
ce_loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

def combined_loss(logits, targets):
    """0.6 * CE  +  0.4 * Dice.

    Args:
        logits  : (B, 7, H, W) float32 raw logits
        targets : (B, H, W)    int64  class IDs 0-6
    Returns:
        scalar loss tensor
    """
    # CrossEntropyLoss expects (B, C, H, W), (B, H, W)
    ce   = ce_loss_fn(logits, targets)
    # DiceLoss (smp) expects same shapes
    dice = dice_loss_fn(logits, targets)
    return 0.6 * ce + 0.4 * dice

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & Scheduler
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
# CosineAnnealingWarmRestarts: periodic LR resets help escape local minima
# in multi-class loss landscapes; T_mult=2 doubles the period after each restart.
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
scaler = torch.amp.GradScaler('cuda')


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_landcover_checkpoint(ckpt_dir):
    """Return path of the highest-epoch 'landcover_epoch*.pth' in ckpt_dir.
    Returns None if directory doesn't exist or contains no matching files."""
    import re
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r'landcover_epoch(\d+)\.pth$')
    candidates = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), os.path.join(ckpt_dir, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(preds, targets, num_classes=NUM_CLASSES):
    """Accumulate confusion matrix from a single batch.

    Args:
        preds   : (B, H, W) int64  predicted class IDs
        targets : (B, H, W) int64  ground-truth class IDs
    Returns:
        conf_mat : (C, C) int64  numpy array
    """
    import numpy as np
    mask = (targets >= 0) & (targets < num_classes)
    idx  = num_classes * targets[mask] + preds[mask]
    import numpy as np
    conf = torch.bincount(idx, minlength=num_classes ** 2)
    return conf.reshape(num_classes, num_classes)


def iou_from_conf(conf):
    """Per-class IoU from confusion matrix.

    IoU_c = conf[c,c] / (row_c + col_c - conf[c,c])
    Returns tensor of shape (C,); nan for absent classes.
    """
    tp  = conf.diagonal().float()
    fp  = conf.sum(dim=0).float() - tp   # predicted as c but not c
    fn  = conf.sum(dim=1).float() - tp   # is c but not predicted as c
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float('nan')))
    return iou


def pixel_accuracy(conf):
    """Overall pixel accuracy from confusion matrix."""
    return conf.diagonal().sum().float() / conf.sum().float()

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
best_miou         = 0.0
epochs_no_improve = 0
start_epoch       = 1

# ── Auto-resume from latest checkpoint ──────────────────────────────────
latest_ckpt = find_latest_landcover_checkpoint(RESUME_CKPT_DIR)
if latest_ckpt:
    print(f"\n📂 Auto-detected checkpoint: {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    best_miou         = ckpt.get('best_miou', 0.0)
    epochs_no_improve = ckpt.get('epochs_no_improve', 0)
    start_epoch       = ckpt['epoch'] + 1
    # Restore class weights if saved (avoids re-scanning all masks)
    if 'class_weights' in ckpt:
        class_weights = ckpt['class_weights'].to(device)
        ce_loss_fn    = nn.CrossEntropyLoss(weight=class_weights)
    print(f"   ✅ Restored  : epoch {ckpt['epoch']} | best_mIoU={best_miou:.4f}")
    print(f"   ⏳ Patience  : {epochs_no_improve}/{PATIENCE}")
    print(f"   ⚡ LR        : {optimizer.param_groups[0]['lr']:.2e}")
    print(f"   ▶️  Resuming from epoch {start_epoch}/{EPOCHS}\n")
    del ckpt
else:
    print("📋 No checkpoint found — starting from scratch.\n")

print(f"\n{'='*60}")
print(f"  Training Land Cover (7-class) for {EPOCHS} epochs")
print(f"  Batch={BATCH_SIZE} | LR={LR} | AMP=ON | Patience={PATIENCE}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
                leave=False)

    for images, masks in pbar:
        # images: (B, 3, 512, 512) float32
        # masks:  (B, 512, 512)    int64
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            logits = model(images)
            # logits: (B, 7, 512, 512)
            loss = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        # Gradient clipping prevents exploding gradients in multi-class tasks
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    scheduler.step()
    train_loss /= len(train_loader)

    # ── Validate ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64,
                           device=device)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]",
                                  leave=False):
            # images: (B, 3, 512, 512)
            # masks:  (B, 512, 512) int64
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).long()

            with torch.amp.autocast('cuda'):
                logits = model(images)
                # logits: (B, 7, 512, 512)
                loss   = combined_loss(logits, masks)

            val_loss += loss.item()

            # Predictions: (B, 7, 512, 512) → argmax → (B, 512, 512) class IDs
            preds = logits.argmax(dim=1)  # (B, 512, 512)

            # Accumulate confusion matrix
            mask_flat  = masks.view(-1)   # (B*H*W,)
            pred_flat  = preds.view(-1)   # (B*H*W,)
            valid      = (mask_flat >= 0) & (mask_flat < NUM_CLASSES)
            idx        = NUM_CLASSES * mask_flat[valid] + pred_flat[valid]
            conf_mat  += torch.bincount(
                idx, minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    val_loss /= len(val_loader)

    # ── Metrics ──────────────────────────────────────────────────────────────
    per_class_iou = iou_from_conf(conf_mat)   # (7,) — nan for absent class
    miou          = per_class_iou.nanmean().item()
    px_acc        = pixel_accuracy(conf_mat).item()
    elapsed       = time.time() - t0

    # GPU memory
    if device.type == 'cuda':
        mem_gb = torch.cuda.memory_allocated() / 1e9
    else:
        mem_gb = 0.0

    print(f"\nEpoch {epoch:02d}/{EPOCHS}  "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"mIoU={miou:.4f}  px_acc={px_acc:.4f}  "
          f"time={elapsed:.1f}s  GPU={mem_gb:.2f}GB")

    print("  Per-class IoU:")
    valid_iou = per_class_iou[~per_class_iou.isnan()]
    best_val  = valid_iou.max().item() if valid_iou.numel() > 0 else float('nan')
    for name, iou_val in zip(CLASS_NAMES, per_class_iou.tolist()):
        if iou_val != iou_val:   # NaN guard
            print(f"    {name:<12s}: N/A (absent in val split)")
        else:
            flag = " ← best" if iou_val == best_val else ""
            print(f"    {name:<12s}: {iou_val:.4f}{flag}")


    # ── Checkpointing ────────────────────────────────────────────────────────
    if miou > best_miou:
        best_miou = miou
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
            'class_weights': class_weights.cpu(),
        }, SAVE_PATH)
        print(f"  ✅ Best model saved (mIoU={best_miou:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  ⏳ No improvement ({epochs_no_improve}/{PATIENCE})")

    if epoch % CKPT_EVERY == 0:
        ckpt_path = os.path.join(CKPT_DIR, f'landcover_epoch{epoch:03d}.pth')
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou':            best_miou,
            'epochs_no_improve':    epochs_no_improve,
            'class_weights':        class_weights.cpu(),
        }, ckpt_path)
        print(f"  💾 Checkpoint saved: {ckpt_path}")

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch} "
              f"(no val mIoU improvement for {PATIENCE} epochs)")
        break

    torch.cuda.empty_cache()
    print()

print(f"\n🏁 Training complete. Best val mIoU = {best_miou:.4f}")
print(f"   Model saved to: {SAVE_PATH}")
