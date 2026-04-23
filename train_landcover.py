# =============================================================================
# train_landcover.py  –  Stage 3: Land Cover Classification (7-class)
# =============================================================================
# Architecture : DeepLabV3+ ResNet34  (encoder_output_stride=16)
# Loss         : 0.6 * WeightedCrossEntropy  +  0.4 * DiceLoss (multiclass)
# Optimizer    : AdamW  lr=2e-4  wd=1e-4
# Scheduler    : CosineAnnealingWarmRestarts  T0=10  T_mult=2
# Epochs       : 40   |  Early stopping patience: 7  (based on val mIoU)
# Precision    : Mixed (torch.cuda.amp)  mandatory for T4 15 GB VRAM
# =============================================================================

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import get_landcover_splits, DeepGlobeLandCoverDataset
from models import get_landcover_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR   = '/content/drive/MyDrive/datasets/landcover/images'
MASK_DIR    = '/content/drive/MyDrive/datasets/landcover/masks'
SAVE_PATH   = '/content/drive/MyDrive/datasets/landcover_model_latest.pth'
CKPT_DIR    = '/content/drive/MyDrive/datasets/landcover_ckpts'

NUM_CLASSES   = 7
EPOCHS        = 40
BATCH_SIZE    = 8    # T4 with AMP + stride-16 encoder: 8 is safe
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7    # early stopping based on val mIoU
CKPT_EVERY    = 5    # save checkpoint every N epochs
NUM_WORKERS   = 2
PIN_MEMORY    = True

CLASS_NAMES = DeepGlobeLandCoverDataset.CLASS_NAMES

os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥  Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

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
# Model
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
# AdamW: decoupled weight decay — better regularisation for multi-class tasks
#        than vanilla Adam, especially with deeply imbalanced classes.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
# CosineAnnealingWarmRestarts: periodic LR resets help escape local minima
# in multi-class loss landscapes; T_mult=2 doubles the period after each restart.
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
scaler = GradScaler()

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
best_miou      = 0.0
epochs_no_improve = 0

print(f"\n{'='*60}")
print(f"  Training Land Cover (7-class) for {EPOCHS} epochs")
print(f"  Batch={BATCH_SIZE} | LR={LR} | AMP=ON | Patience={PATIENCE}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
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

        with autocast():
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

            with autocast():
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
    for name, iou_val in zip(CLASS_NAMES, per_class_iou.tolist()):
        flag = " ← best" if (not torch.isnan(
            per_class_iou[CLASS_NAMES.index(name)]
        ) and iou_val == per_class_iou.nanmax().item()) else ""
        print(f"    {name:<12s}: {iou_val:.4f}{flag}"
              if not (iou_val != iou_val) else  # nan guard
              f"    {name:<12s}: N/A (absent in val split)")

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
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'miou': miou}, ckpt_path)
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
