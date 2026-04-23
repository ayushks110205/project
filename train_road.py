# =============================================================================
# train_road.py  –  Road Extraction Training Script (Kaggle T4 ×2)
# =============================================================================
# KEY CHANGES vs v1:
#   • BUG FIX: imports DeepGlobeRoadDataset (was 'RoadDataset' — class never existed)
#   • 80/20 train/val split via get_road_splits() from dataset.py
#   • Combined loss: 0.5 * FocalLoss + 0.5 * DiceLoss
#       – FocalLoss handles severe class imbalance (few road pixels)
#       – DiceLoss penalises thin-road misses that FocalLoss alone overlooks
#   • CosineAnnealingLR scheduler (T_max = num_epochs)
#   • Mixed-precision training via torch.cuda.amp (GradScaler + autocast)
#       – Halves VRAM usage, ~40% speedup on T4
#   • Gradient clipping (max_norm=1.0) prevents exploding gradients
#   • Saves best model based on VAL loss (not train loss)
#   • Early stopping (patience=5)
#   • Checkpoint every 5 epochs saved to /kaggle/working/road_ckpts/
#   • torch.cuda.empty_cache() between epochs to prevent OOM
#   • pin_memory=True, num_workers=2 for Kaggle DataLoader (OOM fix)
#   • tqdm progress bars with live loss display
#   • Epoch-end IoU + Dice printout from val loop
#   • nn.DataParallel for T4 ×2 (30 GB total VRAM)
#
# OOM FIXES applied (v2):
#   • BATCH_SIZE reduced 16 → 8  (8 samples total, 4 per GPU — safe headroom)
#   • num_workers reduced 4 → 2  (fewer shared-mem copies of 512×512 batches)
#   • Val IoU/Dice now computed INCREMENTALLY per-batch (no full-set np.concat)
#   • Intermediate tensors (outputs, preds_np) explicitly deleted after use
#   • torch.cuda.reset_peak_memory_stats() called each epoch for clean logging
# =============================================================================

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("⚠️  psutil not found — install with: pip install psutil")
    print("   RAM diagnostics will be skipped.")


def _ram_gb() -> float:
    """Return current process RSS in GB (requires psutil)."""
    if not _HAS_PSUTIL:
        return -1.0
    import os
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# ── Local imports ─────────────────────────────────────────────────────────────
from dataset import get_road_splits
from models import get_road_model


# =============================================================================
# Section 1 ▸ Configuration
# =============================================================================

BASE_PATH  = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset'
IMAGE_DIR  = f'{BASE_PATH}/train'
MASK_DIR   = f'{BASE_PATH}/train'

LOCAL_BEST = '/kaggle/working/road_model_best.pth'
CKPT_DIR   = '/kaggle/working/road_ckpts'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── OOM FIX #1: Reduced batch size ───────────────────────────────────────────
# DeepLabV3+ ResNet34 at 512×512 with ASPP holds ~1.5 GB of activations per
# sample in fp16. At batch=16 (8 per GPU) that is 12 GB of activations alone,
# leaving very little headroom for gradients and the optimizer states.
# batch=8 (4 per GPU) gives comfortable headroom on both T4 cards.
BATCH_SIZE = 8

NUM_EPOCHS          = 35
LR                  = 1e-4
VAL_RATIO           = 0.2
EARLY_STOP_PATIENCE = 5
CHECKPOINT_EVERY    = 5


# =============================================================================
# Section 2 ▸ GPU / Multi-GPU Setup
# =============================================================================

device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus        = torch.cuda.device_count()
USE_MULTI_GPU = n_gpus > 1

if torch.cuda.is_available():
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"🖥️  GPU {i}: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
    print(f"   DataParallel: {'YES (×' + str(n_gpus) + ')' if USE_MULTI_GPU else 'NO (single GPU)'}")
else:
    print("⚠️  No GPU detected — running on CPU (will be slow!)")


# =============================================================================
# Section 3 ▸ Metric Helpers (incremental — no full-set accumulation)
# =============================================================================

def update_iou_dice(pred_prob: np.ndarray, target: np.ndarray,
                    iou_sum: float, dice_sum: float,
                    n: int, threshold: float = 0.5):
    """
    Incrementally accumulate IoU and Dice per-batch.
    Returns updated (iou_sum, dice_sum, n) — call np.divide at epoch end.

    This avoids the OOM caused by concatenating the entire val set into RAM.
    """
    binary  = (pred_prob > threshold).astype(np.uint8)
    targets = target.astype(np.uint8)
    for p, t in zip(binary, targets):
        inter      = np.logical_and(p, t).sum()
        union      = np.logical_or(p, t).sum()
        iou_sum   += (inter + 1e-6) / (union + 1e-6)
        dice_sum  += (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
        n         += 1
    return iou_sum, dice_sum, n


# =============================================================================
# Section 4 ▸ Main Training Function
# =============================================================================

def train_road(epochs: int = NUM_EPOCHS):
    # ── 4a. Data ──────────────────────────────────────────────────────────────
    print(f"\n🩺 RAM before split  : {_ram_gb():.2f} GB")
    train_ds, val_ds = get_road_splits(IMAGE_DIR, MASK_DIR, val_ratio=VAL_RATIO)
    gc.collect()   # flush any lingering temporaries from the split
    print(f"🩺 RAM after split   : {_ram_gb():.2f} GB")

    # num_workers=0: data loading runs in the main process.
    # No subprocess workers → zero shared-memory buffers → eliminates the
    # RAM spike that was crashing at epoch 2.  pin_memory is also disabled
    # because it only helps when workers copy into pinned memory from
    # subprocesses (irrelevant at num_workers=0).
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    print(f"🩺 RAM after loaders : {_ram_gb():.2f} GB")

    print(f"\n🚀 Road Training | Device: {device}")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"   Epochs: {epochs} | Batch size: {BATCH_SIZE} | LR: {LR}\n")

    # ── 4b. Model  +  DataParallel ────────────────────────────────────────────
    model = get_road_model().to(device)
    print(f"🩺 RAM after model   : {_ram_gb():.2f} GB")
    if USE_MULTI_GPU:
        model = nn.DataParallel(model)
        print(f"   Model wrapped in DataParallel across {n_gpus} GPUs")

    # ── 4c. Combined Loss ─────────────────────────────────────────────────────
    focal_loss = smp.losses.FocalLoss(mode='binary')
    dice_loss  = smp.losses.DiceLoss(mode='binary')

    def criterion(logits, targets):
        return 0.5 * focal_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    # ── 4d. Optimiser + Scheduler ─────────────────────────────────────────────
    base_model = model.module if USE_MULTI_GPU else model
    optimizer  = optim.Adam(base_model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 4e. AMP Scaler ────────────────────────────────────────────────────────
    scaler = GradScaler()

    # ── 4f. Training State ────────────────────────────────────────────────────
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    # ── 4g. Epoch Loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        torch.cuda.reset_peak_memory_stats()   # clean slate for peak tracking

        # ── Train Phase ───────────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                         unit='batch', leave=False)
        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            # OOM FIX #3: free intermediate tensors immediately
            del outputs, loss, images, masks

        avg_train_loss = train_loss_accum / len(train_loader)
        torch.cuda.empty_cache()

        # ── Validation Phase ──────────────────────────────────────────────────
        model.eval()
        val_loss_accum = 0.0

        # OOM FIX #4: incremental metric accumulation — NO full-set np.concat
        iou_sum, dice_sum, metric_n = 0.0, 0.0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs} [Val]  ",
                       unit='batch', leave=False)
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()

                with autocast():
                    outputs  = model(images)
                    val_loss = criterion(outputs, masks)

                val_loss_accum += val_loss.item()

                # Compute metrics on CPU immediately — don't accumulate tensors
                preds_np   = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                targets_np = masks.squeeze(1).cpu().numpy()
                iou_sum, dice_sum, metric_n = update_iou_dice(
                    preds_np, targets_np, iou_sum, dice_sum, metric_n
                )

                del outputs, val_loss, images, masks, preds_np, targets_np

        avg_val_loss = val_loss_accum / len(val_loader)
        val_iou      = iou_sum  / max(metric_n, 1)
        val_dice     = dice_sum / max(metric_n, 1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        mem_alloc = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
        mem_peak  = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"GPU alloc={mem_alloc:.1f}GB peak={mem_peak:.1f}GB"
        )

        # ── Best Model Checkpoint ─────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            state = (model.module if USE_MULTI_GPU else model).state_dict()
            torch.save(state, LOCAL_BEST)
            print(f"   🌟 New best val_loss={best_val_loss:.4f} → saved to {LOCAL_BEST}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # ── Periodic Backup Checkpoint ────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'road_ckpt_ep{epoch:02d}.pth')
            state = (model.module if USE_MULTI_GPU else model).state_dict()
            torch.save({
                'epoch':         epoch,
                'model_state':   state,
                'optim_state':   optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history':       history,
            }, ckpt_path)
            print(f"   💾 Backup → {ckpt_path}")

        # ── Clear GPU cache ────────────────────────────────────────────────────
        torch.cuda.empty_cache()

        # ── Early Stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs "
                  f"(no val improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("          TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Best Validation Loss : {best_val_loss:.4f}")
    print(f"  Best Val IoU         : {max(history['val_iou']):.4f}")
    print(f"  Best Val Dice        : {max(history['val_dice']):.4f}")
    print(f"  Best model saved to  : {LOCAL_BEST}")
    print(f"  Epoch checkpoints    : {CKPT_DIR}/")
    print("=" * 55)

    return history


# =============================================================================
# Section 5 ▸ Entry Point
# =============================================================================

if __name__ == '__main__':
    history = train_road(epochs=NUM_EPOCHS)