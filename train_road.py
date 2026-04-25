# =============================================================================
# train_road.py  –  Road Extraction Training Script (Kaggle P100)
# =============================================================================
# Architecture : DeepLabV3+ ResNet34  (encoder_output_stride=16)
# Loss         : 0.5 * FocalLoss  +  0.5 * DiceLoss
# Optimizer    : Adam  lr=1e-4
# Scheduler    : CosineAnnealingLR  T_max=num_epochs
# Epochs       : 35  |  Early stopping patience=5  (val loss)
# Precision    : Mixed (torch.cuda.amp  GradScaler + autocast)
# Environment  : Kaggle P100  (16 GB VRAM, single GPU, 30 GB RAM)
#
# P100 vs T4×2 rationale:
#   • Single GPU  → no DataParallel/DDP overhead, no NCCL buffers
#   • Flat RAM profile: baseline ~15 GB vs T4×2 ~24–28 GB  (much more headroom)
#   • BATCH_SIZE=12  (safe at 512×512 fp16; T4×2 was capped at 8 due to DP overhead)
#   • num_workers=0, pin_memory=False  (RAM headroom first, speed second)
#   • malloc_trim(0) + gc.collect() after every epoch prevents heap creep
# =============================================================================

import os
import gc
import ctypes
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


def _ram_gb() -> float:
    """Return current process RSS in GB (requires psutil)."""
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _trim_heap():
    """
    Force glibc to return free heap pages to the OS.

    Python's memory allocator (glibc malloc) holds freed memory speculatively
    for reuse rather than returning it to the OS.  Over 35 epochs of loading
    12K+ satellite images this fragmented heap grows by several GB even though
    gc.collect() reports 0 uncollectable objects.

    malloc_trim(0) instructs glibc to release all releasable free pages
    immediately.  On Linux (Kaggle) libc.so.6 is always present.
    This is a no-op on non-Linux platforms — the try/except makes it safe.
    """
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass  # non-Linux (Windows dev machines) — silently skip

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

# ── Batch size ────────────────────────────────────────────────────────────────
# P100 has 16 GB VRAM (single GPU). DeepLabV3+ ResNet34 at 512×512 fp16
# uses ~1.5 GB activations per sample. Batch=12 → ~18 GB activations in fp16,
# which fits comfortably with optimizer states + gradients in 16 GB.
# We keep num_workers=0 to preserve RAM headroom (each worker costs ~1.5 GB).
BATCH_SIZE = 12

NUM_EPOCHS          = 35
LR                  = 1e-4
VAL_RATIO           = 0.2
EARLY_STOP_PATIENCE = 5
CHECKPOINT_EVERY    = 5


# =============================================================================
# Section 2 ▸ GPU Setup  (P100 — single GPU)
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── cuDNN memory control ──────────────────────────────────────────────────────
# benchmark=False : cuDNN picks a deterministic algorithm on first call and
#   reuses it forever.  benchmark=True (the default) probes EVERY convolution
#   shape variant it encounters and caches the fastest algorithm in RAM.
#   With 35 epochs × many batches this cache grows continuously.
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"🖥️  GPU: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
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

    # ── DataLoaders ────────────────────────────────────────────────────────────
    # num_workers=0: ALL data loading runs in the main process.
    #
    # WHY NOT workers > 0 despite slower GPU utilisation:
    #   Each DataLoader worker spawns a FULL Python subprocess that imports
    #   PyTorch, cv2, albumentations, and the dataset class.  On Kaggle, each
    #   worker costs ~1.5–2 GB of RAM at startup.  With 2 workers that is an
    #   immediate +3–4 GB hit on top of the existing ~24 GB baseline, pushing
    #   the total to ~28 GB and leaving only ~2 GB headroom — any training
    #   tensor allocation then triggers the kernel OOM killer.
    #
    # pin_memory=False: pinned memory is only beneficial when workers write
    #   into it from subprocesses (irrelevant at num_workers=0).
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

    # ── 4b. Model  (single GPU — P100) ───────────────────────────────────────
    model = get_road_model().to(device)
    print(f"🩺 RAM after model   : {_ram_gb():.2f} GB")

    # ── 4c. Combined Loss ─────────────────────────────────────────────────────
    focal_loss = smp.losses.FocalLoss(mode='binary')
    dice_loss  = smp.losses.DiceLoss(mode='binary')

    def criterion(logits, targets):
        return 0.5 * focal_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    # ── 4d. Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            torch.save(model.state_dict(), LOCAL_BEST)
            print(f"   🌟 New best val_loss={best_val_loss:.4f} → saved to {LOCAL_BEST}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # ── Periodic Backup Checkpoint ────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'road_ckpt_ep{epoch:02d}.pth')
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optim_state':   optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history':       history,
            }, ckpt_path)
            print(f"   💾 Backup → {ckpt_path}")

        # ── End-of-epoch memory hygiene ───────────────────────────────────────
        # Three-layer RAM reclamation to prevent the 24→30 GB creep:
        #
        # 1. empty_cache()  : returns cached GPU tensors to CUDA allocator
        # 2. gc.collect()   : breaks Python reference cycles
        # 3. malloc_trim(0) : forces glibc to return free heap pages to the OS
        #    (Python holds freed RAM speculatively — this is what causes the
        #     slow per-epoch RAM growth even after gc.collect() succeeds)
        torch.cuda.empty_cache()
        gc.collect()
        _trim_heap()
        print(f"   🩺 RAM end-of-epoch {epoch:02d}: {_ram_gb():.2f} GB")

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