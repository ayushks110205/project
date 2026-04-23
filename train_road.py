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
#   • pin_memory=True, num_workers=4 for Kaggle DataLoader optimisation
#   • tqdm progress bars with live loss display
#   • Epoch-end IoU + Dice printout from val loop
#   • nn.DataParallel for T4 ×2 (30 GB total VRAM)
# =============================================================================

# ── Kaggle T4 ×2 Notes ────────────────────────────────────────────────────────
# • DataParallel is used when 2 GPUs are detected (splits batch across both).
# • All inputs  : /kaggle/input/  (read-only)
# • All outputs : /kaggle/working/ (persisted after session ends)
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ── Local imports ─────────────────────────────────────────────────────────────
from dataset import get_road_splits
from models import get_road_model


# =============================================================================
# Section 1 ▸ Configuration
# =============================================================================

# --- Path & Hyperparameter Constants ---
BASE_PATH  = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset'

# Both point to the same folder because sat.jpg and _mask.png live together
IMAGE_DIR  = f'{BASE_PATH}/train'
MASK_DIR   = f'{BASE_PATH}/train'

# Kaggle output paths  (no Google Drive on Kaggle)
LOCAL_BEST = '/kaggle/working/road_model_best.pth'   # best val-loss checkpoint
CKPT_DIR   = '/kaggle/working/road_ckpts'             # periodic epoch backups
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16        # T4 ×2 = 30 GB total; 8 samples per GPU with AMP
NUM_EPOCHS = 35
LR         = 1e-4
VAL_RATIO  = 0.2       # 80% train / 20% val

# Training controls
EARLY_STOP_PATIENCE = 5
CHECKPOINT_EVERY    = 5   # save a numbered backup every N epochs


# =============================================================================
# Section 2 ▸ GPU / Multi-GPU Setup
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
USE_MULTI_GPU = n_gpus > 1

if torch.cuda.is_available():
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"🖥️  GPU {i}: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
    print(f"   DataParallel: {'YES (×' + str(n_gpus) + ')' if USE_MULTI_GPU else 'NO (single GPU)'}")
else:
    print("⚠️  No GPU detected — running on CPU (will be slow!)")


# =============================================================================
# Section 3 ▸ Metric Helpers
# =============================================================================

def compute_iou_dice(preds_np: np.ndarray, targets_np: np.ndarray,
                     threshold: float = 0.5):
    """
    Compute mean IoU and mean Dice over a batch of numpy arrays.

    Args:
        preds_np   : sigmoid probability maps  [B, H, W]
        targets_np : binary ground-truth masks [B, H, W]
        threshold  : binarisation cutoff

    Returns:
        mean_iou (float), mean_dice (float)
    """
    binary  = (preds_np > threshold).astype(np.uint8)
    targets = targets_np.astype(np.uint8)

    ious, dices = [], []
    for p, t in zip(binary, targets):
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
        dices.append((2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))

    return float(np.mean(ious)), float(np.mean(dices))


# =============================================================================
# Section 4 ▸ Main Training Function
# =============================================================================

def train_road(epochs: int = NUM_EPOCHS):
    # ── 4a. Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = get_road_splits(IMAGE_DIR, MASK_DIR, val_ratio=VAL_RATIO)

    # num_workers=4: Kaggle has more CPU cores than Colab free tier
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"\n🚀 Road Training | Device: {device}")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"   Epochs: {epochs} | Batch size: {BATCH_SIZE} | LR: {LR}\n")

    # ── 4b. Model  +  DataParallel ────────────────────────────────────────────
    model = get_road_model().to(device)
    if USE_MULTI_GPU:
        # DataParallel splits the batch across both GPUs along dim=0.
        # Each GPU gets (B/2, 3, 512, 512); gradients are summed automatically.
        # Always access real weights via model.module when saving.
        model = nn.DataParallel(model)
        print(f"   Model wrapped in DataParallel across {n_gpus} GPUs")

    # ── 4c. Combined Loss ─────────────────────────────────────────────────────
    # FocalLoss: handles extreme class imbalance (background >> road pixels)
    # DiceLoss:  penalises thin-road misses; optimises F1/overlap directly
    focal_loss = smp.losses.FocalLoss(mode='binary')
    dice_loss  = smp.losses.DiceLoss(mode='binary')

    def criterion(logits, targets):
        return 0.5 * focal_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    # ── 4d. Optimiser + Scheduler ─────────────────────────────────────────────
    # Build optimizer from the underlying module so DataParallel wrapper doesn't
    # interfere with weight decay / parameter groups.
    base_model = model.module if USE_MULTI_GPU else model
    optimizer  = optim.Adam(base_model.parameters(), lr=LR)
    # CosineAnnealingLR: smoothly decays LR → near-zero over T_max epochs.
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 4e. AMP Scaler (Mixed Precision) ─────────────────────────────────────
    scaler = GradScaler()

    # ── 4f. Training State ────────────────────────────────────────────────────
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    # ── 4g. Epoch Loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # ── Train Phase ───────────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                         unit='batch', leave=False)
        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).unsqueeze(1)  # [B,1,H,W]

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

        avg_train_loss = train_loss_accum / len(train_loader)

        # ── Validation Phase ──────────────────────────────────────────────────
        model.eval()
        val_loss_accum  = 0.0
        all_preds, all_targets = [], []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs} [Val]  ",
                       unit='batch', leave=False)
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True).unsqueeze(1)

                with autocast():
                    outputs  = model(images)
                    val_loss = criterion(outputs, masks)

                val_loss_accum += val_loss.item()

                preds_np   = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                targets_np = masks.squeeze(1).cpu().numpy()
                all_preds.append(preds_np)
                all_targets.append(targets_np)

        avg_val_loss = val_loss_accum / len(val_loader)

        # Epoch-level metrics
        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_iou, val_dice = compute_iou_dice(all_preds, all_targets)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        # GPU memory display
        mem_gb = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"GPU: {mem_gb:.2f} GB"
        )

        # ── Best Model Checkpoint (based on VAL loss) ─────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            # Unwrap DataParallel before saving — checkpoint portable on any machine
            state = (model.module if USE_MULTI_GPU else model).state_dict()
            torch.save(state, LOCAL_BEST)
            print(f"   🌟 New best val_loss={best_val_loss:.4f} → saved to {LOCAL_BEST}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # ── Periodic Backup Checkpoint (every N epochs) ───────────────────────
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
            print(f"   💾 Backup checkpoint saved → {ckpt_path}")

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