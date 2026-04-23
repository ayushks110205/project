# =============================================================================
# train_road.py  –  Road Extraction Training Script (Colab T4 Optimised)
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
#   • Checkpoint every 5 epochs as Colab disconnect insurance
#   • torch.cuda.empty_cache() between epochs to prevent OOM
#   • pin_memory=True, num_workers=2 for T4 DataLoader optimisation
#   • tqdm progress bars with live loss display
#   • Epoch-end IoU + Dice printout from val loop
#   • GPU memory check at startup
# =============================================================================

# ── Colab Session Keepalive Tip ───────────────────────────────────────────────
# Paste this in a Colab JS console to prevent idle disconnects:
#   setInterval(() => { document.querySelector("#connect").click() }, 60000)
# Or install the `colab-keep-alive` Chrome extension.
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ── Local imports (class names now consistent across all files) ──────────────
from dataset import get_road_splits                  # ← fixed: was RoadDataset
from models import get_road_model


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Paths
# ✅ Use the full path that your test cell successfully validated
BASE_PATH    = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset'

# 📍 Both point to the same folder because sat.jpg and _mask.png live together
IMAGE_DIR    = f'{BASE_PATH}/train'
MASK_DIR     = f'{BASE_PATH}/train'

# ✅ Paths for saving road-specific outputs
SAVE_PATH    = '/kaggle/working/road_model_best.pth'
CKPT_DIR     = '/kaggle/working/road_ckpts'

# Hyperparameters
BATCH_SIZE  = 16        # Safe for T4 at 512×512 with AMP fp16
NUM_EPOCHS  = 35        # Cosine schedule needs room to breathe vs hard cutoff
LR          = 1e-4
VAL_RATIO   = 0.2       # 80% train / 20% val

# Training controls
EARLY_STOP_PATIENCE = 5
CHECKPOINT_EVERY    = 5  # Save a numbered backup every N epochs


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ GPU Memory Check
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram  = (torch.cuda.get_device_properties(0).total_memory
                  - torch.cuda.memory_allocated()) / 1e9
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📦  VRAM — Total: {total_vram:.1f} GB | Free: {free_vram:.1f} GB")
else:
    print("⚠️  No GPU detected — running on CPU (will be slow!)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    binary = (preds_np > threshold).astype(np.uint8)
    targets = targets_np.astype(np.uint8)

    ious, dices = [], []
    for p, t in zip(binary, targets):
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
        dices.append((2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))

    return float(np.mean(ious)), float(np.mean(dices))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_road(epochs: int = NUM_EPOCHS):
    # ── 4a. Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = get_road_splits(IMAGE_DIR, MASK_DIR, val_ratio=VAL_RATIO)

    # pin_memory=True speeds up CPU→GPU transfers on T4
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\n🚀 Road Training | Device: {device}")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"   Epochs: {epochs} | Batch size: {BATCH_SIZE} | LR: {LR}\n")

    # ── 4b. Model ─────────────────────────────────────────────────────────────
    model = get_road_model().to(device)

    # ── 4c. Combined Loss ─────────────────────────────────────────────────────
    # FocalLoss: handles extreme class imbalance (background >> road pixels)
    # DiceLoss:  penalises thin-road misses; optimises F1/overlap directly
    # 50/50 blend gives FocalLoss's hard-example mining + DiceLoss's shape fidelity
    focal_loss = smp.losses.FocalLoss(mode='binary')
    dice_loss  = smp.losses.DiceLoss(mode='binary')

    def criterion(logits, targets):
        return 0.5 * focal_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    # ── 4d. Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # CosineAnnealingLR: smoothly decays LR from LR → near-zero over T_max epochs.
    # This lets the model escape local minima early then fine-tune precisely.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 4e. AMP Scaler (Mixed Precision) ─────────────────────────────────────
    # GradScaler prevents gradient underflow with fp16 computation.
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

            optimizer.zero_grad()

            # autocast: runs forward pass in fp16 (saves ~50% VRAM on T4)
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, masks)

            # Scaled backward pass then unscale + clip gradients
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
        val_loss_accum = 0.0
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

                # Collect for IoU/Dice computation
                preds_np   = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                targets_np = masks.squeeze(1).cpu().numpy()
                all_preds.append(preds_np)
                all_targets.append(targets_np)

        avg_val_loss = val_loss_accum / len(val_loader)

        # Compute epoch-level metrics
        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_iou, val_dice = compute_iou_dice(all_preds, all_targets)

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ── Best Model Checkpoint (based on VAL loss) ─────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), LOCAL_BEST)
            if os.path.exists('/content/drive/MyDrive'):
                torch.save(model.state_dict(), DRIVE_PATH)
                print(f"   🌟 New best val_loss={best_val_loss:.4f} → saved to Drive")
            else:
                print(f"   🌟 New best val_loss={best_val_loss:.4f} → saved locally")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # ── Periodic Backup Checkpoint (every N epochs) ───────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = f"/content/drive/MyDrive/datasets/road_ckpt_ep{epoch:02d}.pth" \
                        if os.path.exists('/content/drive/MyDrive') \
                        else f"road_ckpt_ep{epoch:02d}.pth"
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optim_state':    optimizer.state_dict(),
                'best_val_loss':  best_val_loss,
                'history':        history,
            }, ckpt_path)
            print(f"   💾 Backup checkpoint saved → {ckpt_path}")

        # ── Clear GPU cache between epochs (prevents Colab OOM crashes) ───────
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
    print(f"  Model saved to       : {LOCAL_BEST}")
    print("=" * 55)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history = train_road(epochs=NUM_EPOCHS)