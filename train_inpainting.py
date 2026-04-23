# =============================================================================
# train_inpainting.py  –  Stage 2 Road Mask Inpainting Training Loop
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TWO-STAGE PIPELINE OVERVIEW                                            │
# │                                                                         │
# │  STAGE 1: Satellite Image  →  Road Mask                                 │
# │           Model: DeepLabV3+ ResNet34 (train_road.py)                   │
# │           Input : (B, 3, 512, 512) RGB satellite                        │
# │           Output: (B, 1, 512, 512) binary road mask                     │
# │                                                                         │
# │  STAGE 2: Incomplete Road Mask  →  Complete Road Mask                   │
# │           Model: Partial Conv U-Net (this file)                         │
# │           Input : (B, 2, 512, 512) [corrupted_mask, hole_mask]          │
# │           Output: (B, 1, 512, 512) filled road mask                     │
# │                                                                         │
# │  INFERENCE: Run Stage 1 first, pipe its output into Stage 2             │
# │             (see pipeline.py for the chained inference wrapper)         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Training setup (T4 free-tier optimised):
#   • Batch size   : 8   (smaller than Stage 1 — partial conv is VRAM-heavy)
#   • AMP          : torch.cuda.amp GradScaler + autocast  (mandatory for T4)
#   • Optimizer    : Adam lr=2e-4, weight_decay=1e-5
#   • Scheduler    : ReduceLROnPlateau(patience=4, factor=0.5)
#   • Gradient clip: max_norm=1.0
#   • Early stop   : patience=8 epochs on val L_hole
#   • Checkpoints  : every 5 epochs (Colab disconnect insurance)
# =============================================================================

import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
from inpainting_dataset import get_inpainting_splits
from inpainting_model   import get_inpainting_model
from inpainting_losses  import InpaintingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Configuration
# ─────────────────────────────────────────────────────────────────────────────

MASK_DIR    = 'datasets/train'                  # Same folder as Stage 1
IMAGE_SIZE  = 512

DRIVE_BEST  = '/content/drive/MyDrive/datasets/inpainting_best.pth'
DRIVE_CKPT  = '/content/drive/MyDrive/datasets/inpainting_ckpt_ep{:02d}.pth'
LOCAL_BEST  = 'inpainting_best.pth'

BATCH_SIZE  = 8           # Smaller than Stage 1 due to partial conv overhead
NUM_EPOCHS  = 40
LR          = 2e-4
WD          = 1e-5
VAL_RATIO   = 0.20

EARLY_STOP_PATIENCE = 8
CHECKPOINT_EVERY    = 5

# Loss weights (must match InpaintingLoss defaults for clarity)
LAMBDA_VALID = 1.0
LAMBDA_HOLE  = 6.0
LAMBDA_PERC  = 0.05
LAMBDA_CONN  = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ GPU Memory Check
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram  = (torch.cuda.get_device_properties(0).total_memory
                  - torch.cuda.memory_allocated()) / 1e9
    print(f"🖥️  GPU : {torch.cuda.get_device_name(0)}")
    print(f"📦 VRAM: Total {total_vram:.1f}GB | Free {free_vram:.1f}GB")
    print("⚡ AMP (fp16) is ENABLED — mandatory for T4 free tier\n")
else:
    print("⚠️  No CUDA GPU detected — running on CPU (training will be very slow)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Quick Metric Helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_hole_iou(pred_np:      np.ndarray,
                     target_np:    np.ndarray,
                     hole_mask_np: np.ndarray,
                     threshold:    float = 0.5) -> float:
    """
    IoU computed ONLY inside the hole regions.
    This is the primary Stage 2 metric — how well did we reconstruct the
    road network in the parts we didn't know?

    Args:
        pred_np      : (B, H, W) sigmoid probabilities
        target_np    : (B, H, W) ground-truth {0,1}
        hole_mask_np : (B, H, W) — 0=hole, 1=valid; we want 0-regions
        threshold    : binarisation threshold

    Returns:
        mean hole IoU over batch (float)
    """
    pred_bin = (pred_np > threshold).astype(np.uint8)
    target   = target_np.astype(np.uint8)
    hole     = (hole_mask_np < 0.5).astype(np.uint8)   # 1 = hole region

    ious = []
    for p, t, h in zip(pred_bin, target, hole):
        # Only evaluate inside hole
        p_h = p  * h
        t_h = t  * h
        inter = np.logical_and(p_h, t_h).sum()
        union = np.logical_or(p_h, t_h).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))

    return float(np.mean(ious))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_inpainting(epochs: int = NUM_EPOCHS):
    # ── 4a. Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = get_inpainting_splits(MASK_DIR, IMAGE_SIZE, VAL_RATIO)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"🚀 Stage 2 Inpainting Training | Device: {device}")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"   Batch: {BATCH_SIZE} | Epochs: {epochs} | LR: {LR}\n")

    # ── 4b. Model ─────────────────────────────────────────────────────────────
    model = get_inpainting_model(base_channels=64).to(device)

    # ── 4c. Loss ──────────────────────────────────────────────────────────────
    loss_fn = InpaintingLoss(
        lambda_valid=LAMBDA_VALID,
        lambda_hole=LAMBDA_HOLE,
        lambda_perc=LAMBDA_PERC,
        lambda_conn=LAMBDA_CONN,
        device=device,
    )

    # ── 4d. Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5, verbose=True
    )

    # ── 4e. AMP Scaler ────────────────────────────────────────────────────────
    scaler = GradScaler()

    # ── 4f. Training State ────────────────────────────────────────────────────
    best_val_hole     = float('inf')
    epochs_no_improve = 0
    history = {
        'train_total': [], 'train_hole': [],
        'val_total':   [], 'val_hole':   [],
        'val_iou':     [],
    }

    # ── 4g. Epoch Loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # ── Train Phase ───────────────────────────────────────────────────────
        model.train()
        t_total, t_hole, t_valid, t_perc, t_conn = 0., 0., 0., 0., 0.

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                         unit='batch', leave=False)

        for corrupted, hole_mask, complete in train_bar:
            # Shapes:
            #   corrupted : (B, 1, 512, 512)
            #   hole_mask : (B, 1, 512, 512)
            #   complete  : (B, 1, 512, 512)

            corrupted = corrupted.to(device, non_blocking=True)
            hole_mask = hole_mask.to(device, non_blocking=True)
            complete  = complete.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                # Forward: (B,1,H,W), (B,1,H,W) → (B,1,H,W)
                pred   = model(corrupted, hole_mask)
                losses = loss_fn(pred, complete, hole_mask)

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_total += losses['total'].item()
            t_hole  += losses['hole'].item()
            t_valid += losses['valid'].item()
            t_perc  += losses['perceptual'].item()
            t_conn  += losses['connectivity'].item()

            train_bar.set_postfix(
                total=f"{losses['total'].item():.3f}",
                hole=f"{losses['hole'].item():.3f}",
                conn=f"{losses['connectivity'].item():.3f}",
            )

        n_train = len(train_loader)
        avg_train_total = t_total / n_train
        avg_train_hole  = t_hole  / n_train

        # ── Validation Phase ──────────────────────────────────────────────────
        model.eval()
        v_total, v_hole = 0., 0.
        all_preds, all_targets, all_holes = [], [], []

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch:02d}/{epochs} [Val]  ",
                       unit='batch', leave=False)

        with torch.no_grad():
            for corrupted, hole_mask, complete in val_bar:
                corrupted = corrupted.to(device, non_blocking=True)
                hole_mask = hole_mask.to(device, non_blocking=True)
                complete  = complete.to(device, non_blocking=True)

                with autocast():
                    pred   = model(corrupted, hole_mask)
                    losses = loss_fn(pred, complete, hole_mask)

                v_total += losses['total'].item()
                v_hole  += losses['hole'].item()

                all_preds.append(
                    torch.sigmoid(pred).squeeze(1).cpu().numpy()
                    if pred.shape[1] == 1 else pred.squeeze(1).cpu().numpy()
                )
                all_targets.append(complete.squeeze(1).cpu().numpy())
                all_holes.append(hole_mask.squeeze(1).cpu().numpy())

        n_val = len(val_loader)
        avg_val_total = v_total / n_val
        avg_val_hole  = v_hole  / n_val

        # Compute hole IoU over entire val set
        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_holes   = np.concatenate(all_holes,   axis=0)
        hole_iou    = compute_hole_iou(all_preds, all_targets, all_holes)

        # Step scheduler based on val hole loss
        scheduler.step(avg_val_hole)
        current_lr = optimizer.param_groups[0]['lr']

        # ── GPU memory report ─────────────────────────────────────────────────
        if torch.cuda.is_available():
            used_vram = torch.cuda.memory_allocated() / 1e9
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            vram_str  = f" | VRAM used={used_vram:.1f}GB peak={peak_vram:.1f}GB"
        else:
            vram_str  = ""

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_total'].append(avg_train_total)
        history['train_hole'].append(avg_train_hole)
        history['val_total'].append(avg_val_total)
        history['val_hole'].append(avg_val_hole)
        history['val_iou'].append(hole_iou)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train [total={avg_train_total:.4f} hole={avg_train_hole:.4f}] | "
            f"Val [total={avg_val_total:.4f} hole={avg_val_hole:.4f}] | "
            f"Hole IoU={hole_iou:.4f} | LR={current_lr:.1e}"
            f"{vram_str}"
        )

        # ── Best Model Checkpoint (by VAL L_hole) ────────────────────────────
        if avg_val_hole < best_val_hole:
            best_val_hole     = avg_val_hole
            epochs_no_improve = 0

            torch.save(model.state_dict(), LOCAL_BEST)
            if os.path.exists('/content/drive/MyDrive'):
                torch.save(model.state_dict(), DRIVE_BEST)
                print(f"   🌟 New best val_hole={best_val_hole:.4f} → saved to Drive")
            else:
                print(f"   🌟 New best val_hole={best_val_hole:.4f} → saved locally")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        # ── Periodic Checkpoint ───────────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = DRIVE_CKPT.format(epoch) \
                        if os.path.exists('/content/drive/MyDrive') \
                        else f'inpainting_ckpt_ep{epoch:02d}.pth'
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optim_state':   optimizer.state_dict(),
                'best_val_hole': best_val_hole,
                'history':       history,
            }, ckpt_path)
            print(f"   💾 Checkpoint → {ckpt_path}")

        # ── Clear GPU cache (prevents Colab OOM) ─────────────────────────────
        torch.cuda.empty_cache()

        # ── Early Stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping after epoch {epoch} "
                  f"(no val_hole improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("          STAGE 2 INPAINTING TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val L_hole  : {best_val_hole:.4f}")
    print(f"  Best Hole IoU    : {max(history['val_iou']):.4f}")
    print(f"  Model saved to   : {LOCAL_BEST}")
    print("=" * 60)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history = train_inpainting(epochs=NUM_EPOCHS)
