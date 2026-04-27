# =============================================================================
# train_building.py  –  Stage 4: Building Footprint Detection (P100)
# =============================================================================
# Architecture : UnetPlusPlus ResNet50 + scse attention
# Loss         : 0.35 * FocalLoss + 0.35 * DiceLoss
#              + 0.15 * BoundaryLoss + 0.15 * SoftDistanceLoss
# Optimizer    : AdamW  lr=1e-4  wd=1e-4
# Scheduler    : OneCycleLR  max_lr=3e-4  epochs=50
# Epochs       : 50   |  Early stopping patience: 8 (based on val IoU)
# Resolution   : 640×640
# Precision    : Mixed (torch.cuda.amp) mandatory
# Environment  : Kaggle P100  (16 GB VRAM, single GPU, 30 GB RAM)
#
# P100 vs T4×2 DDP rationale:
#   • DDP removed — no mp.spawn, no NCCL buffers, no dist.all_reduce overhead
#   • BATCH_SIZE=8 at 640×640 fp16: ~2.5 GB activations, fits P100 easily
#   • num_workers=0, pin_memory=False: RAM headroom over throughput
#   • OneCycleLR steps per batch (not per epoch) — unchanged from DDP version
# =============================================================================

import os
import gc
import ctypes
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import get_massachusetts_building_splits
from models import get_building_model

import cv2
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # suppress TIFF GeoTIFF metadata warnings



# =============================================================================
# Section 1 ▸ Configuration
# =============================================================================

# ✔️ Massachusetts Buildings Dataset (balraj98/massachusetts-buildings-dataset)
DATASET_BASE = '/kaggle/input/datasets/balraj98/massachusetts-buildings-dataset'
IMAGE_DIR    = f'{DATASET_BASE}/tiff/train'
MASK_DIR     = f'{DATASET_BASE}/tiff/train_labels'

BEST_CKPT    = '/kaggle/working/building_model_best.pth'
CKPT_DIR     = '/kaggle/working/building_ckpts'
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
# UnetPlusPlus ResNet50 with scse at 640x640 fp16:
#   ~4 GB activations per sample due to dense nested skip connections.
#   Batch=4 → ~16 GB fits P100's 16 GB safely with AMP.
#   (Plain Unet could fit Batch=8; UnetPlusPlus cannot — 2x skip connections)
BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-4
MAX_LR       = 3e-4      # OneCycleLR peak
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.2
PATIENCE     = 8         # early stopping on val IoU
CKPT_EVERY   = 5
NUM_WORKERS  = 0


# =============================================================================
# Section 2 ▸ Memory Helpers
# =============================================================================

def _trim_heap():
    """Force glibc to return free heap pages to the OS (Linux/Kaggle only)."""
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass  # silently skip on Windows dev machines


def find_latest_building_checkpoint(ckpt_dir: str):
    """Scan CKPT_DIR for the latest building_ckpt_ep*.pth file.

    Returns:
        (path, epoch) tuple if found, else (None, 0)
    """
    if not os.path.isdir(ckpt_dir):
        return None, 0
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith('building_ckpt_ep') and f.endswith('.pth')
    ]
    if not candidates:
        return None, 0
    # Parse epoch numbers from filenames
    def _epoch(name):
        try:
            return int(name.replace('building_ckpt_ep', '').replace('.pth', ''))
        except ValueError:
            return -1
    latest = max(candidates, key=_epoch)
    epoch  = _epoch(latest)
    return os.path.join(ckpt_dir, latest), epoch


# =============================================================================
# Section 3 ▸ Loss Functions
# =============================================================================

class BoundaryLoss(nn.Module):
    """
    Binary cross-entropy weighted 6× on edge pixels.

    Buildings have sharp rectilinear boundaries — BCE alone assigns equal
    weight to all pixels. By up-weighting the thin boundary band (from the
    edge_mask), we force the model to predict crisp outlines rather than
    blob-like fuzzy shapes.

    Args:
        edge_weight : multiplier applied to edge-pixel losses (default 6.0)
    """

    def __init__(self, edge_weight: float = 6.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits: torch.Tensor,
                targets:     torch.Tensor,
                edge_masks:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits : (B, 1, H, W) float32 — raw logits
            targets     : (B, 1, H, W) float32 — binary {0, 1}
            edge_masks  : (B, 1, H, W) float32 — binary {0, 1}
        Returns:
            scalar loss
        """
        pixel_loss = self.bce(pred_logits, targets)          # (B, 1, H, W)
        # Up-weight edge pixels by edge_weight, keep interior pixels at 1.0
        weight_map = 1.0 + (self.edge_weight - 1.0) * edge_masks
        return (pixel_loss * weight_map).mean()


class SoftDistanceLoss(nn.Module):
    """
    Penalise errors proportional to distance from the building boundary.

    Intuition:
        - Errors NEAR the boundary: already handled by BoundaryLoss → less penalty
        - Errors DEEP inside a building (high dist_map value): model left interior
          pixels as background (hollow building prediction) → penalise MORE

    Complements BoundaryLoss: together they force both sharp outlines AND
    solid interior fill with no holes.

    Args:
        pred_logits : (B, 1, H, W) float32 — raw logits
        dist_maps   : (B, 1, H, W) float32 — normalised distance transform [0,1]
                      (0 = boundary, 1 = building centre)
    Returns:
        scalar loss
    """

    def forward(self, pred_logits: torch.Tensor,
                dist_maps: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred_logits)               # (B, 1, H, W)
        # Ground truth binary: any pixel with dist_map > 0 is inside a building
        gt_binary = (dist_maps > 0).float()                  # (B, 1, H, W)
        error     = torch.abs(pred_prob - gt_binary)         # per-pixel error
        # Weight errors by (1 + dist_map) — larger penalty further from boundary
        weighted  = error * (1.0 + dist_maps)
        return weighted.mean()


def build_loss_fns(device: torch.device):
    """Instantiate all four loss components on the correct device."""
    focal    = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
    dice     = smp.losses.DiceLoss(mode='binary', from_logits=True)
    boundary = BoundaryLoss(edge_weight=6.0).to(device)
    soft_dist = SoftDistanceLoss().to(device)
    return focal, dice, boundary, soft_dist


def compute_total_loss(logits:     torch.Tensor,
                       masks:      torch.Tensor,
                       edge_masks: torch.Tensor,
                       dist_maps:  torch.Tensor,
                       focal, dice, boundary, soft_dist) -> torch.Tensor:
    """
    Four-component loss:
        0.35 * FocalLoss   — class imbalance (background >> buildings)
        0.35 * DiceLoss    — overall shape quality
        0.15 * BoundaryLoss — sharp crisp edges
        0.15 * SoftDist    — solid interior fill, no holes

    Args:
        logits     : (B, 1, H, W) raw logits
        masks      : (B, 1, H, W) binary float32 {0,1}
        edge_masks : (B, 1, H, W) binary float32 {0,1}
        dist_maps  : (B, 1, H, W) float32 [0,1]
    """
    fl = focal(logits, masks)
    dl = dice(logits, masks)
    bl = boundary(logits, masks, edge_masks)
    sd = soft_dist(logits, dist_maps)
    return 0.35 * fl + 0.35 * dl + 0.15 * bl + 0.15 * sd


# =============================================================================
# Section 4 ▸ Metric Helper
# =============================================================================

def compute_iou_batch(logits: torch.Tensor,
                      masks:  torch.Tensor,
                      threshold: float = 0.5) -> float:
    """
    Compute mean binary IoU over a batch.

    Args:
        logits : (B, 1, H, W) raw logits
        masks  : (B, 1, H, W) binary float32
    Returns:
        mean IoU as Python float
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks).clamp(max=1).sum(dim=(1, 2, 3))
    # Skip images where BOTH prediction and GT are empty — avoids false 1.0
    valid = union > 0
    if valid.sum() == 0:
        return 0.0
    iou = (inter[valid] / union[valid]).mean().item()
    return iou


# =============================================================================
# Section 5 ▸ GPU Setup  (P100 — single GPU)
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduce memory fragmentation — critical for UnetPlusPlus dense skip connections
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# cuDNN: disable benchmark mode to prevent RAM creep from algo-probe caching
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"🖥️  GPU: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
    print(f"   AMP (fp16) ENABLED")
else:
    print("⚠️  No GPU detected — running on CPU (will be slow!)")


# =============================================================================
# Section 6 ▸ Main Training Function
# =============================================================================

def train_building(epochs: int = NUM_EPOCHS):
    # ── 6a. Data ──────────────────────────────────────────────────────────────
    # Massachusetts dataset has pre-defined splits — use them directly
    # train/ + train_labels/  →  training set
    # val/   + val_labels/    →  validation set
    val_image_dir = IMAGE_DIR.replace('/train', '/val')
    val_mask_dir  = MASK_DIR.replace('/train_labels', '/val_labels')

    from dataset import MassachusettsBuildingDataset, building_train_transform, building_val_transform
    train_ds = MassachusettsBuildingDataset(
        IMAGE_DIR, MASK_DIR, transform=building_train_transform
    )
    val_ds = MassachusettsBuildingDataset(
        val_image_dir, val_mask_dir, transform=building_val_transform
    )
    print(f"📂 Massachusetts Building → Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print(f"\n🚀 Building Detection | P100 Single-GPU Training")
    print(f"   Batch={BATCH_SIZE} | Resolution=640×640 | Epochs={epochs}")
    print(f"   Train samples={len(train_ds)} | Val samples={len(val_ds)}\n")

    # ── 6b. Model ─────────────────────────────────────────────────────────────
    model = get_building_model().to(device)

    # ── 6c. Loss functions ────────────────────────────────────────────────────
    focal, dice, boundary, soft_dist = build_loss_fns(device)

    # ── 6d. Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: warms up LR from LR→max_lr over first 30% of training,
    # then anneals to LR/10. Steps per BATCH (not per epoch).
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
    )

    scaler = torch.amp.GradScaler('cuda')

    # ── 6e. Training state + Auto-Resume ─────────────────────────────────────
    best_iou          = 0.0
    epochs_no_improve = 0
    start_epoch       = 1
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

    resume_path, resume_epoch = find_latest_building_checkpoint(CKPT_DIR)
    if resume_path:
        print(f"\n⏳ Resuming from checkpoint: {resume_path}")
        ckpt_data = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt_data['model_state'])
        optimizer.load_state_dict(ckpt_data['optim_state'])
        scheduler.load_state_dict(ckpt_data['sched_state'])
        scaler.load_state_dict(ckpt_data['scaler_state'])
        best_iou          = ckpt_data.get('best_iou', 0.0)
        epochs_no_improve = ckpt_data.get('epochs_no_improve', 0)
        history           = ckpt_data.get('history', history)
        start_epoch       = resume_epoch + 1
        print(f"   Resumed at epoch {resume_epoch} | best_iou={best_iou:.4f} | "
              f"no_improve={epochs_no_improve}")
    else:
        print("\n✔️ No checkpoint found — starting from scratch.")

    # ── 6f. Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                         unit='batch', leave=False)

        for images, masks, edge_masks, dist_maps in train_bar:
            # images:     (B, 3, 640, 640) float32
            # masks:      (B, 640, 640)    float32 — unsqueeze below
            # edge_masks: (B, 640, 640)    float32 — unsqueeze below
            # dist_maps:  (B, 640, 640)    float32 — unsqueeze below
            images     = images.to(device, non_blocking=True)
            masks      = masks.to(device, non_blocking=True).unsqueeze(1)
            edge_masks = edge_masks.to(device, non_blocking=True).unsqueeze(1)
            dist_maps  = dist_maps.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(images)     # (B, 1, 640, 640)
                loss   = compute_total_loss(
                    logits, masks, edge_masks, dist_maps,
                    focal, dice, boundary, soft_dist
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # OneCycleLR steps per BATCH

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            del images, masks, edge_masks, dist_maps, logits, loss

        avg_train_loss = train_loss / len(train_loader)
        torch.cuda.empty_cache()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_accum = 0.0
        iou_sum        = 0.0
        n_batches      = 0

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch:02d}/{epochs} [Val]  ",
                       unit='batch', leave=False)

        with torch.no_grad():
            for images, masks, edge_masks, dist_maps in val_bar:
                images     = images.to(device, non_blocking=True)
                masks      = masks.to(device, non_blocking=True).unsqueeze(1)
                edge_masks = edge_masks.to(device, non_blocking=True).unsqueeze(1)
                dist_maps  = dist_maps.to(device, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast('cuda'):
                    logits   = model(images)
                    val_loss = compute_total_loss(
                        logits, masks, edge_masks, dist_maps,
                        focal, dice, boundary, soft_dist
                    )

                val_loss_accum += val_loss.item()
                iou_sum        += compute_iou_batch(logits, masks)
                n_batches      += 1

                del images, masks, edge_masks, dist_maps, logits, val_loss

        avg_val_loss = val_loss_accum / max(n_batches, 1)
        val_iou      = iou_sum / max(n_batches, 1)

        elapsed    = time.time() - t0
        mem_alloc  = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
        mem_peak   = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s | "
            f"GPU alloc={mem_alloc:.1f}GB peak={mem_peak:.1f}GB"
        )

        # ── Best model checkpoint ──────────────────────────────────────────────
        if val_iou > best_iou:
            best_iou          = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"   🌟 New best IoU={best_iou:.4f} → {BEST_CKPT}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement ({epochs_no_improve}/{PATIENCE})")

        # ── Periodic backup checkpoint ─────────────────────────────────────────
        if epoch % CKPT_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'building_ckpt_ep{epoch:02d}.pth')
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optim_state':    optimizer.state_dict(),
                'sched_state':    scheduler.state_dict(),
                'scaler_state':   scaler.state_dict(),
                'best_iou':       best_iou,
                'epochs_no_improve': epochs_no_improve,
                'history':        history,
            }, ckpt_path)
            print(f"   💾 Backup → {ckpt_path}")

        # ── End-of-epoch memory hygiene ────────────────────────────────────────
        torch.cuda.empty_cache()
        gc.collect()
        _trim_heap()

        # ── Early stopping ─────────────────────────────────────────────────────
        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch} "
                  f"(no val IoU improvement for {PATIENCE} epochs).")
            break

    # ── Final Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("          BUILDING TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Best Val IoU  : {best_iou:.4f}")
    print(f"  Best model    : {BEST_CKPT}")
    print(f"  Checkpoints   : {CKPT_DIR}/")
    print("=" * 55)

    return history


# =============================================================================
# Section 7 ▸ Entry Point
# =============================================================================

if __name__ == '__main__':
    history = train_building(epochs=NUM_EPOCHS)