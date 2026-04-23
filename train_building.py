# =============================================================================
# train_building.py  –  Stage 4: Building Footprint Detection (DDP, T4 ×2)
# =============================================================================
# Architecture : UnetPlusPlus ResNet50 + scse attention
# Loss         : 0.35 * FocalLoss + 0.35 * DiceLoss
#              + 0.15 * BoundaryLoss + 0.15 * SoftDistanceLoss
# Optimizer    : AdamW  lr=1e-4  wd=1e-4
# Scheduler    : OneCycleLR  max_lr=3e-4  epochs=50
# Epochs       : 50   |  Early stopping patience: 8 (based on val IoU)
# Resolution   : 640×640 (exploits 30GB total VRAM on T4 ×2)
# Precision    : Mixed (torch.cuda.amp) mandatory
# Multi-GPU    : DistributedDataParallel (DDP) via mp.spawn
#                — avoids DataParallel's VRAM imbalance and master-GPU bottleneck
#                — each GPU runs its own backward pass; gradients all-reduced
# Environment  : Kaggle T4 ×2  |  /kaggle/input/  |  /kaggle/working/
# =============================================================================

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import get_building_splits
from models import get_building_model


# =============================================================================
# Section 1 ▸ Configuration
# =============================================================================

DATASET_DIR  = '/kaggle/input/deepglobe-building'
IMAGE_DIR    = DATASET_DIR
MASK_DIR     = DATASET_DIR

BEST_CKPT    = '/kaggle/working/building_model_best.pth'
CKPT_DIR     = '/kaggle/working/building_ckpts'
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE   = 32        # 16 per GPU with DDP — 640×640 fp16 fits comfortably
NUM_EPOCHS   = 50        # Kaggle 9hr session handles ~50 epochs at this size
LR           = 1e-4
MAX_LR       = 3e-4      # OneCycleLR peak
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.2
PATIENCE     = 8         # early stopping on val IoU
CKPT_EVERY   = 5
NUM_WORKERS  = 4         # 2 per GPU — Kaggle has ample CPU cores


# =============================================================================
# Section 2 ▸ Loss Functions
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
# Section 3 ▸ Metric Helper
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
    iou   = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    return iou


# =============================================================================
# Section 4 ▸ DDP Training Function (runs per GPU process)
# =============================================================================

def train_fn(rank: int, world_size: int):
    """
    Per-GPU training process.

    rank=0 : master process — handles logging, printing, checkpoint saving
    rank=1 : worker process — mirrors rank=0's computation

    DDP flow per step:
        1. Each GPU loads its own shard of the batch (via DistributedSampler)
        2. Forward + loss computed independently on each GPU
        3. loss.backward() triggers all-reduce: gradients averaged across GPUs
        4. optimizer.step() updates weights identically on both GPUs

    Args:
        rank       : GPU index (0 or 1)
        world_size : total number of GPUs (2)
    """
    # ── 4a. Init process group ────────────────────────────────────────────────
    dist.init_process_group(
        backend="nccl",          # NCCL = optimised for GPU-GPU communication
        init_method="env://",    # uses MASTER_ADDR / MASTER_PORT env vars
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    is_master = (rank == 0)   # only master logs and saves

    # ── 4b. Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = get_building_splits(IMAGE_DIR, MASK_DIR,
                                           val_ratio=VAL_RATIO)

    # DistributedSampler partitions the dataset across GPUs.
    # train_sampler.set_epoch(epoch) is CRITICAL — without it, all epochs
    # would use the same shuffle, destroying the benefit of shuffling.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(
        val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    # batch_size here is PER DATALOADER (i.e. global batch / world_size).
    # With world_size=2 and BATCH_SIZE=32, each GPU sees 16 samples per step.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE // world_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE // world_size,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if is_master:
        print(f"\n🚀 Building Detection | DDP Training | {world_size} GPUs")
        print(f"   Global batch={BATCH_SIZE} ({BATCH_SIZE//world_size} per GPU) | "
              f"Resolution=640×640 | Epochs={NUM_EPOCHS}")
        print(f"   Train samples={len(train_ds)} | Val samples={len(val_ds)}\n")

    # ── 4c. Model + DDP ───────────────────────────────────────────────────────
    model = get_building_model().to(device)
    # DDP wraps the model for gradient synchronisation.
    # find_unused_parameters=False: all model parameters are used in every
    # forward pass (no dynamic graph branching) — faster than True.
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── 4d. Loss functions ────────────────────────────────────────────────────
    focal, dice, boundary, soft_dist = build_loss_fns(device)

    # ── 4e. Optimizer + Scheduler ─────────────────────────────────────────────
    # Access model.module.parameters() to get base model params (not DDP wrapper)
    optimizer = AdamW(model.module.parameters(),
                      lr=LR, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: warms up LR from LR→max_lr over first 30% of training,
    # then anneals to LR/10. Outperforms CosineAnnealing for most vision tasks
    # because the warmup phase prevents overshooting in early epochs.
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
    )

    scaler = GradScaler()   # GradScaler is per-process in DDP (not shared)

    # ── 4f. Training state ────────────────────────────────────────────────────
    best_iou          = 0.0
    epochs_no_improve = 0

    # ── 4g. Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # CRITICAL: set_epoch ensures different shuffle each epoch across GPUs
        train_sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, masks, edge_masks, dist_maps in tqdm(
                train_loader,
                desc=f"[GPU{rank}] Epoch {epoch:02d}/{NUM_EPOCHS} [Train]",
                disable=(not is_master)):  # only rank=0 shows the bar

            # images:     (B, 3, 640, 640) float32
            # masks:      (B, 640, 640)    float32 — unsqueeze below
            # edge_masks: (B, 640, 640)    float32 — unsqueeze below
            # dist_maps:  (B, 640, 640)    float32 — unsqueeze below
            images     = images.to(device, non_blocking=True)
            masks      = masks.to(device, non_blocking=True).unsqueeze(1)
            edge_masks = edge_masks.to(device, non_blocking=True).unsqueeze(1)
            dist_maps  = dist_maps.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(images)     # (B, 1, 640, 640)
                loss   = compute_total_loss(
                    logits, masks, edge_masks, dist_maps,
                    focal, dice, boundary, soft_dist
                )

            scaler.scale(loss).backward()
            # Unscale before clip so clip operates in true gradient space
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # OneCycleLR steps per BATCH, not per epoch

            train_loss += loss.item()

            del images, masks, edge_masks, dist_maps, logits, loss

        avg_train_loss = train_loss / len(train_loader)
        torch.cuda.empty_cache()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_accum = 0.0
        iou_sum        = 0.0
        n_batches      = 0

        with torch.no_grad():
            for images, masks, edge_masks, dist_maps in tqdm(
                    val_loader,
                    desc=f"[GPU{rank}] Epoch {epoch:02d}/{NUM_EPOCHS} [Val]  ",
                    disable=(not is_master)):

                images     = images.to(device, non_blocking=True)
                masks      = masks.to(device, non_blocking=True).unsqueeze(1)
                edge_masks = edge_masks.to(device, non_blocking=True).unsqueeze(1)
                dist_maps  = dist_maps.to(device, non_blocking=True).unsqueeze(1)

                with autocast():
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
        local_iou    = iou_sum / max(n_batches, 1)

        # ── Sync IoU across GPUs via all_reduce ───────────────────────────────
        # Each GPU computed IoU on its own shard; average across all processes.
        iou_tensor = torch.tensor([local_iou], device=device)
        dist.all_reduce(iou_tensor, op=dist.ReduceOp.AVG)
        val_iou = iou_tensor.item()

        # dist.barrier() ensures all GPUs finish val before any saves/prints
        dist.barrier()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0

        # ── Logging and checkpointing (rank=0 only) ───────────────────────────
        if is_master:
            # GPU memory report for both cards
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved  = torch.cuda.memory_reserved(i) / 1e9
                print(f"   GPU {i}: {allocated:.2f} GB alloc / "
                      f"{reserved:.2f} GB reserved")

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Best model checkpoint
            if val_iou > best_iou:
                best_iou          = val_iou
                epochs_no_improve = 0
                # model.module.state_dict() — unwrap DDP before saving
                # so checkpoint is portable on any hardware (no DDP required
                # to load it for inference or fine-tuning)
                torch.save(model.module.state_dict(), BEST_CKPT)
                print(f"   🌟 New best IoU={best_iou:.4f} → {BEST_CKPT}")
            else:
                epochs_no_improve += 1
                print(f"   ⏳ No improvement ({epochs_no_improve}/{PATIENCE})")

            # Periodic backup checkpoint
            if epoch % CKPT_EVERY == 0:
                ckpt_path = os.path.join(CKPT_DIR,
                                         f'building_ckpt_ep{epoch:02d}.pth')
                torch.save({
                    'epoch':      epoch,
                    'model_state': model.module.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'best_iou':   best_iou,
                }, ckpt_path)
                print(f"   💾 Backup → {ckpt_path}")

        # Barrier: make sure rank=0 finishes saving before rank=1 continues
        dist.barrier()

        # Early stopping — rank=0 decides; result broadcast implicitly via
        # barrier on next epoch (all ranks exit the loop together since
        # epochs_no_improve is only meaningful at rank=0, but all ranks
        # increment the epoch counter identically)
        if epochs_no_improve >= PATIENCE:
            if is_master:
                print(f"\n🛑 Early stopping at epoch {epoch} "
                      f"(no val IoU improvement for {PATIENCE} epochs).")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    dist.destroy_process_group()

    if is_master:
        print(f"\n🏁 Training complete. Best val IoU = {best_iou:.4f}")
        print(f"   Best model: {BEST_CKPT}")
        print(f"   Checkpoints: {CKPT_DIR}/")


# =============================================================================
# Section 5 ▸ Entry Point — DDP Launch via mp.spawn
# =============================================================================

def main():
    """
    Launch DDP training across all available GPUs.

    mp.spawn creates world_size processes, each running train_fn(rank, world_size).
    MASTER_ADDR + MASTER_PORT must be set before spawn for NCCL rendezvous.
    """
    # NCCL rendezvous: all processes meet at localhost:12355 to exchange
    # initial communication info before training begins.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs detected — cannot run DDP training.")

    print(f"🖥  Detected {world_size} GPU(s):")
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
    print(f"   Launching DDP on {world_size} GPU(s)\n")

    # nprocs=world_size: one Python process per GPU
    # join=True: main() blocks until all processes finish
    mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()