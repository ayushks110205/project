# =============================================================================
# evaluate_building.py  –  Stage 4 Building Detection Evaluation
# =============================================================================
# Features:
#   • TTA (Test Time Augmentation): 8-variant ensemble for +1-2% IoU
#   • Distributed evaluation via DistributedSampler + dist.all_reduce
#   • Side-by-side comparison: standard vs TTA metrics
#   • Full metric suite: IoU, Dice, Precision, Recall, F1
#   • Only rank=0 prints the final report
#
# Usage (from Kaggle notebook):
#   from evaluate_building import run_evaluation
#   run_evaluation(model_path='/kaggle/working/building_model_best.pth')
#
# Or with DDP:
#   python evaluate_building.py --model /kaggle/working/building_model_best.pth
# =============================================================================

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

from dataset import get_building_splits
from models import get_building_model


# =============================================================================
# Section 1 ▸ Test Time Augmentation (TTA)
# =============================================================================

def tta_predict(model: torch.nn.Module,
                images: torch.Tensor,
                device: torch.device) -> torch.Tensor:
    """
    8-variant TTA ensemble prediction.

    Runs the model on 8 geometrically augmented versions of the input batch,
    reverses each augmentation on the prediction, then averages all 8 sigmoid
    probability maps.

    Variants:
        0: original
        1: horizontal flip
        2: vertical flip
        3: rot90  (counter-clockwise)
        4: rot180
        5: rot270
        6: horizontal flip + rot90
        7: vertical flip  + rot90

    Args:
        model  : model in eval() mode
        images : (B, 3, 640, 640) float32 — normalised
        device : torch.device

    Returns:
        averaged_pred : (B, 1, 640, 640) float32 — averaged sigmoid probabilities
    """
    images = images.to(device)
    preds  = []

    with torch.no_grad():
        with autocast():

            # 0: original
            preds.append(torch.sigmoid(model(images)))

            # 1: horizontal flip
            aug  = torch.flip(images, [3])
            pred = torch.sigmoid(model(aug))
            preds.append(torch.flip(pred, [3]))   # flip back

            # 2: vertical flip
            aug  = torch.flip(images, [2])
            pred = torch.sigmoid(model(aug))
            preds.append(torch.flip(pred, [2]))

            # 3: rot90 (k=1, dims 2,3)
            aug  = torch.rot90(images, k=1, dims=[2, 3])
            pred = torch.sigmoid(model(aug))
            preds.append(torch.rot90(pred, k=-1, dims=[2, 3]))   # rotate back

            # 4: rot180
            aug  = torch.rot90(images, k=2, dims=[2, 3])
            pred = torch.sigmoid(model(aug))
            preds.append(torch.rot90(pred, k=-2, dims=[2, 3]))

            # 5: rot270
            aug  = torch.rot90(images, k=3, dims=[2, 3])
            pred = torch.sigmoid(model(aug))
            preds.append(torch.rot90(pred, k=-3, dims=[2, 3]))

            # 6: horizontal flip + rot90
            aug  = torch.flip(images, [3])
            aug  = torch.rot90(aug, k=1, dims=[2, 3])
            pred = torch.sigmoid(model(aug))
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            preds.append(torch.flip(pred, [3]))

            # 7: vertical flip + rot90
            aug  = torch.flip(images, [2])
            aug  = torch.rot90(aug, k=1, dims=[2, 3])
            pred = torch.sigmoid(model(aug))
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            preds.append(torch.flip(pred, [2]))

    # Stack and average: (8, B, 1, H, W) → mean → (B, 1, H, W)
    return torch.stack(preds, dim=0).mean(dim=0)


# =============================================================================
# Section 2 ▸ Metric Computation
# =============================================================================

def compute_metrics(pred_prob: torch.Tensor,
                    masks:     torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """
    Compute binary segmentation metrics for a batch.

    Args:
        pred_prob : (B, 1, H, W) float32 sigmoid probabilities
        masks     : (B, 1, H, W) float32 binary {0, 1}
        threshold : binarisation threshold (default 0.5)

    Returns:
        dict with keys: iou, dice, precision, recall, f1
        all values are Python floats (batch-averaged)
    """
    pred_bin = (pred_prob > threshold).float()
    smooth   = 1e-6

    tp = (pred_bin * masks).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * masks).sum(dim=(1, 2, 3))
    tn = ((1 - pred_bin) * (1 - masks)).sum(dim=(1, 2, 3))

    iou       = ((tp + smooth) / (tp + fp + fn + smooth)).mean().item()
    dice      = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).mean().item()
    precision = ((tp + smooth) / (tp + fp + smooth)).mean().item()
    recall    = ((tp + smooth) / (tp + fn + smooth)).mean().item()
    f1        = ((2 * precision * recall + smooth) /
                 (precision + recall + smooth))

    return {'iou': iou, 'dice': dice,
            'precision': precision, 'recall': recall, 'f1': f1}


def accumulate_metrics(accum: dict, batch_metrics: dict, n: int) -> dict:
    """Running-sum accumulator for epoch-level metrics."""
    for k, v in batch_metrics.items():
        accum[k] = accum.get(k, 0.0) + v * n
    accum['_n'] = accum.get('_n', 0) + n
    return accum


def finalise_metrics(accum: dict) -> dict:
    """Divide accumulated sums by total sample count."""
    n = max(accum.pop('_n', 1), 1)
    return {k: v / n for k, v in accum.items()}


# =============================================================================
# Section 3 ▸ Distributed Evaluation Function
# =============================================================================

def eval_fn(rank: int, world_size: int,
            model_path: str, use_tta: bool,
            image_dir: str, mask_dir: str):
    """
    Per-GPU evaluation process.

    Each GPU evaluates on its shard of the val set (via DistributedSampler).
    Metrics are summed across GPUs using dist.all_reduce before printing.
    Only rank=0 prints the final report.

    Args:
        rank       : GPU index
        world_size : total GPUs
        model_path : path to best checkpoint .pth
        use_tta    : whether to run TTA evaluation in addition to standard
        image_dir  : images directory
        mask_dir   : masks directory
    """
    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    device    = torch.device(f"cuda:{rank}")
    is_master = (rank == 0)

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_building_model().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    if is_master:
        print(f"✅ Model loaded: {model_path}")

    # Wrap in DDP for distributed val
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_ds = get_building_splits(image_dir, mask_dir, val_ratio=0.2)

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False)
    val_loader  = DataLoader(val_ds, batch_size=8, sampler=val_sampler,
                             num_workers=2, pin_memory=True)

    if is_master:
        print(f"📂 Val samples: {len(val_ds)} | "
              f"Batches per GPU: {len(val_loader)}")
        print(f"   TTA: {'ON (8 variants)' if use_tta else 'OFF'}\n")

    # ── Standard evaluation ───────────────────────────────────────────────────
    std_accum = {}
    tta_accum = {} if use_tta else None

    with torch.no_grad():
        for images, masks, edge_masks, dist_maps in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()
            b      = images.size(0)

            # Standard prediction
            with autocast():
                logits    = model(images)
                std_prob  = torch.sigmoid(logits)

            std_metrics = compute_metrics(std_prob, masks)
            std_accum   = accumulate_metrics(std_accum, std_metrics, b)

            # TTA prediction (optional)
            if use_tta:
                tta_prob    = tta_predict(model, images, device)
                tta_metrics = compute_metrics(tta_prob, masks)
                tta_accum   = accumulate_metrics(tta_accum, tta_metrics, b)

            del images, masks, logits

    # ── Sync metrics across GPUs ──────────────────────────────────────────────
    # All-reduce sums the metric tensors across all processes, then we divide
    # by world_size to get the global average.
    def sync_metric_dict(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            t = torch.tensor([v], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            out[k] = t.item()
        return out

    std_final = sync_metric_dict(finalise_metrics(std_accum))
    tta_final = sync_metric_dict(finalise_metrics(tta_accum)) \
                if use_tta else None

    dist.barrier()

    # ── Print report (rank=0 only) ────────────────────────────────────────────
    if is_master:
        print("=" * 55)
        print("    Building Detection — Evaluation Report")
        print("=" * 55)
        header = f"{'Metric':<12} {'Standard':>12}"
        if use_tta:
            header += f" {'TTA (×8)':>12} {'Δ TTA':>10}"
        print(header)
        print("-" * (55 if not use_tta else 70))

        metric_names = ['iou', 'dice', 'precision', 'recall', 'f1']
        for m in metric_names:
            s_val = std_final[m]
            row   = f"{m.capitalize():<12} {s_val:>12.4f}"
            if use_tta:
                t_val = tta_final[m]
                delta = t_val - s_val
                sign  = '+' if delta >= 0 else ''
                row  += f" {t_val:>12.4f} {sign}{delta*100:>8.2f}%"
            print(row)

        print("=" * (55 if not use_tta else 70))
        if use_tta:
            iou_gain = tta_final['iou'] - std_final['iou']
            print(f"\n📈 TTA IoU gain: {iou_gain*100:+.2f}%")
        print(f"   Model: {model_path}")

    dist.destroy_process_group()


# =============================================================================
# Section 4 ▸ Entry Points
# =============================================================================

def run_evaluation(model_path: str = '/kaggle/working/building_model_best.pth',
                   image_dir:  str = '/kaggle/input/deepglobe-building',
                   mask_dir:   str = '/kaggle/input/deepglobe-building',
                   use_tta:    bool = True):
    """
    Call this from a Kaggle notebook cell.

    Example:
        from evaluate_building import run_evaluation
        run_evaluation(use_tta=True)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'   # different port from training

    world_size = torch.cuda.device_count()
    mp.spawn(eval_fn,
             args=(world_size, model_path, use_tta, image_dir, mask_dir),
             nprocs=world_size, join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Building Detection Distributed Evaluation')
    parser.add_argument('--model',
                        default='/kaggle/working/building_model_best.pth')
    parser.add_argument('--images',
                        default='/kaggle/input/deepglobe-building')
    parser.add_argument('--masks',
                        default='/kaggle/input/deepglobe-building')
    parser.add_argument('--no-tta', action='store_true',
                        help='Skip TTA (faster, ~1-2% lower IoU)')
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        image_dir=args.images,
        mask_dir=args.masks,
        use_tta=not args.no_tta,
    )