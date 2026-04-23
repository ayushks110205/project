# =============================================================================
# inpainting_dataset.py  –  Stage 2 Inpainting Dataset
# =============================================================================
# Loads binary road masks (from Stage 1 ground-truth _mask.png files) and
# dynamically generates holes at __getitem__ time using one of three strategies,
# chosen randomly each call.
#
# Returns (corrupted_mask, hole_mask, complete_mask):
#   corrupted_mask : (1, H, W) float32 — road mask with hole regions zeroed
#   hole_mask      : (1, H, W) float32 — 1=valid known region, 0=missing hole
#   complete_mask  : (1, H, W) float32 — original ground-truth road mask
#
# The 2-channel model INPUT is torch.cat([corrupted_mask, hole_mask], dim=0)
# which gives shape (2, H, W). This is done in the training loop, not here,
# to keep the dataset clean and reusable.
#
# Hole generation strategies (chosen randomly with equal probability):
#   1. Random rectangles  — 1-3 rects each covering 10-30% of mask area
#   2. Brush strokes      — irregular convex polygon blobs
#   3. Large single block — one rectangle covering 40-50% of image
# =============================================================================

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Hole Generation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _random_bbox(h: int, w: int, min_frac: float, max_frac: float):
    """
    Return a random bounding box (x1, y1, x2, y2) covering min_frac–max_frac
    of the image area.
    """
    area   = h * w
    target = random.uniform(min_frac, max_frac) * area
    bh     = int(np.sqrt(target * random.uniform(0.5, 2.0)))   # random aspect
    bw     = int(target / max(bh, 1))
    bh     = min(bh, h)
    bw     = min(bw, w)
    x1     = random.randint(0, max(0, w - bw))
    y1     = random.randint(0, max(0, h - bh))
    return x1, y1, x1 + bw, y1 + bh


def generate_rect_holes(h: int, w: int,
                        n_rects: int = None,
                        min_frac: float = 0.10,
                        max_frac: float = 0.30) -> np.ndarray:
    """
    Strategy 1 — Random Rectangles.

    Creates 1-3 rectangular holes.  Each rectangle covers 10-30% of the image.

    Returns:
        hole_mask : (H, W) uint8 — 0 = hole, 1 = valid  (same convention as dataset)
    """
    hole_mask = np.ones((h, w), dtype=np.uint8)   # start fully valid
    n = n_rects if n_rects is not None else random.randint(1, 3)
    for _ in range(n):
        x1, y1, x2, y2 = _random_bbox(h, w, min_frac, max_frac)
        hole_mask[y1:y2, x1:x2] = 0               # punch hole
    return hole_mask


def generate_brush_strokes(h: int, w: int,
                           n_strokes: int = None) -> np.ndarray:
    """
    Strategy 2 — Brush Strokes (Irregular Blobs).

    Draws n_strokes convex-polygon blobs of random size and position.
    More realistic than rectangles — simulates cloud shadows or irregular
    data-missing regions encountered in satellite imagery.

    Returns:
        hole_mask : (H, W) uint8 — 0 = hole, 1 = valid
    """
    hole_mask = np.ones((h, w), dtype=np.uint8)
    n = n_strokes if n_strokes is not None else random.randint(2, 5)
    for _ in range(n):
        # Random-sized blob
        max_r = int(min(h, w) * random.uniform(0.08, 0.22))
        cx    = random.randint(max_r, w - max_r)
        cy    = random.randint(max_r, h - max_r)
        n_pts = random.randint(5, 10)
        # Generate points around centre with random radius variation
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        radii  = np.random.uniform(max_r * 0.5, max_r * 1.0, n_pts)
        pts    = np.stack([
            (cx + radii * np.cos(angles)).astype(np.int32),
            (cy + radii * np.sin(angles)).astype(np.int32),
        ], axis=1)
        pts = np.clip(pts, 0, [w - 1, h - 1])
        cv2.fillPoly(hole_mask, [pts.reshape(-1, 1, 2)], color=0)
    return hole_mask


def generate_large_block(h: int, w: int,
                         min_frac: float = 0.40,
                         max_frac: float = 0.50) -> np.ndarray:
    """
    Strategy 3 — Large Single Block.

    One large rectangle covering 40-50% of the image — the hardest case.
    Forces the model to learn global road topology rather than relying on
    nearby context.

    Returns:
        hole_mask : (H, W) uint8 — 0 = hole, 1 = valid
    """
    hole_mask = np.ones((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = _random_bbox(h, w, min_frac, max_frac)
    hole_mask[y1:y2, x1:x2] = 0
    return hole_mask


def apply_hole(complete_mask: np.ndarray, hole_mask: np.ndarray):
    """
    Apply a hole to a complete binary mask.

    Args:
        complete_mask : (H, W) float32 — ground-truth road mask {0, 1}
        hole_mask     : (H, W) uint8   — 1=valid, 0=hole

    Returns:
        corrupted_mask : (H, W) float32 — mask with hole zeroed out
    """
    return complete_mask * hole_mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ InpaintingDataset
# ─────────────────────────────────────────────────────────────────────────────

class InpaintingDataset(Dataset):
    """
    Stage 2 dataset for road mask inpainting.

    Loads ground-truth binary road masks and dynamically generates holes
    at __getitem__ time using one of three strategies chosen randomly.

    Args:
        mask_dir   : folder containing *_mask.png files
        image_size : spatial resolution (square), default 512
        transform  : optional callable applied to complete_mask (H, W) float32
                     before hole generation.  Should return a numpy array.
        indices    : optional list of integer indices for train/val split

    Returns per item:
        corrupted_mask : (1, H, W) float32 — road mask with holes zeroed
        hole_mask      : (1, H, W) float32 — 1=valid, 0=hole
        complete_mask  : (1, H, W) float32 — original ground-truth mask
    """

    STRATEGIES = ['rects', 'brush', 'block']

    def __init__(self,
                 mask_dir:    str,
                 image_size:  int = 512,
                 transform=None,
                 indices=None):
        self.mask_dir   = mask_dir
        self.image_size = image_size
        self.transform  = transform

        all_masks = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
        )
        self.masks = [all_masks[i] for i in indices] \
                     if indices is not None else all_masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx: int):
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ── Load binary road mask ─────────────────────────────────────────────
        raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   # (H, W) uint8
        if raw is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Resize to target resolution
        raw = cv2.resize(raw, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_NEAREST)

        # Binarise: road ≥ 128 → 1.0, background → 0.0
        complete_mask = np.where(raw >= 128, 1.0, 0.0).astype(np.float32)
        # (H, W) float32

        # ── Optional transform (e.g. Rotate90, Flip) ─────────────────────────
        if self.transform is not None:
            complete_mask = self.transform(complete_mask)
        # (H, W) float32

        h, w = complete_mask.shape

        # ── Randomly choose hole strategy ─────────────────────────────────────
        strategy = random.choice(self.STRATEGIES)

        if strategy == 'rects':
            hole_mask = generate_rect_holes(h, w)
        elif strategy == 'brush':
            hole_mask = generate_brush_strokes(h, w)
        else:   # 'block'
            hole_mask = generate_large_block(h, w)
        # hole_mask: (H, W) uint8 — 1=valid, 0=hole

        # ── Apply hole to complete mask ───────────────────────────────────────
        corrupted_mask = apply_hole(complete_mask, hole_mask)
        # corrupted_mask: (H, W) float32

        # ── Convert to tensors (add channel dim) ─────────────────────────────
        # All masks: (H, W) → (1, H, W) float32
        corrupted_t = torch.from_numpy(corrupted_mask).unsqueeze(0)   # (1, H, W)
        hole_t      = torch.from_numpy(hole_mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        complete_t  = torch.from_numpy(complete_mask).unsqueeze(0)    # (1, H, W)

        return corrupted_t, hole_t, complete_t


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Transform Helpers (mask-safe — NO normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class MaskTrainTransform:
    """
    Spatial augmentations for binary masks.

    IMPORTANT: do NOT use brightness/contrast/noise transforms on masks.
    Masks are binary {0, 1} — photometric augmentations will break them.
    Only spatial transforms are safe here.
    """

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            mask : (H, W) float32 binary mask
        Returns:
            mask : (H, W) float32 binary mask (possibly flipped/rotated)
        """
        # Horizontal flip
        if random.random() < 0.5:
            mask = np.fliplr(mask).copy()

        # Vertical flip
        if random.random() < 0.5:
            mask = np.flipud(mask).copy()

        # RandomRotate90
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])   # 90°, 180°, 270°
            mask = np.rot90(mask, k=k).copy()

        return mask


class MaskValTransform:
    """
    Validation transform — identity (no augmentation).

    Resize is already handled inside InpaintingDataset.__init__
    via cv2.resize, so val transform is a no-op.
    """

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return mask   # pass-through


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Split Helper (mirrors Stage 1's get_road_splits interface)
# ─────────────────────────────────────────────────────────────────────────────

def get_inpainting_splits(mask_dir:   str,
                          image_size: int   = 512,
                          val_ratio:  float = 0.20):
    """
    Returns (train_dataset, val_dataset) with deterministic 80/20 split.

    Uses the same index-slicing strategy as Stage 1's get_road_splits()
    so both stages operate on identical train/val splits when the mask_dir
    is the same folder.

    Args:
        mask_dir   : folder with *_mask.png files
        image_size : resize target (default 512)
        val_ratio  : fraction reserved for validation

    Returns:
        train_ds, val_ds  –  InpaintingDataset instances
    """
    all_masks = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    )
    n_total  = len(all_masks)
    n_val    = int(n_total * val_ratio)
    n_train  = n_total - n_val

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    train_ds = InpaintingDataset(
        mask_dir=mask_dir, image_size=image_size,
        transform=MaskTrainTransform(), indices=train_indices
    )
    val_ds = InpaintingDataset(
        mask_dir=mask_dir, image_size=image_size,
        transform=MaskValTransform(), indices=val_indices
    )
    print(f"📂 Inpainting split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    mask_dir = 'datasets/train'
    if not os.path.isdir(mask_dir):
        print(f"Skipping smoke test — '{mask_dir}' not found (run on Colab).")
        sys.exit(0)

    train_ds, val_ds = get_inpainting_splits(mask_dir)
    corrupted, hole, complete = train_ds[0]

    print(f"corrupted_mask : {corrupted.shape}  min={corrupted.min():.2f}  max={corrupted.max():.2f}")
    print(f"hole_mask      : {hole.shape}  min={hole.min():.2f}  max={hole.max():.2f}")
    print(f"complete_mask  : {complete.shape}  min={complete.min():.2f}  max={complete.max():.2f}")

    # Verify 2-channel model input construction
    model_input = torch.cat([corrupted, hole], dim=0)   # (2, H, W)
    print(f"Model input    : {model_input.shape}")       # expect (2, 512, 512)
    print("✅ InpaintingDataset smoke test passed")
