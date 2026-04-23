# =============================================================================
# dataset.py  –  DeepGlobe Road / LandCover / Building Datasets
# =============================================================================
# KEY CHANGES vs v1:
#   • Class name kept as DeepGlobeRoadDataset (was mismatched as 'RoadDataset'
#     in train_road.py – that file is fixed on its side).
#   • get_road_splits()  – returns (train_dataset, val_dataset) with 80/20 split
#     using integer index slicing (reproducible without random seed drama).
#   • train_transform upgraded with:
#       RandomRotate90, RandomBrightnessContrast, GaussNoise,
#       ShiftScaleRotate, GridDistortion, CoarseDropout (Cutout).
#   • CoarseDropout is intentional: it simulates missing image patches,
#     making the model robust to occluded/missing regions — a direct
#     preparation for the MAP INPAINTING downstream task.
#   • val_transform stays minimal (Resize + Normalize + ToTensorV2).
#
# Stage-3 Land Cover additions (v3):
#   • DeepGlobeLandCoverDataset upgraded:
#       - indices param for deterministic 80/20 split support
#       - _rgb_to_mask() uses nearest-colour fallback for JPEG artifacts
#       - get_class_weights() computes inverse-frequency tensor
#   • get_landcover_splits() helper
#   • landcover_train_transform / landcover_val_transform with INTER_NEAREST
# =============================================================================

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Road Extraction Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DeepGlobeRoadDataset(Dataset):
    """
    Binary segmentation dataset for the DeepGlobe Road Extraction track.

    Directory layout expected:
        image_dir/  xxxxx_sat.jpg   (RGB satellite image)
        mask_dir/   xxxxx_mask.png  (Grayscale roads mask, white=road)
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 indices=None):
        """
        Args:
            image_dir  : folder containing *_sat.jpg images
            mask_dir   : folder containing *_mask.png masks
            transform  : albumentations Compose pipeline
            indices    : optional list/array of integer indices to use
                         (enables sliced train/val subsets without copying data)
        """
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        all_images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        )
        # Apply index subset if provided (used for train/val split)
        self.images = [all_images[i] for i in indices] if indices is not None \
                      else all_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        # Load satellite image (BGR → RGB)
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load binary road mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # Safe fallback for inference without masks
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Roads = white (≥128) → 1.0,  Background → 0.0
            mask = np.where(mask >= 128, 1.0, 0.0).astype(np.float32)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        return image, mask


def get_road_splits(image_dir: str, mask_dir: str, val_ratio: float = 0.2):
    """
    Returns (train_dataset, val_dataset) with a deterministic 80/20 split.

    Uses index slicing so both subsets share the same underlying file list
    and transforms differ (train gets augmentations, val stays clean).

    Args:
        image_dir  : path to folder with *_sat.jpg images
        mask_dir   : path to folder with *_mask.png masks
        val_ratio  : fraction of data to reserve for validation (default 0.2)

    Returns:
        train_ds, val_ds  –  DeepGlobeRoadDataset instances
    """
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_total  = len(all_images)
    n_val    = int(n_total * val_ratio)
    n_train  = n_total - n_val

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    train_ds = DeepGlobeRoadDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        transform=train_transform, indices=train_indices
    )
    val_ds = DeepGlobeRoadDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        transform=val_transform, indices=val_indices
    )
    print(f"📂 Dataset split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Land Cover Dataset (multi-class, 7 categories)
# ─────────────────────────────────────────────────────────────────────────────

class DeepGlobeLandCoverDataset(Dataset):
    """
    Multi-class segmentation dataset for the DeepGlobe Land Cover track.
    Masks are RGB images; colours are mapped to integer class IDs (0-6).

    Robustness features:
      • _rgb_to_mask() uses nearest-colour fallback for JPEG artifacts.
      • get_class_weights() returns inverse-frequency weights tensor.
      • indices param enables deterministic 80/20 splits.
    """

    COLOR_MAP = {
        (0, 255, 255):   0,   # Urban land
        (255, 255, 0):   1,   # Agriculture
        (255, 0, 255):   2,   # Rangeland
        (0, 255, 0):     3,   # Forest
        (0, 0, 255):     4,   # Water
        (255, 255, 255): 5,   # Barren land
        (0, 0, 0):       6,   # Unknown
    }
    NUM_CLASSES = 7
    CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest',
                   'Water', 'Barren', 'Unknown']

    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 indices=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        all_images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        )
        self.images = [all_images[i] for i in indices] \
                      if indices is not None else all_images
        # Pre-build colour lookup for nearest-colour fallback
        self._palette  = np.array(list(self.COLOR_MAP.keys()),
                                  dtype=np.float32)   # (7, 3)
        self._class_ids = np.array(list(self.COLOR_MAP.values()),
                                   dtype=np.int64)    # (7,)

    def _rgb_to_mask(self, mask_rgb: np.ndarray) -> np.ndarray:
        """RGB colour mask → integer class-ID map (H, W) int64.

        Step 1: exact colour match (fast vectorised path).
        Step 2: nearest-colour by squared L2 distance for JPEG artifacts
                that shift boundary pixels off the exact palette value.
        """
        H, W, _ = mask_rgb.shape
        id_mask = np.full((H, W), fill_value=-1, dtype=np.int64)

        # Fast exact-match pass
        for rgb, idx in self.COLOR_MAP.items():
            match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            id_mask[match] = idx

        # Nearest-colour fallback for unmatched pixels
        unmatched = id_mask == -1
        if unmatched.any():
            # (N, 3) float32
            pixels = mask_rgb[unmatched].astype(np.float32)
            # Squared L2 to each palette colour: (N, 7)
            diff   = pixels[:, None, :] - self._palette[None, :, :]
            sq_dist = (diff ** 2).sum(axis=-1)
            id_mask[unmatched] = self._class_ids[sq_dist.argmin(axis=-1)]

        return id_mask.astype(np.int64)

    def get_class_weights(self) -> torch.Tensor:
        """Scan all masks and return inverse-frequency weights (7,) float32.

        weight_c = total_pixels / (NUM_CLASSES * count_c)
        Ready for nn.CrossEntropyLoss(weight=...).
        """
        print(f"⏳ Computing class weights from {len(self.images)} masks...")
        counts = np.zeros(self.NUM_CLASSES, dtype=np.float64)
        for img_name in self.images:
            mask_path = os.path.join(
                self.mask_dir, img_name.replace('_sat.jpg', '_mask.png'))
            if not os.path.exists(mask_path):
                continue
            mask_rgb = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            id_mask  = self._rgb_to_mask(mask_rgb)
            for c in range(self.NUM_CLASSES):
                counts[c] += (id_mask == c).sum()
        total = counts.sum()
        weights = np.where(
            counts > 0,
            total / (self.NUM_CLASSES * counts),
            0.0
        ).astype(np.float32)
        print("📊 Class distribution:")
        for name, cnt, w in zip(self.CLASS_NAMES, counts, weights):
            pct = 100.0 * cnt / total if total > 0 else 0.0
            print(f"   {name:<12s}: {cnt:>12,.0f} px ({pct:5.2f}%)  "
                  f"weight={w:.4f}")
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        # image: (H, W, 3) uint8 RGB
        image    = cv2.cvtColor(
            cv2.imread(os.path.join(self.image_dir, img_name)),
            cv2.COLOR_BGR2RGB
        )
        # mask_rgb: (H, W, 3) uint8 RGB colour-coded
        mask_rgb = cv2.cvtColor(
            cv2.imread(os.path.join(self.mask_dir, mask_name)),
            cv2.COLOR_BGR2RGB
        )
        # mask: (H, W) int64 class IDs 0-6
        mask = self._rgb_to_mask(mask_rgb)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']   # (3, 512, 512) float32 tensor
            mask  = aug['mask']    # (512, 512) int64 tensor
        return image, mask


def get_landcover_splits(image_dir: str, mask_dir: str,
                         val_ratio: float = 0.2):
    """Returns (train_ds, val_ds) with deterministic 80/20 split."""
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_total = len(all_images)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds = DeepGlobeLandCoverDataset(
        image_dir, mask_dir,
        transform=landcover_train_transform,
        indices=list(range(n_train))
    )
    val_ds = DeepGlobeLandCoverDataset(
        image_dir, mask_dir,
        transform=landcover_val_transform,
        indices=list(range(n_train, n_total))
    )
    print(f"📂 LandCover split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Building Footprint Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DeepGlobeBuildingDataset(Dataset):
    """Binary segmentation dataset for building footprint extraction."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        image = cv2.cvtColor(
            cv2.imread(os.path.join(self.image_dir, img_name)),
            cv2.COLOR_BGR2RGB
        )

        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask >= 128, 1.0, 0.0).astype(np.float32)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Transformation Pipelines
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation constants (standard for ImageNet pre-trained backbones)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# ── Training Transform ──────────────────────────────────────────────────────
# Augmentation tuned for GPU throughput on Kaggle T4 ×2.
#
# Rule: ONLY use transforms that are fast on CPU so workers keep up with GPU.
#   REMOVED:  A.GridDistortion  — bilinear mesh warp, very slow (~15ms/sample)
#   REMOVED:  A.GaussNoise      — per-pixel RNG, slow at 512×512
#   KEPT:     Flips, Rotate90   — single memcpy, <1ms
#   KEPT:     ShiftScaleRotate  — single affine warp, ~3ms
#   KEPT:     BrightnessContrast, HueSaturationValue — fast LUT operations
#   KEPT:     CoarseDropout     — rectangle fill, <1ms
#
# If GPU utilisation drops below ~70%, reduce augmentation further or
# increase num_workers in the DataLoader.

train_transform = A.Compose([
    # ── Resize to network input ──────────────────────────────────────────
    A.Resize(height=512, width=512),

    # ── Spatial / geometric (all fast: single-pass warp or memcpy) ───────
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=30,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.35
    ),

    # ── Photometric (fast LUT / channel ops) ─────────────────────────────
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                         val_shift_limit=10, p=0.2),

    # ── Cutout (fast rectangle fill) ──────────────────────────────────────
    A.CoarseDropout(
        max_holes=6, max_height=32, max_width=32,
        min_holes=1, min_height=8,  min_width=8,
        fill_value=0, p=0.1
    ),

    # ── Normalise & convert ───────────────────────────────────────────────
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

# ── Validation Transform ─────────────────────────────────────────────────────
# CLEAN pipeline — no augmentation, only resize + normalise.
# Ensures evaluation metrics are honest and reproducible.

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])


# ── 4c  Land Cover Training Transform ────────────────────────────────────────
# CRITICAL: LandCover masks are int64 CLASS IDs — bilinear interpolation would
# produce nonsensical fractional values (e.g. 1.7 between Agriculture &
# Rangeland). Every spatial transform MUST use INTER_NEAREST on the mask.
#
# HueSaturationValue is essential here: Forest vs Rangeland differ mainly in
# hue/saturation — adding variance prevents the model memorising one shade.

landcover_train_transform = A.Compose(
    [
        A.Resize(512, 512,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20,
            val_shift_limit=15, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.15),
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.1),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ],
    is_check_shapes=False,
)

# ── 4d  Land Cover Validation Transform ──────────────────────────────────────
landcover_val_transform = A.Compose(
    [
        A.Resize(512, 512,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ],
    is_check_shapes=False,
)