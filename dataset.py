# =============================================================================
# dataset.py  –  DeepGlobe Road / LandCover / Building Datasets
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  AVAILABLE DATASET CLASSES                                              │
# │  ─────────────────────────────────────────────────────────────────────  │
# │  DeepGlobeRoadDataset     – binary road segmentation                   │
# │    └── get_road_splits()                                                │
# │  DeepGlobeLandCoverDataset– 7-class land cover segmentation            │
# │    └── get_landcover_splits()                                           │
# │  DeepGlobeBuildingDataset – binary building footprint extraction       │
# │    └── get_building_splits()                                            │
# │                                                                         │
# │  TRANSFORMS                                                             │
# │  train_transform / val_transform           – Road (512×512)            │
# │  landcover_train_transform / val_transform – LandCover (512×512)       │
# │  building_train_transform / val_transform  – Building (640×640)        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Stage-4 Building Detection additions (v4):
#   • DeepGlobeBuildingDataset upgraded:
#       - 640×640 resolution (exploits 30GB VRAM on T4×2)
#       - Returns (image, mask, edge_mask, dist_map) 4-tuple
#       - edge_mask: Canny edges from binary mask (binary float32)
#       - dist_map:  cv2.distanceTransform normalised 0-1
#   • building_train_transform / building_val_transform at 640×640
#   • get_building_splits() helper
#
# Previous additions kept intact — road and landcover unchanged.
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
                 indices=None, _prebuilt_list=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        # Accept a pre-built sorted list to avoid redundant os.listdir() calls
        # when get_road_splits() has already scanned the directory.
        if _prebuilt_list is not None:
            source = _prebuilt_list
        else:
            source = sorted(
                [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
            )
        self.images = [source[i] for i in indices] if indices is not None \
                      else list(source)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


def get_road_splits(image_dir: str, mask_dir: str, val_ratio: float = 0.2):
    """Returns (train_dataset, val_dataset) with a deterministic 80/20 split.

    Scans the directory only ONCE and passes the pre-built list into each
    Dataset so __init__ never calls os.listdir() again.  The intermediate
    list is explicitly deleted after the Datasets have sliced their indices,
    freeing ~3–5 MB of string objects immediately.
    """
    import gc
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_total  = len(all_images)
    n_val    = int(n_total * val_ratio)
    n_train  = n_total - n_val

    train_ds = DeepGlobeRoadDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        transform=train_transform,
        indices=list(range(n_train)),
        _prebuilt_list=all_images,          # ← skip redundant os.listdir()
    )
    val_ds = DeepGlobeRoadDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        transform=val_transform,
        indices=list(range(n_train, n_total)),
        _prebuilt_list=all_images,          # ← same pre-built list reused
    )

    del all_images   # ← free intermediate list now that Datasets hold their slices
    gc.collect()

    print(f"📂 Road split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Land Cover Dataset (multi-class, 7 categories)
# ─────────────────────────────────────────────────────────────────────────────

class DeepGlobeLandCoverDataset(Dataset):
    """
    Multi-class segmentation dataset for the DeepGlobe Land Cover track.
    Masks are RGB images; colours are mapped to integer class IDs (0-6).
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
                 indices=None, _prebuilt_list=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        if _prebuilt_list is not None:
            source = _prebuilt_list
        else:
            source = sorted(
                [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
            )
        self.images = [source[i] for i in indices] \
                      if indices is not None else list(source)
        self._palette   = np.array(list(self.COLOR_MAP.keys()),  dtype=np.float32)
        self._class_ids = np.array(list(self.COLOR_MAP.values()), dtype=np.int64)

    def _rgb_to_mask(self, mask_rgb: np.ndarray) -> np.ndarray:
        """RGB colour mask → integer class-ID map (H, W) int64."""
        H, W, _ = mask_rgb.shape
        id_mask = np.full((H, W), fill_value=-1, dtype=np.int64)
        for rgb, idx in self.COLOR_MAP.items():
            match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            id_mask[match] = idx
        unmatched = id_mask == -1
        if unmatched.any():
            pixels  = mask_rgb[unmatched].astype(np.float32)
            diff    = pixels[:, None, :] - self._palette[None, :, :]
            sq_dist = (diff ** 2).sum(axis=-1)
            id_mask[unmatched] = self._class_ids[sq_dist.argmin(axis=-1)]
        return id_mask.astype(np.int64)

    def get_class_weights(self) -> torch.Tensor:
        """Scan all masks and return inverse-frequency weights (7,) float32."""
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
        # Cap weights at 10.0 — very rare classes (e.g. Unknown at 0.06%)
        # produce weights >200 which cause catastrophic loss spikes.
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        print("📊 Class distribution:")
        for name, cnt, w in zip(self.CLASS_NAMES, counts, weights):
            pct = 100.0 * cnt / total if total > 0 else 0.0
            print(f"   {name:<12s}: {cnt:>12,.0f} px ({pct:5.2f}%)  weight={w:.4f}")
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        image    = cv2.cvtColor(
            cv2.imread(os.path.join(self.image_dir, img_name)),
            cv2.COLOR_BGR2RGB
        )
        mask_rgb = cv2.cvtColor(
            cv2.imread(os.path.join(self.mask_dir, mask_name)),
            cv2.COLOR_BGR2RGB
        )
        mask = self._rgb_to_mask(mask_rgb)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']
        return image, mask


def get_landcover_splits(image_dir: str, mask_dir: str, val_ratio: float = 0.2):
    """Returns (train_ds, val_ds) with deterministic 80/20 split."""
    import gc
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_total = len(all_images)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds = DeepGlobeLandCoverDataset(
        image_dir, mask_dir,
        transform=landcover_train_transform,
        indices=list(range(n_train)),
        _prebuilt_list=all_images,
    )
    val_ds = DeepGlobeLandCoverDataset(
        image_dir, mask_dir,
        transform=landcover_val_transform,
        indices=list(range(n_train, n_total)),
        _prebuilt_list=all_images,
    )
    del all_images
    gc.collect()
    print(f"📂 LandCover split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Building Footprint Dataset (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
# Two dataset classes share the same 4-tuple interface:
#   (image, mask, edge_mask, dist_map)
#
#   DeepGlobeBuildingDataset  – reads *_sat.jpg / *_mask.png in same folder
#   MassachusettsBuildingDataset – reads *.tiff images + *.tiff masks from
#                                   separate sibling folders (train / train_labels)
# ─────────────────────────────────────────────────────────────────────────────


class MassachusettsBuildingDataset(Dataset):
    """
    Binary segmentation dataset for the Massachusetts Buildings Dataset.

    Directory layout:
        image_dir/   <tile_id>.tiff   (RGB satellite tile)
        mask_dir/    <tile_id>.tiff   (Grayscale building mask, 255=building)

    The two folders are siblings, e.g.:
        .../tiff/train/          ← image_dir
        .../tiff/train_labels/   ← mask_dir
    Filenames match 1-to-1 between the two folders.

    Returns per __getitem__:
        image     : (3, 640, 640) float32 – normalised satellite image
        mask      : (640, 640)    float32 – binary building mask {0, 1}
        edge_mask : (640, 640)    float32 – Canny edges of building boundary
        dist_map  : (640, 640)    float32 – distance transform (0-1 normalised)
    """

    def __init__(self, image_dir: str, mask_dir: str,
                 transform=None, indices=None, _prebuilt_list=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        if _prebuilt_list is not None:
            source = _prebuilt_list
        else:
            # Accept any image extension — Massachusetts uses .tiff
            exts = ('.tiff', '.tif', '.png', '.jpg')
            source = sorted(
                [f for f in os.listdir(image_dir)
                 if os.path.splitext(f)[1].lower() in exts]
            )
        self.images = [source[i] for i in indices] \
                      if indices is not None else list(source)

        # Build stem→path lookup for masks (handles .tif vs .tiff mismatches)
        _exts = ('.tiff', '.tif', '.png', '.jpg')
        self._mask_lookup = {}
        if os.path.isdir(mask_dir):
            for mf in os.listdir(mask_dir):
                if os.path.splitext(mf)[1].lower() in _exts:
                    stem = os.path.splitext(mf)[0]
                    self._mask_lookup[stem] = os.path.join(mask_dir, mf)

    def __len__(self):
        return len(self.images)

    # Reuse the same static helpers as DeepGlobeBuildingDataset
    @staticmethod
    def _make_edge_mask(binary_mask: np.ndarray) -> np.ndarray:
        edges  = cv2.Canny(binary_mask, threshold1=100, threshold2=200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=1)
        return (edges > 0).astype(np.float32)

    @staticmethod
    def _make_dist_map(binary_mask: np.ndarray) -> np.ndarray:
        dist    = cv2.distanceTransform(binary_mask,
                                        cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        max_val = dist.max()
        if max_val > 0:
            dist = dist / max_val
        return dist.astype(np.float32)

    def __getitem__(self, idx):
        from PIL import Image
        img_name = self.images[idx]

        # ── Load satellite image via PIL (no libtiff GeoTIFF warnings) ───────
        img_path = os.path.join(self.image_dir, img_name)
        pil_img  = Image.open(img_path).convert('RGB')
        image    = np.array(pil_img, dtype=np.uint8)   # (H, W, 3) uint8 RGB

        # ── Load binary mask via PIL (extension-agnostic lookup) ─────────────
        img_stem  = os.path.splitext(img_name)[0]
        mask_path = self._mask_lookup.get(img_stem)
        if mask_path is None:
            h, w     = image.shape[:2]
            mask_raw = np.zeros((h, w), dtype=np.uint8)
        else:
            pil_mask = Image.open(mask_path).convert('L')   # grayscale
            mask_raw = np.array(pil_mask, dtype=np.uint8)
            mask_raw = np.where(mask_raw >= 128, np.uint8(255), np.uint8(0))

        # ── Auxiliary maps ────────────────────────────────────────────────────
        edge_mask  = self._make_edge_mask(mask_raw)   # (H, W) float32
        dist_map   = self._make_dist_map(mask_raw)    # (H, W) float32
        mask_float = (mask_raw > 0).astype(np.float32)

        # ── Apply transform ───────────────────────────────────────────────────
        if self.transform:
            aug       = self.transform(
                image=image, mask=mask_float,
                edge_mask=edge_mask, dist_map=dist_map,
            )
            image     = aug['image']
            mask      = aug['mask']
            edge_mask = aug['edge_mask']
            dist_map  = aug['dist_map']
        else:
            image     = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask      = torch.from_numpy(mask_float)
            edge_mask = torch.from_numpy(edge_mask)
            dist_map  = torch.from_numpy(dist_map)

        return image, mask, edge_mask, dist_map


def get_massachusetts_building_splits(image_dir: str, mask_dir: str,
                                      val_ratio: float = 0.2):
    """
    Returns (train_ds, val_ds) with a deterministic 80/20 split.

    Args:
        image_dir : folder with *.tiff satellite images  (e.g. .../tiff/train)
        mask_dir  : folder with *.tiff masks             (e.g. .../tiff/train_labels)
        val_ratio : fraction reserved for validation
    """
    import gc
    exts = ('.tiff', '.tif', '.png', '.jpg')
    all_images = sorted(
        [f for f in os.listdir(image_dir)
         if os.path.splitext(f)[1].lower() in exts]
    )
    n_total = len(all_images)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds = MassachusettsBuildingDataset(
        image_dir, mask_dir,
        transform=building_train_transform,
        indices=list(range(n_train)),
        _prebuilt_list=all_images,
    )
    val_ds = MassachusettsBuildingDataset(
        image_dir, mask_dir,
        transform=building_val_transform,
        indices=list(range(n_train, n_total)),
        _prebuilt_list=all_images,
    )
    del all_images
    gc.collect()
    print(f"📂 Massachusetts Building split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


class DeepGlobeBuildingDataset(Dataset):
    """
    Binary segmentation dataset for the DeepGlobe Building Detection track.

    Upgraded for dual T4 (640×640, 4-output tuple):

    Returns per __getitem__:
        image     : (3, 640, 640) float32 – normalised satellite image
        mask      : (640, 640)    float32 – binary building mask {0, 1}
        edge_mask : (640, 640)    float32 – Canny edges of building boundary
        dist_map  : (640, 640)    float32 – distance transform (0-1 normalised)
                                            how far each building pixel is from
                                            its boundary (0=boundary, 1=centre)

    edge_mask is used by BoundaryLoss to sharpen building outlines.
    dist_map  is used by SoftDistanceLoss to penalise interior holes.

    Args:
        image_dir : folder with *_sat.jpg images
        mask_dir  : folder with *_mask.png masks (255=building, 0=background)
        transform : albumentations Compose with additional_targets for
                    edge_mask and dist_map
        indices   : optional int list for deterministic train/val splits
    """

    def __init__(self, image_dir: str, mask_dir: str,
                 transform=None, indices=None, _prebuilt_list=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        if _prebuilt_list is not None:
            source = _prebuilt_list
        else:
            source = sorted(
                [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
            )
        self.images = [source[i] for i in indices] \
                      if indices is not None else list(source)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _make_edge_mask(binary_mask: np.ndarray) -> np.ndarray:
        """
        Generate edge mask from binary building mask using Canny.

        Args:
            binary_mask : (H, W) uint8 {0, 255}
        Returns:
            edge_mask   : (H, W) float32 {0.0, 1.0}
        """
        # Canny on uint8 [0,255] image — low/high thresholds tuned for
        # binary masks where edges are perfectly sharp pixel boundaries.
        edges = cv2.Canny(binary_mask, threshold1=100, threshold2=200)
        # Dilate slightly so loss has a non-zero gradient near edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=1)
        return (edges > 0).astype(np.float32)

    @staticmethod
    def _make_dist_map(binary_mask: np.ndarray) -> np.ndarray:
        """
        Generate normalised distance transform map from binary building mask.

        Each building pixel's value = distance to the nearest background pixel
        (i.e. building boundary), normalised to [0, 1].
          0.0 = boundary pixel
          1.0 = centre of the largest building in the image

        Args:
            binary_mask : (H, W) uint8 {0, 255}
        Returns:
            dist_map    : (H, W) float32 in [0, 1]
        """
        # distanceTransform expects: foreground=255, background=0
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Normalise to [0, 1]
        max_val = dist.max()
        if max_val > 0:
            dist = dist / max_val
        return dist.astype(np.float32)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        # ── Load satellite image (BGR → RGB) ─────────────────────────────────
        # image: (H, W, 3) uint8
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        if image is None:
            raise FileNotFoundError(
                f"Image not found: {os.path.join(self.image_dir, img_name)}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Load binary mask ──────────────────────────────────────────────────
        # mask_raw: (H, W) uint8 {0, 255}
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            h, w = image.shape[:2]
            mask_raw = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_raw = np.where(mask_raw >= 128,
                                np.uint8(255), np.uint8(0))

        # ── Generate auxiliary maps ───────────────────────────────────────────
        # Both built from the full-resolution uint8 mask BEFORE resizing
        # so Canny and distanceTransform see the original sharp boundaries.
        edge_mask = self._make_edge_mask(mask_raw)   # (H, W) float32 {0,1}
        dist_map  = self._make_dist_map(mask_raw)    # (H, W) float32 [0,1]

        # Binary float32 mask for augmentation pipeline
        mask_float = (mask_raw > 0).astype(np.float32)  # (H, W) float32 {0,1}

        # ── Apply albumentations transform ────────────────────────────────────
        # additional_targets maps 'edge_mask' and 'dist_map' so they receive
        # IDENTICAL spatial transforms as image and mask.
        if self.transform:
            aug = self.transform(
                image=image,
                mask=mask_float,
                edge_mask=edge_mask,
                dist_map=dist_map,
            )
            image     = aug['image']       # (3, 640, 640) float32 tensor
            mask      = aug['mask']        # (640, 640) float32 tensor
            edge_mask = aug['edge_mask']   # (640, 640) float32 tensor
            dist_map  = aug['dist_map']    # (640, 640) float32 tensor
        else:
            # No transform: convert image manually
            image     = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask      = torch.from_numpy(mask_float)
            edge_mask = torch.from_numpy(edge_mask)
            dist_map  = torch.from_numpy(dist_map)

        # shapes:
        #   image:     (3, 640, 640) float32
        #   mask:      (640, 640)    float32
        #   edge_mask: (640, 640)    float32
        #   dist_map:  (640, 640)    float32
        return image, mask, edge_mask, dist_map


def get_building_splits(image_dir: str, mask_dir: str,
                        val_ratio: float = 0.2):
    """
    Returns (train_ds, val_ds) with deterministic 80/20 split.

    Args:
        image_dir : folder with *_sat.jpg images
        mask_dir  : folder with *_mask.png masks
        val_ratio : fraction reserved for validation (default 0.2)

    Returns:
        train_ds, val_ds – DeepGlobeBuildingDataset instances
    """
    import gc
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    )
    n_total = len(all_images)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds = DeepGlobeBuildingDataset(
        image_dir, mask_dir,
        transform=building_train_transform,
        indices=list(range(n_train)),
        _prebuilt_list=all_images,
    )
    val_ds = DeepGlobeBuildingDataset(
        image_dir, mask_dir,
        transform=building_val_transform,
        indices=list(range(n_train, n_total)),
        _prebuilt_list=all_images,
    )
    del all_images
    gc.collect()
    print(f"📂 Building split → Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Transformation Pipelines
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# ── 4a  Road Training Transform (512×512, fast pipeline) ─────────────────────
# GridDistortion/GaussNoise removed — too slow for main-process loading.

train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale=(0.85, 1.15),
        rotate=(-30, 30),
        border_mode=cv2.BORDER_REFLECT_101, p=0.35
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                         val_shift_limit=10, p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(8, 32),
        hole_width_range=(8, 32),
        fill=0, p=0.1
    ),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

# ── 4b  Road Validation Transform ────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

# ── 4c  Land Cover Training Transform (512×512, INTER_NEAREST on mask) ───────
# CRITICAL: class-ID masks must use INTER_NEAREST to avoid interpolation
# creating invalid fractional class IDs between palette values.

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
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0, p=0.1),
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

# ── 4e  Building Training Transform (640×640) ─────────────────────────────────
# 640×640 is viable with 30GB VRAM (batch=32, 16 per GPU).
# Richer augmentation is appropriate here — buildings have complex rooftop
# textures (tile, metal, concrete, glass) and cast shadows that vary by
# time-of-day. CLAHE + Sharpen help with shadow recovery and edge clarity.
#
# additional_targets ensures edge_mask and dist_map receive identical
# spatial transforms as the image and primary mask.

building_train_transform = A.Compose(
    [
        A.Resize(640, 640),
        # ── Spatial ────────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.15),
        # ── Appearance ─────────────────────────────────────────────────────
        A.RandomShadow(p=0.3),               # simulate building shadows
        A.ElasticTransform(p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.2, p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 64),
            hole_width_range=(8, 64),
            fill=0, p=0.15
        ),
        A.CLAHE(p=0.2),                      # contrast-limited AHE for shadows
        A.Sharpen(p=0.15),                   # sharpens building edge details
        # ── Normalise ──────────────────────────────────────────────────────
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ],
    additional_targets={
        'edge_mask': 'mask',   # receives same spatial transforms as mask
        'dist_map':  'mask',   # same — no photometric transforms applied
    },
)

# ── 4f  Building Validation Transform ────────────────────────────────────────
building_val_transform = A.Compose(
    [
        A.Resize(640, 640),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ],
    additional_targets={
        'edge_mask': 'mask',
        'dist_map':  'mask',
    },
)