# =============================================================================
# pipeline.py  –  End-to-End Two-Stage Road Extraction + Inpainting Pipeline
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  FULL PIPELINE                                                           │
# │                                                                          │
# │  INPUT  : Raw satellite image (B, 3, H, W) RGB                          │
# │                                                                          │
# │  STAGE 1: DeepLabV3+ ResNet34                                            │
# │           Satellite Image → Binary Road Mask                             │
# │           (B, 3, 512, 512) → (B, 1, 512, 512)                           │
# │                                                                          │
# │  [Optional]: Provide or auto-detect hole mask                            │
# │                                                                          │
# │  STAGE 2: Partial Conv U-Net                                             │
# │           Incomplete Road Mask → Complete Road Mask                      │
# │           (B, 2, 512, 512) → (B, 1, 512, 512)                           │
# │                                                                          │
# │  OUTPUT : Completed road mask + 4-panel visualisation                   │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   from pipeline import RoadInpaintingPipeline
#   pipe   = RoadInpaintingPipeline(stage1_path, stage2_path)
#   result = pipe.run(image_path, hole_mask_path=None)
#
# Or from CLI:
#   python pipeline.py --image sat.jpg --s1 stage1.pth --s2 stage2.pth
# =============================================================================

import os
import argparse
import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stage 1 imports
from models  import get_road_model
from dataset import val_transform      # Stage 1 image pre-processing

# Stage 2 imports
from inpainting_model  import get_inpainting_model
from infer_inpainting  import auto_detect_holes

# ── Kaggle dataset paths ────────────────────────────────────────────────────────
# Dataset "best path" is mounted at /kaggle/input/best-path/
_W_ROAD     = '/kaggle/input/best-path/road_model_best.pth'
_W_INPAINT  = '/kaggle/input/best-path/inpainting_best.pth'
_W_LC       = '/kaggle/input/best-path/landcover_best.pth'
_W_BUILDING = '/kaggle/input/best-path/building_model_best.pth'
_RESULTS    = '/kaggle/working/results/pipeline'

# ImageNet normalisation (must match dataset.py val_transform)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Image Pre-processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_satellite(image_path: str, size: int = 512) -> tuple:
    """
    Load and pre-process a satellite image for Stage 1.

    Returns:
        tensor     : (1, 3, 512, 512) float32 — normalised, ready for model
        vis_img    : (512, 512, 3) float32 in [0,1] — for visualisation
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0

    vis_img = img_rgb.copy()

    # Apply ImageNet normalisation
    normalised = (img_rgb - _MEAN) / _STD   # (512, 512, 3)

    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.float(), vis_img


def load_hole_mask(mask_path: str, size: int = 512) -> np.ndarray:
    """
    Load an explicit hole mask from file.
    Convention: white (255) = valid/known, black (0) = hole.

    Returns:
        hole_mask : (H, W) float32 — 1=valid, 0=hole
    """
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.where(raw >= 128, 1.0, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_figure(vis_img:      np.ndarray,
                         stage1_mask:  np.ndarray,
                         corrupted:    np.ndarray,
                         stage2_mask:  np.ndarray,
                         save_path:    str):
    """
    4-panel pipeline visualisation:
        1. Input Satellite
        2. Stage 1 Road Mask  (raw extraction)
        3. Corrupted Mask     (with holes)
        4. Stage 2 Completed  (inpainted)
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("End-to-End Road Extraction + Inpainting Pipeline",
                 fontsize=14, fontweight='bold', y=1.01)

    axes[0].imshow(vis_img)
    axes[0].set_title("Input Satellite", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(stage1_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Stage 1: Road Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(corrupted, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Corrupted (with holes)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(stage2_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Stage 2: Completed Mask", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼️  Pipeline figure saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Class
# ─────────────────────────────────────────────────────────────────────────────

class RoadInpaintingPipeline:
    """
    End-to-end two-stage pipeline: satellite image → complete road mask.

    Args:
        stage1_path : path to Stage 1 DeepLabV3+ weights (.pth)
        stage2_path : path to Stage 2 Partial Conv U-Net weights (.pth)
        device      : torch.device, or None for auto-detect
        threshold1  : Stage 1 binarisation threshold (default 0.5)
        threshold2  : Stage 2 binarisation threshold (default 0.5)
    """

    def __init__(self,
                 stage1_path: str,
                 stage2_path: str,
                 device:      torch.device = None,
                 threshold1:  float = 0.5,
                 threshold2:  float = 0.5):

        self.device     = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold1 = threshold1
        self.threshold2 = threshold2

        print(f"🚀 Loading pipeline on: {self.device}")

        # ── Stage 1: DeepLabV3+ ResNet34 ──────────────────────────────────────
        self.stage1 = get_road_model().to(self.device)
        self.stage1.load_state_dict(
            torch.load(stage1_path, map_location=self.device))
        self.stage1.eval()
        print(f"  ✅ Stage 1 loaded: {stage1_path}")

        # ── Stage 2: Partial Conv U-Net ────────────────────────────────────────
        self.stage2 = get_inpainting_model(base_channels=64).to(self.device)
        self.stage2.load_state_dict(
            torch.load(stage2_path, map_location=self.device))
        self.stage2.eval()
        print(f"  ✅ Stage 2 loaded: {stage2_path}")

    def run(self,
            image_path:      str,
            hole_mask_path:  str   = None,
            min_hole_frac:   float = 0.05,
            save_dir:        str   = '/kaggle/working/results/pipeline',
            save_figure:     bool  = True) -> dict:
        """
        Run the full two-stage pipeline on a satellite image.

        Args:
            image_path     : path to satellite image (JPG/PNG)
            hole_mask_path : optional path to explicit hole mask
                             (white=valid, black=hole).
                             If None, auto-detect from Stage 1 output.
            min_hole_frac  : minimum area fraction for auto-detection
            save_dir       : output directory for results
            save_figure    : whether to save the 4-panel figure

        Returns:
            dict with keys:
                'stage1_mask'   — (512, 512) float32 Stage 1 road mask
                'hole_mask'     — (512, 512) float32 hole mask used
                'corrupted'     — (512, 512) float32 corrupted mask
                'stage2_mask'   — (512, 512) float32 completed road mask
                'output_path'   — path of saved completed mask PNG
        """
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # ── Step 1: Load satellite image ──────────────────────────────────────
        img_tensor, vis_img = load_satellite(image_path, size=512)
        # img_tensor : (1, 3, 512, 512)
        # vis_img    : (512, 512, 3) float32

        img_tensor = img_tensor.to(self.device)

        # ── Step 2: Stage 1 — Extract road mask ───────────────────────────────
        with torch.no_grad():
            with autocast():
                stage1_logits = self.stage1(img_tensor)   # (1, 1, 512, 512)

        stage1_prob = torch.sigmoid(stage1_logits).squeeze().cpu().numpy()
        # (512, 512) float32

        stage1_mask = (stage1_prob > self.threshold1).astype(np.float32)
        # (512, 512) float32 binary road mask

        print(f"📍 Stage 1 complete | Road coverage: "
              f"{stage1_mask.mean()*100:.1f}%")

        # ── Step 3: Determine hole mask ───────────────────────────────────────
        if hole_mask_path is not None:
            hole_mask = load_hole_mask(hole_mask_path, size=512)
            print(f"🗺️  Hole mask loaded from: {hole_mask_path}")
        else:
            hole_mask = auto_detect_holes(stage1_mask, min_frac=min_hole_frac)

        # ── Step 4: Build corrupted mask ──────────────────────────────────────
        corrupted = stage1_mask * hole_mask   # (512, 512)

        # ── Step 5: Stage 2 — Inpaint holes ───────────────────────────────────
        # Model input: (1, 2, 512, 512) = [corrupted_mask, hole_mask]
        c_t = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).to(self.device)
        h_t = torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        # c_t : (1, 1, 512, 512)
        # h_t : (1, 1, 512, 512)

        with torch.no_grad():
            with autocast():
                stage2_pred = self.stage2(c_t, h_t)   # (1, 1, 512, 512)

        stage2_prob = stage2_pred.squeeze().cpu().numpy()   # (512, 512)
        stage2_mask = (stage2_prob > self.threshold2).astype(np.float32)

        # Preserve known-good regions from Stage 1 in final output
        final_mask = stage2_mask * (1 - hole_mask) + stage1_mask * hole_mask
        # (512, 512) float32

        print(f"📍 Stage 2 complete | Final coverage: "
              f"{final_mask.mean()*100:.1f}%")

        # ── Step 6: Save outputs ──────────────────────────────────────────────
        out_name    = f"pipeline_{ts}_completed.png"
        output_path = os.path.join(save_dir, out_name)
        cv2.imwrite(output_path, (final_mask * 255).astype(np.uint8))
        print(f"💾 Complete road mask saved: {output_path}")

        if save_figure:
            fig_path = os.path.join(save_dir, f"pipeline_{ts}_figure.png")
            save_pipeline_figure(vis_img, stage1_mask, corrupted, final_mask,
                                 save_path=fig_path)

        return {
            'stage1_mask':  stage1_mask,
            'hole_mask':    hole_mask,
            'corrupted':    corrupted,
            'stage2_mask':  final_mask,
            'output_path':  output_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point  (Road + Inpainting pipeline)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__' and False:   # guarded — see unified CLI below
    pass


# =============================================================================
# SatellitePipeline  –  Unified 4-Stage End-to-End Pipeline
# =============================================================================
# Combines all four trained models into a single inference call:
#
#   Stage 1 : Road Extraction       DeepLabV3+ ResNet34   (512×512)
#   Stage 2 : Road Inpainting       Partial Conv U-Net    (512×512)
#   Stage 3 : Land Cover            DeepLabV3+ ResNet34   (512×512, 7-class)
#   Stage 4 : Building Detection    UnetPlusPlus ResNet50 (640×640)
#
# Output: 5-panel figure  +  individual mask PNGs
# =============================================================================

from models  import get_landcover_model, get_building_model
from dataset import building_val_transform
from PIL     import Image


# Land cover colour palette (RGB) — matches DeepGlobe label specification
_LC_COLORS = np.array([
    [0,   255, 255],   # 0  Urban land
    [255, 255, 0  ],   # 1  Agriculture
    [255, 0,   255],   # 2  Rangeland
    [0,   255, 0  ],   # 3  Forest
    [0,   0,   255],   # 4  Water
    [255, 255, 255],   # 5  Barren land
    [0,   0,   0  ],   # 6  Unknown
], dtype=np.uint8)

_LC_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest',
             'Water', 'Barren', 'Unknown']


class SatellitePipeline:
    """
    Unified 4-stage satellite image analysis pipeline.

    Args:
        road_path     : path to Stage-1 road extraction weights
        inpaint_path  : path to Stage-2 inpainting weights
        lc_path       : path to Stage-3 land cover weights
        building_path : path to Stage-4 building detection weights
        device        : torch.device or None (auto-detect)
        road_thr      : road binarisation threshold  (default 0.5)
        building_thr  : building binarisation threshold (default 0.5)
    """

    def __init__(self,
                 road_path:     str,
                 inpaint_path:  str,
                 lc_path:       str,
                 building_path: str,
                 device:        torch.device = None,
                 road_thr:      float = 0.5,
                 building_thr:  float = 0.5):

        self.device      = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.road_thr    = road_thr
        self.build_thr   = building_thr

        print(f"🛰️  SatellitePipeline initialising on: {self.device}")
        self.road_model, self.inpaint_model, self.lc_model, self.build_model = \
            self._load_all(road_path, inpaint_path, lc_path, building_path)
        print("✅ All 4 models loaded.\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_all(self, road_path, inpaint_path, lc_path, building_path):
        """Load + return the four models."""
        def _load(model, path, label):
            state = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and 'model_state' in state:
                model.load_state_dict(state['model_state'])
            else:
                model.load_state_dict(state)
            model.to(self.device).eval()
            print(f"  ✅ {label}  ←  {path}")
            return model

        m1 = _load(get_road_model(),       road_path,    'Stage 1 Road Extraction')
        m2 = _load(get_inpainting_model(base_channels=64),
                   inpaint_path,            'Stage 2 Road Inpainting')
        m3 = _load(get_landcover_model(),   lc_path,     'Stage 3 Land Cover')
        m4 = _load(get_building_model(),    building_path,'Stage 4 Building Det.')
        return m1, m2, m3, m4

    @staticmethod
    def _load_image_512(image_path: str):
        """Load with cv2, resize to 512, return (tensor, vis_img_float)."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_512 = cv2.resize(rgb, (512, 512)).astype(np.float32) / 255.0
        norm    = (rgb_512 - _MEAN) / _STD
        tensor  = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, rgb_512                           # tensor for model, float vis

    @staticmethod
    def _load_image_640(image_path: str):
        """Load with PIL (TIFF-safe), apply building_val_transform → tensor."""
        pil  = Image.open(image_path).convert('RGB')
        arr  = np.array(pil, dtype=np.uint8)
        dummy = np.zeros(arr.shape[:2], dtype=np.float32)
        aug  = building_val_transform(
            image=arr, mask=dummy, edge_mask=dummy, dist_map=dummy)
        return aug['image'].unsqueeze(0).float()         # (1,3,640,640)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self,
            image_path:    str,
            hole_mask_path: str   = None,
            min_hole_frac:  float = 0.05,
            save_dir:       str   = '/kaggle/working/results/pipeline',
            save_figure:    bool  = True) -> dict:
        """
        Run the full 4-stage pipeline on a satellite image.

        Args:
            image_path      : path to satellite image (JPG/PNG/TIFF)
            hole_mask_path  : optional explicit hole mask for Stage 2
            min_hole_frac   : minimum hole fraction for auto-detection
            save_dir        : output directory
            save_figure     : save 5-panel composite figure

        Returns:
            dict with keys:
                road_mask      – (512,512) float32 completed road network
                lc_mask        – (512,512) int32   7-class land cover ids
                lc_rgb         – (512,512,3) uint8 colour-coded land cover
                building_mask  – (640,640) float32 binary building footprints
                figure_path    – path to saved 5-panel figure (or None)
        """
        os.makedirs(save_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # ── Stage 1: Road extraction ──────────────────────────────────────────
        t512, vis512 = self._load_image_512(image_path)
        t512 = t512.to(self.device)

        with torch.no_grad():
            with autocast():
                road_logits = self.road_model(t512)
        road_prob = torch.sigmoid(road_logits).squeeze().cpu().numpy()
        road_bin  = (road_prob > self.road_thr).astype(np.float32)
        print(f"  Stage 1 ✓  road coverage {road_bin.mean()*100:.1f}%")

        # ── Stage 2: Inpainting ───────────────────────────────────────────────
        if hole_mask_path:
            hole_mask = load_hole_mask(hole_mask_path)
        else:
            hole_mask = auto_detect_holes(road_bin, min_frac=min_hole_frac)

        corrupted = road_bin * hole_mask
        c_t = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).to(self.device)
        h_t = torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast():
                inpaint_out = self.inpaint_model(c_t, h_t)
        inpaint_bin = (inpaint_out.squeeze().cpu().numpy() > self.road_thr).astype(np.float32)
        road_final  = inpaint_bin * (1 - hole_mask) + road_bin * hole_mask
        print(f"  Stage 2 ✓  road after inpainting {road_final.mean()*100:.1f}%")

        # ── Stage 3: Land cover ───────────────────────────────────────────────
        with torch.no_grad():
            with autocast():
                lc_logits = self.lc_model(t512)
        lc_ids  = torch.argmax(lc_logits, dim=1).squeeze().cpu().numpy().astype(np.int32)
        lc_rgb  = _LC_COLORS[lc_ids]               # (512,512,3) uint8
        print(f"  Stage 3 ✓  dominant class: {_LC_NAMES[int(np.bincount(lc_ids.ravel()).argmax())]}")

        # ── Stage 4: Building detection ───────────────────────────────────────
        t640 = self._load_image_640(image_path).to(self.device)
        with torch.no_grad():
            with autocast():
                build_logits = self.build_model(t640)
        build_prob = torch.sigmoid(build_logits).squeeze().cpu().numpy()
        build_bin  = (build_prob > self.build_thr).astype(np.float32)
        print(f"  Stage 4 ✓  building coverage {build_bin.mean()*100:.2f}%")

        # ── Save individual masks ─────────────────────────────────────────────
        cv2.imwrite(os.path.join(save_dir, f"{stem}_road.png"),
                    (road_final * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{stem}_landcover.png"),
                    cv2.cvtColor(lc_rgb, cv2.COLOR_RGB2BGR))
        build_512 = cv2.resize(build_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_dir, f"{stem}_buildings.png"),
                    (build_512 * 255).astype(np.uint8))

        # ── 5-panel figure ────────────────────────────────────────────────────
        fig_path = None
        if save_figure:
            fig_path = os.path.join(save_dir, f"{stem}_pipeline_{ts}.png")
            self._save_figure(vis512, road_final, lc_rgb, build_512, fig_path, stem)

        return {
            'road_mask':     road_final,
            'lc_mask':       lc_ids,
            'lc_rgb':        lc_rgb,
            'building_mask': build_bin,
            'figure_path':   fig_path,
        }

    @staticmethod
    def _save_figure(vis, road, lc_rgb, buildings, save_path, title=''):
        """5-panel composite: Satellite | Road | Land Cover | Buildings | Overlay."""
        # Build composite overlay
        overlay = (vis * 255).astype(np.uint8).copy()
        # Roads: cyan
        overlay[road > 0] = np.clip(
            overlay[road > 0].astype(int) // 2 + np.array([0, 128, 128]), 0, 255)
        # Buildings: orange
        overlay[buildings > 0] = np.clip(
            overlay[buildings > 0].astype(int) // 2 + np.array([128, 64, 0]), 0, 255)

        panels = [
            (vis,                     'Input Satellite'),
            (road,                    'Road Network\n(Stage 1+2)'),
            (lc_rgb.astype(np.float32)/255, 'Land Cover\n(Stage 3)'),
            (buildings,               'Building Footprints\n(Stage 4)'),
            (overlay.astype(np.float32)/255,'Composite Overlay'),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(f'4-Stage Satellite Pipeline  |  {title}',
                     fontsize=13, fontweight='bold', y=1.01)

        for ax, (img, label) in zip(axes, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼️  Pipeline figure → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Unified CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Satellite Analysis Pipeline — Road, Land Cover, Building')
    sub = parser.add_subparsers(dest='cmd')

    # ── road sub-command  (original 2-stage) ──────────────────────────────────
    p_road = sub.add_parser('road', help='2-stage road extraction + inpainting')
    p_road.add_argument('--image',  required=True)
    p_road.add_argument('--s1',     required=True, help='Stage-1 weights .pth')
    p_road.add_argument('--s2',     required=True, help='Stage-2 weights .pth')
    p_road.add_argument('--hole',   default=None)
    p_road.add_argument('--outdir', default='/kaggle/working/results/pipeline')
    p_road.add_argument('--t1',     type=float, default=0.5)
    p_road.add_argument('--t2',     type=float, default=0.5)

    # ── full sub-command  (4-stage unified) ───────────────────────────────────
    p_full = sub.add_parser('full', help='4-stage unified pipeline')
    p_full.add_argument('--image',    required=True)
    p_full.add_argument('--road',     required=True, help='Stage-1 road weights')
    p_full.add_argument('--inpaint',  required=True, help='Stage-2 inpainting weights')
    p_full.add_argument('--lc',       required=True, help='Stage-3 land cover weights')
    p_full.add_argument('--building', required=True, help='Stage-4 building weights')
    p_full.add_argument('--outdir',   default='/kaggle/working/results/pipeline')
    p_full.add_argument('--road_thr', type=float, default=0.5)
    p_full.add_argument('--build_thr',type=float, default=0.5)

    args = parser.parse_args()

    if args.cmd == 'road':
        pipe   = RoadInpaintingPipeline(args.s1, args.s2,
                                        threshold1=args.t1, threshold2=args.t2)
        result = pipe.run(args.image, hole_mask_path=args.hole, save_dir=args.outdir)
        print(f"\n✅ Road pipeline complete → {result['output_path']}")

    elif args.cmd == 'full':
        pipe   = SatellitePipeline(
            road_path=args.road, inpaint_path=args.inpaint,
            lc_path=args.lc,     building_path=args.building,
            road_thr=args.road_thr, building_thr=args.build_thr)
        result = pipe.run(args.image, save_dir=args.outdir)
        print(f"\n✅ Full pipeline complete → {result['figure_path']}")

    else:
        parser.print_help()
