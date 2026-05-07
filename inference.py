import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import json
import numpy as np
import cv2
import torch
from PIL import Image

from models  import get_road_model, get_landcover_model, get_building_model
from dataset import val_transform, building_val_transform

# ── Tier 1 Road Intelligence Layer (optional) ────────────────────────────────
_TIER1_AVAILABLE = False
try:
    from road_width           import RoadWidthEstimator
    from road_type_classifier  import RoadTypeClassifier
    from vizualize_road_tier1  import save_tier1_figure
    _TIER1_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# inference.py  –  Single-Image Inference for All Pipeline Stages
# =============================================================================
# Supported model types:
#   road      – binary road segmentation          (DeepLabV3+ ResNet34, 512×512)
#   landcover – 7-class land cover classification (DeepLabV3+ ResNet34, 512×512)
#   building  – binary building footprint         (UnetPlusPlus ResNet50, 640×640)
#
# Usage (CLI):
#   python inference.py --image sat.jpg --model road
#   python inference.py --image tile.tif --model building
# =============================================================================

# ── Kaggle model paths ────────────────────────────────────────────────────────
# Dataset "best path" (ayushks07) is mounted at /kaggle/input/datasets/ayushks07/best-path/
ROAD_WEIGHTS      = '/kaggle/input/datasets/ayushks07/best-path/road_model_best.pth'
LANDCOVER_WEIGHTS = '/kaggle/input/datasets/ayushks07/best-path/landcover_best.pth'
BUILDING_WEIGHTS  = '/kaggle/input/datasets/ayushks07/best-path/building_model_best.pth'
RESULTS_DIR       = '/kaggle/working/results'

# Building detection threshold (lower than road/LC due to fine footprints)
BUILDING_THRESHOLD = 0.5

# Land cover colour palette (RGB) — matches DeepGlobe label spec
LANDCOVER_COLORS = {
    0: [0,   255, 255],   # Urban land
    1: [255, 255, 0  ],   # Agriculture
    2: [255, 0,   255],   # Rangeland
    3: [0,   255, 0  ],   # Forest
    4: [0,   0,   255],   # Water
    5: [255, 255, 255],   # Barren land
    6: [0,   0,   0  ],   # Unknown
}


def _load_model(model_type: str, device: torch.device):
    """Instantiate model + load weights for the given model_type."""
    if model_type == 'road':
        model        = get_road_model()
        weights_path = ROAD_WEIGHTS
    elif model_type == 'landcover':
        model        = get_landcover_model()
        weights_path = LANDCOVER_WEIGHTS
    elif model_type == 'building':
        model        = get_building_model()
        weights_path = BUILDING_WEIGHTS
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: road, landcover, building")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Handle plain state_dict, 'model_state' (road/inpainting/building), and
    # 'model_state_dict' (landcover — train_landcover.py naming convention)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    model.to(device).eval()
    print(f"✅ Loaded {model_type} model  ←  {weights_path}")
    return model


def _preprocess(image_path: str, model_type: str, device: torch.device):
    """
    Load and pre-process image for the specified model.

    Returns:
        tensor  : (1, 3, H, W) float32 on device — ready for model.forward()
        vis_img : (H, W, 3)   uint8 RGB            — for saving / display
    """
    if model_type == 'building':
        # PIL handles TIFF/GeoTIFF without libtiff warnings
        pil  = Image.open(image_path).convert('RGB')
        arr  = np.array(pil, dtype=np.uint8)
        dummy = np.zeros(arr.shape[:2], dtype=np.float32)
        aug  = building_val_transform(
            image=arr, mask=dummy, edge_mask=dummy, dist_map=dummy)
        tensor  = aug['image'].unsqueeze(0).to(device)
        vis_img = arr
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = val_transform(image=rgb)['image'].unsqueeze(0).to(device)
        vis_img = rgb

    return tensor, vis_img


def run_inference(image_path: str, model_type: str = 'road') -> str:
    """
    Run single-image inference and save the prediction mask.

    Args:
        image_path : path to input satellite image
        model_type : 'road' | 'landcover' | 'building'

    Returns:
        save_path : absolute path of saved prediction PNG
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(model_type, device)
    tensor, _ = _preprocess(image_path, model_type, device)

    with torch.no_grad():
        logits = model(tensor)

    # ── Post-process: model-type–specific decoding ─────────────────────────
    if model_type in ('road', 'building'):
        thr  = BUILDING_THRESHOLD if model_type == 'building' else 0.5
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (prob > thr).astype(np.uint8) * 255          # (H, W) grayscale
        out  = mask                                          # 1-channel for imwrite

    else:  # landcover — argmax → colour map
        ids  = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # (H, W)
        h, w = ids.shape
        rgb_out = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, colour in LANDCOVER_COLORS.items():
            rgb_out[ids == cls_id] = colour
        out  = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)     # 3-channel for imwrite

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem      = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(RESULTS_DIR, f"{stem}_{model_type}_pred.png")
    cv2.imwrite(save_path, out)
    print(f"💾 Saved → {save_path}")
    return save_path


# ───────────────────────────────────────────────────────────────────────────────
# Tier 1 Road Intelligence
# ───────────────────────────────────────────────────────────────────────────────

def run_tier1_inference(image_path:   str,
                        road_mask_np: np.ndarray,
                        vis_image_np: np.ndarray,
                        results_dir:  str = RESULTS_DIR) -> dict:
    """
    Run the Tier 1 Road Intelligence Layer on a binary road mask and save
    all derived artefacts to *results_dir*.

    Artefacts saved:
        ``<stem>_width_heatmap.png``    — plasma width heatmap
        ``<stem>_surface_overlay.png``  — surface type colour overlay
        ``<stem>_tier1_figure.png``     — 5-panel composite figure
        ``<stem>_tier1_summary.json``   — JSON summary statistics

    Args:
        image_path   : path to the original satellite image (used for stem).
        road_mask_np : (H, W) uint8 binary road mask (0/255).
        vis_image_np : (H, W, 3) uint8 RGB satellite image for visualisation.
        results_dir  : output directory.

    Returns:
        dict with keys ``width_result``, ``type_result``, ``summary``,
        and ``saved_paths`` (sub-dict of artefact paths).

    Raises:
        RuntimeError : If Tier 1 modules are not importable.
    """
    if not _TIER1_AVAILABLE:
        raise RuntimeError(
            "Tier 1 modules unavailable.  "
            "Install scikit-image and scikit-learn:"
            "  pip install scikit-image scikit-learn")

    os.makedirs(results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # ── Module 1: Road Width Estimation ───────────────────────────────────
    width_est    = RoadWidthEstimator()
    width_result = width_est.analyse(road_mask_np)
    print(f"  Tier1-M1 ✓  skeleton px: {width_result.skeleton.sum():,}  | "
          f"mean width: {width_result.summary_stats['mean_m']:.2f} m")

    # ── Module 2: Road Surface Type Classifier ───────────────────────────
    clf         = RoadTypeClassifier()
    type_result = clf.predict(vis_image_np, road_mask_np, width_result=width_result)
    print(f"  Tier1-M2 ✓  dominant surface: "
          f"{type_result['summary']['dominant_type']} | "
          f"sampled: {type_result['summary']['n_sampled_pts']} pts")

    # ── Save individual artefacts ───────────────────────────────────────────
    heatmap_path = os.path.join(results_dir, f"{stem}_width_heatmap.png")
    surface_path = os.path.join(results_dir, f"{stem}_surface_overlay.png")
    figure_path  = os.path.join(results_dir, f"{stem}_tier1_figure.png")
    json_path    = os.path.join(results_dir, f"{stem}_tier1_summary.json")

    # Width heatmap (BGR for cv2)
    cv2.imwrite(heatmap_path,
                cv2.cvtColor(width_result.width_heatmap_rgb, cv2.COLOR_RGB2BGR))
    print(f"💾  Width heatmap    → {heatmap_path}")

    # Surface overlay (BGR for cv2)
    cv2.imwrite(surface_path,
                cv2.cvtColor(type_result['overlay_rgb'], cv2.COLOR_RGB2BGR))
    print(f"💾  Surface overlay  → {surface_path}")

    # 5-panel composite figure
    tier1_bundle = {'width_result': width_result, 'type_result': type_result}
    save_tier1_figure(
        vis_image_np, road_mask_np, tier1_bundle, figure_path,
        title=f'Tier 1 Road Intelligence  |  {stem}')

    # JSON summary (serialise non-primitive values)
    summary = {
        'mean_width_m':     width_result.summary_stats['mean_m'],
        'median_width_m':   width_result.summary_stats['median_m'],
        'std_width_m':      width_result.summary_stats['std_m'],
        'min_width_m':      width_result.summary_stats['min_m'],
        'max_width_m':      width_result.summary_stats['max_m'],
        'width_class_dist': width_result.class_distribution,
        'dominant_surface': type_result['summary']['dominant_type'],
        'surface_counts':   type_result['summary']['type_counts'],
        'skeleton_pixels':  int(width_result.skeleton.sum()),
        'is_empty':         bool(width_result.is_empty),
    }
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"💾  Tier 1 summary   → {json_path}")

    return {
        'width_result': width_result,
        'type_result':  type_result,
        'summary':      summary,
        'saved_paths': {
            'heatmap':  heatmap_path,
            'surface':  surface_path,
            'figure':   figure_path,
            'json':     json_path,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single-image satellite inference — road / landcover / building')
    parser.add_argument('--image', required=True,
                        help='Path to input satellite image (JPG / PNG / TIFF)')
    parser.add_argument('--model', default='road',
                        choices=['road', 'landcover', 'building'],
                        help='Model type to run  (default: road)')
    parser.add_argument('--tier1', action='store_true',
                        help=(
                            'Run Tier 1 Road Intelligence Layer after road '
                            'inference (Module 1: width estimation, '
                            'Module 2: surface type classification).  '
                            'Only meaningful with --model road.  '
                            'Saves width heatmap, surface overlay, 5-panel '
                            'figure, and JSON summary to RESULTS_DIR.'))
    args = parser.parse_args()

    # ── Standard inference ────────────────────────────────────────────────────────
    run_inference(args.image, model_type=args.model)

    # ── Tier 1 (road-only) ────────────────────────────────────────────────────────
    if args.tier1:
        if args.model != 'road':
            print("⚠️  --tier1 is only applicable for --model road.  Skipping.")
        elif not _TIER1_AVAILABLE:
            print("❌  Tier 1 modules not found.  "
                  "Install scikit-image and scikit-learn to use --tier1.")
        else:
            # Re-load the Stage 1 mask that was just saved
            stem      = os.path.splitext(os.path.basename(args.image))[0]
            mask_path = os.path.join(RESULTS_DIR, f"{stem}_road_pred.png")

            if not os.path.exists(mask_path):
                print(f"❌  Road mask not found at {mask_path}. "
                      "Cannot run Tier 1.")
            else:
                road_mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Load the satellite image for texture features
                bgr      = cv2.imread(args.image)
                vis_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None \
                           else np.zeros((*road_mask_np.shape, 3), dtype=np.uint8)

                print("\n🚀 Running Tier 1 Road Intelligence Layer…")
                t1_result = run_tier1_inference(
                    args.image, road_mask_np, vis_rgb, results_dir=RESULTS_DIR)
                print(f"\n✅ Tier 1 complete → figure: {t1_result['saved_paths']['figure']}")