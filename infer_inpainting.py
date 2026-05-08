# =============================================================================
# infer_inpainting.py  –  Stage 2 Inference + Tile Stitching
# =============================================================================
# Usage modes:
#   1. With explicit hole mask : provide both mask_path and hole_mask_path
#   2. Auto-detect holes       : provide only mask_path; large connected black
#      regions (> 5% of image) are automatically treated as holes
#   3. Tiled inference         : large masks are split into 512×512 tiles with
#      64px overlap, predictions averaged in the overlap bands
# =============================================================================

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast

from inpainting_model import get_inpainting_model

# ── Tier 1 modules (optional — graceful fallback) ─────────────────────────────
try:
    from road_width           import RoadWidthEstimator
    from road_type_classifier  import RoadTypeClassifier
    from road_router           import RoadRouter, _pick_route_endpoints
    _TIER1_OK = True
except ImportError as _e:
    _TIER1_OK = False
    print(f"[Tier1 WARNING] Tier 1 modules not importable: {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Auto-detect Holes
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_holes(binary_mask: np.ndarray,
                      min_frac:    float = 0.05) -> np.ndarray:
    """
    Detect large connected black regions as missing holes.

    A connected component of background pixels (value=0) is classified as a
    hole if its area exceeds min_frac × total_pixels.

    Args:
        binary_mask : (H, W) float32 {0, 1} road mask
        min_frac    : minimum fraction of image area to flag as a hole

    Returns:
        hole_mask : (H, W) float32 — 1=valid/known, 0=detected hole
    """
    H, W   = binary_mask.shape
    thresh = int(min_frac * H * W)

    # Background = 0 in road mask; invert to find black blobs
    bg_mask = (binary_mask < 0.5).astype(np.uint8)   # (H, W) uint8

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg_mask, connectivity=8
    )

    # Label 0 = foreground (road), skip it
    hole_mask = np.ones((H, W), dtype=np.float32)   # start all valid

    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= thresh:
            hole_mask[labels == lbl] = 0.0   # mark as hole

    n_holes = int((hole_mask == 0).sum())
    print(f"🔍 Auto-detected hole pixels: {n_holes} "
          f"({100.*n_holes/(H*W):.1f}% of image)")
    return hole_mask


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Single-tile Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_single(model:         torch.nn.Module,
                 corrupted:     np.ndarray,
                 hole_mask:     np.ndarray,
                 device:        torch.device,
                 threshold:     float = 0.5) -> np.ndarray:
    """
    Run the inpainting model on a single 512×512 patch.

    Args:
        model     : loaded PartialConvUNet (eval mode)
        corrupted : (H, W) float32 road mask with holes zeroed
        hole_mask : (H, W) float32 — 1=valid, 0=hole
        device    : torch device
        threshold : binarisation threshold

    Returns:
        filled : (H, W) float32 binary mask with holes filled
    """
    # (H, W) → (1, 1, H, W)
    c_t = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).to(device)
    h_t = torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            pred = model(c_t, h_t)   # (1, 1, H, W)

    prob   = pred.squeeze().cpu().numpy()           # (H, W)
    filled = (prob > threshold).astype(np.float32)  # (H, W)

    # Preserve known regions exactly (only fill the holes)
    filled = filled * (1 - hole_mask) + corrupted * hole_mask
    return filled.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Tiled Inference with Overlap Averaging
# ─────────────────────────────────────────────────────────────────────────────

def infer_tiled(model:      torch.nn.Module,
                corrupted:  np.ndarray,
                hole_mask:  np.ndarray,
                device:     torch.device,
                tile_size:  int   = 512,
                overlap:    int   = 64,
                threshold:  float = 0.5) -> np.ndarray:
    """
    Tile a large mask into overlapping 512×512 patches, run inpainting
    on each, then stitch results together using weighted averaging in
    the overlap regions.

    Args:
        model      : loaded model
        corrupted  : (H, W) float32 — large road mask with holes
        hole_mask  : (H, W) float32 — 1=valid, 0=hole
        device     : torch device
        tile_size  : tile spatial size (default 512, must match model input)
        overlap    : overlap between adjacent tiles in pixels (default 64)
        threshold  : binarisation threshold

    Returns:
        stitched : (H, W) float32 binary filled mask
    """
    H, W   = corrupted.shape
    stride = tile_size - overlap

    # Accumulators (use soft probability averaging, then binarise at end)
    pred_acc   = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)

    # Build a 2D Hann window for smooth overlap blending
    hann_1d = np.hanning(tile_size).astype(np.float64)
    hann_2d = np.outer(hann_1d, hann_1d)   # (tile_size, tile_size)

    y_starts = list(range(0, H - tile_size + 1, stride))
    x_starts = list(range(0, W - tile_size + 1, stride))

    # Ensure we cover right/bottom edges
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    total_tiles = len(y_starts) * len(x_starts)
    print(f"🔲 Tiled inference: {total_tiles} tiles "
          f"({len(y_starts)}×{len(x_starts)}) | "
          f"tile={tile_size}px overlap={overlap}px")

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + tile_size
            x1 = x0 + tile_size

            tile_c = corrupted[y0:y1, x0:x1]    # (512, 512)
            tile_h = hole_mask[y0:y1, x0:x1]    # (512, 512)

            # Resize if tile is smaller than tile_size (edge case)
            if tile_c.shape != (tile_size, tile_size):
                tile_c = cv2.resize(tile_c, (tile_size, tile_size),
                                    interpolation=cv2.INTER_NEAREST)
                tile_h = cv2.resize(tile_h, (tile_size, tile_size),
                                    interpolation=cv2.INTER_NEAREST)

            c_t = torch.from_numpy(tile_c).unsqueeze(0).unsqueeze(0).to(device)
            h_t = torch.from_numpy(tile_h).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    pred = model(c_t, h_t)   # (1, 1, 512, 512)

            prob = pred.squeeze().cpu().numpy().astype(np.float64)   # (512, 512)

            pred_acc[y0:y1, x0:x1]   += prob   * hann_2d
            weight_acc[y0:y1, x0:x1] += hann_2d

    # Normalise by accumulated weights
    weight_acc = np.where(weight_acc > 0, weight_acc, 1.0)
    avg_prob   = pred_acc / weight_acc   # (H, W) float64

    stitched   = (avg_prob > threshold).astype(np.float32)

    # Preserve known regions exactly
    stitched = stitched * (1 - hole_mask) + corrupted * hole_mask
    return stitched.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ Main Inference Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model_path:      str,
                  mask_path:       str,
                  hole_mask_path:  str   = None,
                  output_path:     str   = None,
                  threshold:       float = 0.50,
                  min_hole_frac:   float = 0.05,
                  use_tiling:      bool  = False,
                  tile_size:       int   = 512,
                  tile_overlap:    int   = 64) -> np.ndarray:
    """
    Run Stage 2 inpainting inference on a road mask.

    Args:
        model_path      : path to trained inpainting model weights
        mask_path       : path to input road mask PNG (grayscale or binary)
        hole_mask_path  : optional path to hole mask PNG (0=hole, 255=valid)
                          If None, holes are auto-detected.
        output_path     : where to save the filled mask PNG; None = auto-name
        threshold       : binarisation threshold (default 0.5)
        min_hole_frac   : minimum fraction for auto-detect (default 0.05)
        use_tiling      : if True, use tiled inference for large images
        tile_size       : tile size for tiled mode (default 512)
        tile_overlap    : overlap between tiles in pixels (default 64)

    Returns:
        filled : (H, W) float32 binary road mask with holes completed
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Inpainting Inference | Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = get_inpainting_model(base_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded: {model_path}")

    # ── Load road mask ─────────────────────────────────────────────────────────
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    orig_H, orig_W = raw.shape

    # Resize to multiple of 512 if needed (round up)
    if use_tiling:
        target_H = max(512, int(np.ceil(orig_H / 512) * 512))
        target_W = max(512, int(np.ceil(orig_W / 512) * 512))
        raw = cv2.resize(raw, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
    else:
        raw = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_NEAREST)

    road_mask = np.where(raw >= 128, 1.0, 0.0).astype(np.float32)
    H, W      = road_mask.shape
    print(f"📐 Mask size: {orig_H}×{orig_W} → resized to {H}×{W}")

    # ── Load or auto-detect hole mask ──────────────────────────────────────────
    if hole_mask_path is not None:
        hm_raw  = cv2.imread(hole_mask_path, cv2.IMREAD_GRAYSCALE)
        hm_raw  = cv2.resize(hm_raw, (W, H), interpolation=cv2.INTER_NEAREST)
        # Convention: 255=valid, 0=hole  →  normalise to {0,1}
        hole_mask = np.where(hm_raw >= 128, 1.0, 0.0).astype(np.float32)
        print(f"🗺️  Hole mask loaded: {hole_mask_path}")
    else:
        hole_mask = auto_detect_holes(road_mask, min_frac=min_hole_frac)

    # Apply hole to road mask
    corrupted = road_mask * hole_mask   # (H, W) float32

    # ── Run inference ──────────────────────────────────────────────────────────
    if use_tiling and (H > 512 or W > 512):
        filled = infer_tiled(model, corrupted, hole_mask, device,
                             tile_size=tile_size, overlap=tile_overlap,
                             threshold=threshold)
    else:
        if H != 512 or W != 512:
            corrupted = cv2.resize(corrupted, (512, 512),
                                   interpolation=cv2.INTER_NEAREST)
            hole_mask = cv2.resize(hole_mask, (512, 512),
                                   interpolation=cv2.INTER_NEAREST)
        filled = infer_single(model, corrupted, hole_mask, device,
                              threshold=threshold)

    # Resize back to original resolution if needed
    if filled.shape != (orig_H, orig_W):
        filled = cv2.resize(filled, (orig_W, orig_H),
                            interpolation=cv2.INTER_NEAREST)

    # ── Save output ────────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(mask_path)[0]
        output_path = f"{base}_inpainted.png"

    out_png = (filled * 255).astype(np.uint8)
    cv2.imwrite(output_path, out_png)
    print(f"💾 Filled mask saved: {output_path}")

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Training-time 8-Panel Inpainting Visualiser
# ─────────────────────────────────────────────────────────────────────────────

def _black_panel_inp(H: int, W: int, msg: str, ax) -> None:
    """Fill *ax* with a black panel and centred white text."""
    ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    ax.text(W / 2, H / 2, msg, color='white', fontsize=8,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))
    ax.axis('off')


def visualise_inpainting(
        original:      np.ndarray,
        corrupted:     np.ndarray,
        prediction:    np.ndarray,
        ground_truth:  np.ndarray,
        hole_iou:      float,
        full_iou:      float,
        save_path:     str,
        satellite_rgb: np.ndarray = None,
        connectivity:  float = 0.0) -> None:
    """
    Save an 8-panel inpainting diagnostic figure.

    Panels:
        1. Original Mask            5. Error Map
        2. Corrupted Mask           6. Width Heatmap  [Tier 1 – M1]
        3. Inpainted Prediction     7. Surface Overlay[Tier 1 – M2]
        4. Ground Truth             8. Route Overlay  [Tier 1 – M3]

    Args:
        original      : (H, W) float32 {0,1} complete original mask.
        corrupted     : (H, W) float32 {0,1} mask with holes.
        prediction    : (H, W) float32 {0,1} inpainted prediction.
        ground_truth  : (H, W) float32 {0,1} ground-truth mask.
        hole_iou      : Hole-region IoU (from training loop).
        full_iou      : Full-image IoU.
        save_path     : Output PNG path.
        satellite_rgb : optional (H, W, 3) uint8 RGB for Modules 2 & 3.
                        If None, a grey canvas is used.
        connectivity  : connectivity loss value (for title).
    """
    H, W = original.shape[:2]
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # ── Tier 1 on the INPAINTED prediction ──────────────────────────────────
    mean_width_m  = 0.0
    dominant_surf = 'n/a'
    width_result  = None
    type_result   = None
    route_overlay = None

    if _TIER1_OK:
        # Convert inpainted prediction to uint8 mask (0/255)
        pred_mask_u8 = (prediction > 0.5).astype(np.uint8) * 255

        # Satellite RGB: use provided or build grey placeholder
        if satellite_rgb is not None:
            sat_u8 = satellite_rgb.astype(np.uint8)
        else:
            grey = (original * 160).astype(np.uint8)
            sat_u8 = np.stack([grey, grey, grey], axis=-1)

        # Module 1 – Width
        try:
            est          = RoadWidthEstimator()
            width_result = est.analyse(pred_mask_u8)
            if not width_result.is_empty:
                mean_width_m = width_result.summary_stats['mean_m']
        except Exception as e:
            print(f"[Tier1 WARNING] Module 1 (width): {e}")

        # Module 2 – Surface
        try:
            clf         = RoadTypeClassifier()
            type_result = clf.predict(sat_u8, pred_mask_u8,
                                      width_result=width_result)
            if type_result and not type_result.get('is_empty', True):
                dominant_surf = type_result['summary']['dominant_type']
        except Exception as e:
            print(f"[Tier1 WARNING] Module 2 (surface): {e}")

        # Module 3 – Route
        try:
            if (width_result is not None and not width_result.is_empty
                    and type_result is not None):
                src, dst = _pick_route_endpoints(width_result.skeleton)
                if src is not None and dst is not None:
                    router = RoadRouter(pred_mask_u8, width_result, type_result)
                    route  = router.find_route(src, dst, vehicle_type='car',
                                               satellite_rgb=sat_u8)
                    route_overlay = route.route_overlay_rgb
        except Exception as e:
            print(f"[Tier1 WARNING] Module 3 (router): {e}")

    # ── Error map (absolute difference) ─────────────────────────────────────
    error_map = np.abs(prediction - ground_truth)

    # ── 8-panel figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 8, figsize=(32, 4))
    fig.patch.set_facecolor('#1a1a2e')

    title = (f"Hole IoU={hole_iou:.4f}  Full IoU={full_iou:.4f}"
             f"  Conn={connectivity:.4f}"
             f"  |  Width={mean_width_m:.1f}m  Surface={dominant_surf}")
    fig.suptitle(title, fontsize=10, fontweight='bold', color='white', y=1.01)

    def _style(ax, ttl):
        ax.set_title(ttl, fontsize=8, fontweight='bold', color='#e0e0e0', pad=4)
        ax.axis('off')

    axes[0].imshow(original,   cmap='gray', vmin=0, vmax=1)
    _style(axes[0], 'Original Mask')

    axes[1].imshow(corrupted,  cmap='gray', vmin=0, vmax=1)
    _style(axes[1], 'Corrupted Mask')

    axes[2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    _style(axes[2], 'Inpainted Prediction')

    axes[3].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    _style(axes[3], 'Ground Truth')

    im4 = axes[4].imshow(error_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    _style(axes[4], 'Error Map')

    # Panel 6 – Width heatmap
    if width_result is not None and not width_result.is_empty:
        im6 = axes[5].imshow(width_result.width_heatmap_rgb)
        plt.colorbar(im6, ax=axes[5], fraction=0.046, pad=0.04, label='Width (m)')
        _style(axes[5], f'Width Heatmap  μ={mean_width_m:.1f}m')
    else:
        _black_panel_inp(H, W, 'No roads\ndetected', axes[5])
        _style(axes[5], 'Width Heatmap')

    # Panel 7 – Surface overlay
    if type_result is not None and not type_result.get('is_empty', True):
        axes[6].imshow(type_result['overlay_rgb'])
        _style(axes[6], f'Surface  [{dominant_surf}]')
    else:
        _black_panel_inp(H, W, 'Surface N/A', axes[6])
        _style(axes[6], 'Surface Overlay')

    # Panel 8 – Route overlay
    if route_overlay is not None:
        axes[7].imshow(route_overlay)
        _style(axes[7], 'Route  [car · cyan]')
    else:
        _black_panel_inp(H, W, 'No route\nfound', axes[7])
        _style(axes[7], 'Route Overlay')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 Inpainting viz → {save_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 Road Mask Inpainting Inference')
    parser.add_argument('--model',     required=True,  help='Path to inpainting model .pth')
    parser.add_argument('--mask',      required=True,  help='Path to input road mask PNG')
    parser.add_argument('--hole',      default=None,   help='Optional hole mask PNG (0=hole)')
    parser.add_argument('--output',    default=None,   help='Output path (auto if not set)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tiling',    action='store_true', help='Enable tiled inference')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap',   type=int, default=64)
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        mask_path=args.mask,
        hole_mask_path=args.hole,
        output_path=args.output,
        threshold=args.threshold,
        use_tiling=args.tiling,
        tile_size=args.tile_size,
        tile_overlap=args.overlap,
    )
