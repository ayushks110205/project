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
from torch.amp import autocast

from inpainting_model import get_inpainting_model

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
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

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
