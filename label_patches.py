# =============================================================================
# label_patches.py  –  Interactive Road Surface Patch Labelling Tool
# =============================================================================
#
# Extracts 16×16 patches from road skeleton pixels across validation images
# and lets you manually label each one as paved / unpaved / damaged.
#
# Controls
# --------
#   P  or  1  →  paved
#   U  or  2  →  unpaved
#   D  or  3  →  damaged
#   S        →  skip this patch (no label saved)
#   Q        →  quit and save progress immediately
#
# The labels + features are saved to patch_features.csv so you can quit at any
# point and resume later (new labels are appended).
#
# Usage
# -----
#   python label_patches.py
#   python label_patches.py --image-dir datasets/valid --n-per-image 10 --output patch_features.csv
#   python label_patches.py --resume   # append to existing CSV
#
# After collecting ~200 labels run:
#   python train_surface_rf.py
# =============================================================================

from __future__ import annotations

import argparse
import csv
import os
import glob
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
# These are the same feature helpers used by the live classifier.
from road_type_classifier import (
    ClassifierConfig,
    _extract_patch_features,
    _sample_skeleton_points,
    _extract_patches,
    RoadTypeClassifier,
)

# ── Constants ──────────────────────────────────────────────────────────────────
LABELS      = ['paved', 'unpaved', 'damaged']
KEY_MAP     = {
    ord('p'): 'paved',   ord('P'): 'paved',   ord('1'): 'paved',
    ord('u'): 'unpaved', ord('U'): 'unpaved', ord('2'): 'unpaved',
    ord('d'): 'damaged', ord('D'): 'damaged', ord('3'): 'damaged',
    ord('s'): 'skip',    ord('S'): 'skip',
    ord('q'): 'quit',    ord('Q'): 'quit',
}
LABEL_COLORS = {
    'paved':   (0, 200, 0),
    'unpaved': (0, 140, 255),
    'damaged': (60, 20, 220),
}
DISPLAY_SIZE = 200   # px — the 16×16 patch is scaled up for visibility


# =============================================================================
# Helpers
# =============================================================================

def _load_image_rgb(path: str) -> np.ndarray | None:
    """Load image as RGB uint8 (H, W, 3)."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _get_road_mask_via_model(rgb: np.ndarray) -> np.ndarray:
    """
    Try to run the trained road segmentation model to get a road mask.
    Falls back to a simple colour-based heuristic if the model isn't available.
    Returns (H, W) uint8 mask with values 0/255.
    """
    try:
        import torch
        from models import get_road_model
        from dataset import val_transform

        # Resolve weight path (same logic as app.py)
        for candidate in ['best_model.pth',
                           'road_model_best.pth',
                           'Best path/road_model_best.pth',
                           'Road Training Checkpoints/road_model_best.pth']:
            if os.path.isfile(candidate):
                weight_path = candidate
                break
        else:
            raise FileNotFoundError("No road model weights found.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = get_road_model().to(device)
        state  = torch.load(weight_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        elif isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()

        tensor = val_transform(image=rgb)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        mask = (prob > 0.40).astype(np.uint8) * 255
        print(f"    [model] road_px={int((mask>0).sum())}")
        return mask

    except Exception as e:
        print(f"    [model] unavailable ({e}) — using colour heuristic")

    # ── Colour heuristic fallback ──────────────────────────────────────────────
    # Road pixels are grey: low saturation, mid-range value.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s   = hsv[:, :, 1].astype(np.float32) / 255.0
    v   = hsv[:, :, 2].astype(np.float32) / 255.0
    road_mask = ((s < 0.30) & (v > 0.25) & (v < 0.85)).astype(np.uint8) * 255
    # Morphological cleanup
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  k, iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k, iterations=2)
    print(f"    [heuristic] road_px={int((road_mask>0).sum())}")
    return road_mask


def _make_display(patch_rgb: np.ndarray, label: str | None,
                  count: int, total: int) -> np.ndarray:
    """
    Build a display panel for the labelling UI.
    Returns a (400, DISPLAY_SIZE+10, 3) BGR image.
    """
    # Scale up the tiny patch
    big = cv2.resize(
        cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR),
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    # Instruction panel below
    panel_h = 160
    panel   = np.zeros((panel_h, DISPLAY_SIZE, 3), dtype=np.uint8)

    lines = [
        f"Patch {count} / {total}",
        "",
        "P / 1  =  paved",
        "U / 2  =  unpaved",
        "D / 3  =  damaged",
        "S      =  skip",
        "Q      =  save & quit",
    ]
    if label is not None:
        col = LABEL_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(big, (0, 0), (DISPLAY_SIZE-1, DISPLAY_SIZE-1),
                      col, thickness=8)

    for i, line in enumerate(lines):
        colour = (200, 200, 200)
        if i == 0:
            colour = (255, 255, 100)
        cv2.putText(panel, line, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)

    return np.vstack([big, panel])


def _existing_count(csv_path: str) -> int:
    """Return the number of labelled rows already in csv_path."""
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, newline='') as f:
        return max(0, sum(1 for _ in f) - 1)   # subtract header


# =============================================================================
# Main labelling loop
# =============================================================================

def label_patches(
    image_dir:    str  = 'datasets/valid',
    output_csv:   str  = 'patch_features.csv',
    n_per_image:  int  = 10,
    target_total: int  = 250,
    patch_size:   int  = 16,
    random_seed:  int  = 42,
    resume:       bool = True,
) -> None:
    """
    Interactive labelling loop.

    Args:
        image_dir    : directory containing *_sat.jpg satellite images
        output_csv   : path where features + labels will be saved
        n_per_image  : max skeleton patches to show per image
        target_total : stop after this many labels have been collected
        patch_size   : side of the square patch in pixels (must match classifier)
        random_seed  : for reproducible patch sampling
        resume       : if True, append to existing CSV; if False, overwrite
    """
    cfg = ClassifierConfig(patch_size=patch_size, n_samples=n_per_image)
    rng = np.random.default_rng(random_seed)

    # ── Discover images ────────────────────────────────────────────────────────
    patterns = [
        os.path.join(image_dir, '*_sat.jpg'),
        os.path.join(image_dir, '*_sat.png'),
        os.path.join(image_dir, '*.jpg'),
        os.path.join(image_dir, '*.png'),
    ]
    all_images: List[str] = []
    for pat in patterns:
        all_images.extend(glob.glob(pat))
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"❌  No images found in '{image_dir}'")
        sys.exit(1)

    random.seed(random_seed)
    random.shuffle(all_images)
    print(f"Found {len(all_images)} images in '{image_dir}'")

    # ── CSV header ────────────────────────────────────────────────────────────
    n_features = 47   # must match _extract_patch_features output length
    header     = [f'f{i}' for i in range(n_features)] + ['label']

    already_labelled = _existing_count(output_csv) if resume else 0
    mode = 'a' if (resume and os.path.isfile(output_csv)) else 'w'

    csv_file = open(output_csv, mode, newline='')
    writer   = csv.writer(csv_file)
    if mode == 'w' or already_labelled == 0:
        writer.writerow(header)

    print(f"{'Resuming' if already_labelled else 'Starting'} — "
          f"{already_labelled} patches already labelled. "
          f"Target: {target_total}.")

    total_labelled = already_labelled
    window_name    = 'Label Road Patches  [P=paved  U=unpaved  D=damaged  S=skip  Q=quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_SIZE + 20, DISPLAY_SIZE + 180)

    # ── Per-image loop ─────────────────────────────────────────────────────────
    quit_flag = False
    for img_path in all_images:
        if quit_flag or total_labelled >= target_total:
            break

        print(f"\n[{total_labelled}/{target_total}]  {os.path.basename(img_path)}")

        rgb = _load_image_rgb(img_path)
        if rgb is None:
            print("  ⚠  Could not load — skipping")
            continue

        # Resize to 512×512 to match model output space
        rgb = cv2.resize(rgb, (512, 512))

        # Get road mask
        road_mask = _get_road_mask_via_model(rgb)
        if (road_mask > 0).sum() < 100:
            print("  ⚠  Too few road pixels — skipping")
            continue

        # Extract skeleton patches
        clf            = RoadTypeClassifier(cfg)
        skeleton       = clf._get_skeleton(road_mask)
        points_raw     = _sample_skeleton_points(skeleton, cfg.n_samples, rng)
        patches, pts   = _extract_patches(rgb, points_raw, patch_size)

        if patches.shape[0] == 0:
            print("  ⚠  No valid patches — skipping")
            continue

        print(f"  {patches.shape[0]} patches to label")

        # ── Per-patch loop ────────────────────────────────────────────────────
        for patch, pt in zip(patches, pts):
            if quit_flag or total_labelled >= target_total:
                break

            feat    = _extract_patch_features(patch, cfg)
            display = _make_display(patch, None,
                                    total_labelled + 1, target_total)
            cv2.imshow(window_name, display)

            label = None
            while label is None:
                key = cv2.waitKey(0) & 0xFF
                action = KEY_MAP.get(key)
                if action == 'quit':
                    quit_flag = True
                    break
                elif action == 'skip':
                    break
                elif action in LABELS:
                    label = action

            if quit_flag:
                break
            if label is None:
                continue   # skipped

            # Flash confirmation colour
            display = _make_display(patch, label,
                                    total_labelled + 1, target_total)
            cv2.imshow(window_name, display)
            cv2.waitKey(300)

            # Write to CSV
            row = list(feat) + [label]
            writer.writerow(row)
            csv_file.flush()
            total_labelled += 1
            print(f"    [{total_labelled:>3}] {label:8s}  row={pt[0]}  col={pt[1]}")

    cv2.destroyAllWindows()
    csv_file.close()

    print(f"\n✅  Done.  {total_labelled} patches labelled → '{output_csv}'")
    if total_labelled >= target_total:
        print(f"🎯  Target of {target_total} reached! Run:  python train_surface_rf.py")
    else:
        remaining = target_total - total_labelled
        print(f"ℹ️   {remaining} more patches needed.  Run again to continue.")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive road surface patch labeller')
    parser.add_argument('--image-dir',    default='datasets/valid',
                        help='Directory of satellite images (default: datasets/valid)')
    parser.add_argument('--output',       default='patch_features.csv',
                        help='Output CSV path (default: patch_features.csv)')
    parser.add_argument('--n-per-image',  type=int, default=10,
                        help='Max patches per image (default: 10)')
    parser.add_argument('--target',       type=int, default=250,
                        help='Total patches to collect (default: 250)')
    parser.add_argument('--patch-size',   type=int, default=16,
                        help='Patch side in pixels (default: 16)')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--no-resume',    action='store_true',
                        help='Overwrite existing CSV instead of appending')
    args = parser.parse_args()

    label_patches(
        image_dir    = args.image_dir,
        output_csv   = args.output,
        n_per_image  = args.n_per_image,
        target_total = args.target,
        patch_size   = args.patch_size,
        random_seed  = args.seed,
        resume       = not args.no_resume,
    )
