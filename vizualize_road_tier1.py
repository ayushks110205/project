# =============================================================================
# vizualize_road_tier1.py  –  5-Panel Tier 1 Road Intelligence Visualiser
# =============================================================================
#
# Extended visualisation panel layout:
#
#   Panel 1 │ Original Satellite Image
#   Panel 2 │ Binary Road Mask  (Stage 1 output)
#   Panel 3 │ Width Heatmap     (plasma colormap on skeleton, Module 1)
#   Panel 4 │ Surface Type Overlay  (green=paved, orange=unpaved, red=damaged)
#   Panel 5 │ Combined Road Intelligence Map
#            │   Skeleton coloured by road type + surface overlaid on satellite
#
# Usage:
#   from vizualize_road_tier1 import visualise_tier1, save_tier1_figure
#   save_tier1_figure(image_np, road_mask_np, tier1_result, save_path)
#
# All inputs are numpy arrays — no file I/O inside this module.
# =============================================================================

from __future__ import annotations

import os
import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.ndimage import grey_dilation


# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes (must match road_width.py and road_type_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

_ROAD_TYPE_COLORS = {
    'footpath':    np.array([255, 215,   0], dtype=np.uint8),  # gold
    'single_lane': np.array([255, 140,   0], dtype=np.uint8),  # dark orange
    'standard':    np.array([ 30, 144, 255], dtype=np.uint8),  # dodger blue
    'highway':     np.array([220,  20,  60], dtype=np.uint8),  # crimson
}

_SURFACE_COLORS = {
    'paved':   np.array([  0, 200,   0], dtype=np.uint8),  # green
    'unpaved': np.array([255, 140,   0], dtype=np.uint8),  # orange
    'damaged': np.array([220,  20,  60], dtype=np.uint8),  # crimson
}


# ─────────────────────────────────────────────────────────────────────────────
# Panel builder helpers
# ─────────────────────────────────────────────────────────────────────────────

_SKEL_DILATION = 1   # pixels to grow skeleton lines (1 → 3×3 footprint)


def _thicken_rgb_lines(rgb_img: np.ndarray,
                        iterations: int = _SKEL_DILATION) -> np.ndarray:
    """
    Dilate non-black skeleton lines in an RGB image so they render
    visibly even on small screens.  Uses per-channel grey dilation
    (max pooling) which preserves the original hue of each line.

    Args:
        rgb_img    : (H, W, 3) uint8 image with coloured skeleton on black bg.
        iterations : dilation radius in pixels (1 → 3×3 kernel).

    Returns:
        (H, W, 3) uint8 image with thicker lines.
    """
    size = 2 * iterations + 1  # kernel side length
    out  = np.zeros_like(rgb_img)
    for ch in range(3):
        out[:, :, ch] = grey_dilation(rgb_img[:, :, ch], size=(size, size))
    return out

def _build_combined_map(image_rgb:   np.ndarray,
                         tier1:       dict,
                         alpha_road:  float = 0.7,
                         alpha_surf:  float = 0.5) -> np.ndarray:
    """
    Build the combined road intelligence map (Panel 5).

    Layers (bottom → top):
        1. Satellite image (darkened as base)
        2. Road-type colour on skeleton pixels  (alpha = alpha_road)
        3. Surface-type colour on skeleton pixels blended on top (alpha = alpha_surf)

    Args:
        image_rgb  : (H, W, 3) float32 in [0, 1] or uint8.
        tier1      : dict returned by :meth:`SatellitePipeline.run_tier1`.
        alpha_road : opacity of road-type colour layer.
        alpha_surf : opacity of surface-type colour layer.

    Returns:
        (H, W, 3) uint8 RGB composite.
    """
    if image_rgb.dtype != np.uint8:
        base = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
    else:
        base = image_rgb.copy()

    # Darken background for contrast
    canvas = (base * 0.45).astype(np.float32)

    width_result = tier1.get('width_result')
    type_result  = tier1.get('type_result')

    if width_result is None or type_result is None:
        return canvas.astype(np.uint8)

    skeleton      = width_result.skeleton if hasattr(width_result, 'skeleton') \
                    else width_result['skeleton']
    road_type_map = width_result.road_type_map if hasattr(width_result, 'road_type_map') \
                    else width_result['road_type_map']
    surface_map   = type_result['surface_map']

    rows, cols = np.where(skeleton)

    # Paint blended colours onto a separate lines layer so we can dilate
    # afterwards without expanding the dark background.
    lines_layer = np.zeros_like(canvas)   # float32, starts transparent

    for r, c in zip(rows, cols):
        rt   = str(road_type_map[r, c])
        surf = str(surface_map[r, c])

        rt_col   = _ROAD_TYPE_COLORS.get(rt,   np.array([200, 200, 200], dtype=np.uint8))
        surf_col = _SURFACE_COLORS.get(surf, np.array([200, 200, 200], dtype=np.uint8))

        # Blend road-type colour, then surface colour on top
        blended = rt_col.astype(np.float32) * alpha_road
        blended = blended * (1 - alpha_surf) + surf_col.astype(np.float32) * alpha_surf
        lines_layer[r, c] = np.clip(blended, 0, 255)

    # Thicken skeleton lines (grey dilation per channel)
    size = 2 * _SKEL_DILATION + 1
    for ch in range(3):
        lines_layer[:, :, ch] = grey_dilation(
            lines_layer[:, :, ch].astype(np.uint8), size=(size, size))

    # Composite: where a line exists, overwrite the dark canvas
    has_line = lines_layer.max(axis=2) > 0
    canvas[has_line] = lines_layer[has_line]

    return canvas.astype(np.uint8)


def _make_legend_patches(color_map: dict, prefix: str = '') -> list:
    """
    Build a list of matplotlib legend patches from a colour dict.

    Args:
        color_map : dict mapping label string → (3,) uint8 RGB array.
        prefix    : optional text prefix for each label.

    Returns:
        list of :class:`matplotlib.patches.Patch`.
    """
    patches = []
    for label, rgb in color_map.items():
        colour_f = tuple(rgb.astype(float) / 255.0)
        patches.append(mpatches.Patch(
            color=colour_f, label=f"{prefix}{label.replace('_', ' ').title()}"))
    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation function
# ─────────────────────────────────────────────────────────────────────────────

def save_tier1_figure(image_rgb:   np.ndarray,
                       road_mask:   np.ndarray,
                       tier1:       dict,
                       save_path:   str,
                       title:       str = '',
                       dpi:         int = 150) -> None:
    """
    Render and save the 5-panel Tier 1 road intelligence visualisation.

    Panel layout:
        1. Satellite Image          — raw RGB input
        2. Binary Road Mask         — Stage 1 extraction (grayscale)
        3. Width Heatmap            — plasma colormap on skeleton, black bg
        4. Surface Type Overlay     — skeleton coloured by surface type
        5. Combined Intelligence    — blended road-type + surface on satellite

    Args:
        image_rgb : (H, W, 3) float32 [0,1] or uint8 [0,255] satellite image.
        road_mask : (H, W) uint8 binary road mask (0/255).
        tier1     : dict from :meth:`SatellitePipeline.run_tier1`.
        save_path : absolute path to save the figure PNG.
        title     : optional figure suptitle string.
        dpi       : figure resolution.
    """
    # Normalise image to float [0, 1] for display
    if image_rgb.dtype == np.uint8:
        img_f32 = image_rgb.astype(np.float32) / 255.0
    else:
        img_f32 = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    width_result = tier1.get('width_result')
    type_result  = tier1.get('type_result')

    # ── Panel data ────────────────────────────────────────────────────────────
    road_binary = (road_mask > 0).astype(np.float32)

    # Panel 3: width heatmap
    if width_result is not None:
        heatmap = (width_result.width_heatmap_rgb
                   if hasattr(width_result, 'width_heatmap_rgb')
                   else width_result.get('width_heatmap_rgb',
                        np.zeros((*road_mask.shape, 3), dtype=np.uint8)))
    else:
        heatmap = np.zeros((*road_mask.shape, 3), dtype=np.uint8)

    # Panel 4: surface overlay
    if type_result is not None:
        surface_overlay = type_result.get(
            'overlay_rgb', np.zeros((*road_mask.shape, 3), dtype=np.uint8))
    else:
        surface_overlay = np.zeros((*road_mask.shape, 3), dtype=np.uint8)

    # Panel 5: combined map
    combined = _build_combined_map(img_f32, tier1)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(35, 7))
    fig.patch.set_facecolor('#1a1a2e')

    suptitle = title or 'Tier 1 Road Intelligence Layer'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', color='white', y=1.01)

    panel_style = dict(fontsize=10, fontweight='bold', color='white', pad=6)

    # ── Panel 1: Satellite ────────────────────────────────────────────────────
    axes[0].imshow(img_f32)
    axes[0].set_title('Satellite Image', **panel_style)
    axes[0].axis('off')

    # ── Panel 2: Binary Mask ──────────────────────────────────────────────────
    axes[1].imshow(road_binary, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Binary Road Mask\n(Stage 1)', **panel_style)
    axes[1].axis('off')

    # ── Panel 3: Width Heatmap ────────────────────────────────────────────────
    axes[2].imshow(_thicken_rgb_lines(heatmap))
    axes[2].set_title('Road Width Heatmap\n(plasma, skeleton only)', **panel_style)
    axes[2].axis('off')
    # Colourbar
    if width_result is not None:
        skel = (width_result.skeleton if hasattr(width_result, 'skeleton')
                else width_result.get('skeleton'))
        wm   = (width_result.width_m  if hasattr(width_result, 'width_m')
                else width_result.get('width_m'))
        if skel is not None and wm is not None and skel.any():
            vals = wm[skel]
            sm   = plt.cm.ScalarMappable(
                cmap='plasma', norm=Normalize(vmin=vals.min(), vmax=vals.max()))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
            cb.set_label('Width (m)', color='white', fontsize=8)
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # ── Panel 4: Surface Type Overlay ─────────────────────────────────────────
    axes[3].imshow(_thicken_rgb_lines(surface_overlay))
    # NOTE: KMeans on arid/semi-arid imagery tends to misclassify
    #       bare-soil road edges as 'damaged'.  Flag this as a known
    #       classifier limitation when reviewing results.
    axes[3].set_title('Road Surface Type\n(KMeans ⚠ arid-terrain bias)', **panel_style)
    axes[3].axis('off')
    surf_patches = _make_legend_patches(_SURFACE_COLORS, prefix='')
    axes[3].legend(handles=surf_patches, loc='lower left',
                   fontsize=7, framealpha=0.6,
                   facecolor='#1a1a2e', labelcolor='white')

    # ── Panel 5: Combined Intelligence Map ────────────────────────────────────
    axes[4].imshow(combined)
    axes[4].set_title('Combined Road Intelligence\n(type + surface blended)', **panel_style)
    axes[4].axis('off')
    rt_patches   = _make_legend_patches(_ROAD_TYPE_COLORS, prefix='Width: ')
    axes[4].legend(handles=rt_patches, loc='lower left',
                   fontsize=7, framealpha=0.6,
                   facecolor='#1a1a2e', labelcolor='white')

    for ax in axes:
        ax.set_facecolor('#1a1a2e')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"🖼️  Tier 1 figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full run + visualise (when used standalone)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_tier1(image_rgb:   np.ndarray,
                     road_mask:   np.ndarray,
                     save_dir:    str = '/kaggle/working/results/tier1',
                     stem:        str = 'road') -> str:
    """
    Run the full Tier 1 pipeline on a single image and save the 5-panel figure.

    This function is self-contained — it imports and runs Module 1 and Module 2
    internally.  Use :meth:`SatellitePipeline.run_tier1` for integrated usage.

    Args:
        image_rgb : (H, W, 3) uint8 RGB satellite image.
        road_mask : (H, W) uint8 binary road mask (0/255).
        save_dir  : output directory.
        stem      : filename stem for the saved PNG.

    Returns:
        Absolute path to the saved figure PNG.
    """
    from road_width import RoadWidthEstimator
    from road_type_classifier import RoadTypeClassifier

    width_est  = RoadWidthEstimator()
    width_res  = width_est.analyse(road_mask)

    clf        = RoadTypeClassifier()
    type_res   = clf.predict(image_rgb, road_mask, width_result=width_res)

    tier1 = {
        'width_result': width_res,
        'type_result':  type_res,
    }

    os.makedirs(save_dir, exist_ok=True)
    ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"{stem}_tier1_{ts}.png")
    save_tier1_figure(image_rgb, road_mask, tier1, save_path,
                      title=f'Tier 1 Road Intelligence  |  {stem}')
    return save_path
