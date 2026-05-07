# =============================================================================
# road_width.py  –  Tier 1 Module 1: Road Width Estimation
# =============================================================================
#
# Computes per-pixel road width along the skeleton of a binary road mask using
# the medial axis distance transform.  Width → metres via GSD = 0.5 m/pixel.
# Each skeleton pixel is classified into a road type category based on width.
#
# Usage:
#   from road_width import RoadWidthEstimator
#   estimator = RoadWidthEstimator()
#   result    = estimator.analyse(road_mask_np)
#
# Input  : binary road mask  (H×W numpy array, dtype uint8, values 0/255)
# Output : RoadWidthResult dict — see analyse() docstring for full schema
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.morphology import medial_axis


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WidthConfig:
    """
    Configuration for road width estimation.

    Attributes:
        gsd_m_per_px      : Ground sampling distance in metres per pixel.
        min_road_pixels   : Minimum road pixels required; below this the mask
                            is treated as empty and a degenerate result is returned.
        footpath_max_m    : Upper width bound (m) for footpath/pedestrian class.
        single_lane_max_m : Upper width bound (m) for single-lane class.
        standard_max_m    : Upper width bound (m) for standard 2-lane road class.
                            Anything wider is classified as highway.
    """
    gsd_m_per_px:      float = 0.5
    min_road_pixels:   int   = 50
    footpath_max_m:    float = 3.0
    single_lane_max_m: float = 6.0
    standard_max_m:    float = 12.0


# ─────────────────────────────────────────────────────────────────────────────
# Road type constants
# ─────────────────────────────────────────────────────────────────────────────

ROAD_TYPES: List[str] = [
    'footpath',    # < 3 m
    'single_lane', # 3 – 6 m
    'standard',    # 6 – 12 m
    'highway',     # > 12 m
]

# RGB colours for the road-type heatmap overlay
_ROAD_TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    'footpath':    (255, 215,   0),   # gold
    'single_lane': (255, 140,   0),   # dark orange
    'standard':    ( 30, 144, 255),   # dodger blue
    'highway':     (220,  20,  60),   # crimson
}


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoadWidthResult:
    """
    Full output of :class:`RoadWidthEstimator.analyse`.

    Attributes:
        skeleton          : (H, W) bool array — skeletonised road centre-line.
        width_px          : (H, W) float32 array — width in pixels at skeleton
                            pixels; 0.0 elsewhere.
        width_m           : (H, W) float32 array — width in metres at skeleton
                            pixels; 0.0 elsewhere.
        road_type_map     : (H, W) str object array — road type label at skeleton
                            pixels; empty string elsewhere.
        summary_stats     : dict with keys mean_m, median_m, std_m, min_m, max_m
                            computed over all skeleton pixels.
        class_distribution: dict mapping each road type to pixel count on skeleton.
        width_heatmap_rgb : (H, W, 3) uint8 RGB array — plasma colourmap on
                            skeleton pixels; black background.
        is_empty          : True when the mask had too few road pixels to analyse.
    """
    skeleton:           np.ndarray
    width_px:           np.ndarray
    width_m:            np.ndarray
    road_type_map:      np.ndarray
    summary_stats:      Dict[str, float]
    class_distribution: Dict[str, int]
    width_heatmap_rgb:  np.ndarray
    is_empty:           bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Main Estimator
# ─────────────────────────────────────────────────────────────────────────────

class RoadWidthEstimator:
    """
    Estimates road width at every skeleton pixel of a binary road mask.

    The medial axis transform is used to obtain both the topological skeleton
    and the distance transform simultaneously.  Width at each skeleton pixel
    equals 2 × distance value (radius → diameter).

    Args:
        config : :class:`WidthConfig` instance.  Defaults to standard settings.
    """

    def __init__(self, config: WidthConfig = None) -> None:
        self.cfg = config or WidthConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse(self, road_mask: np.ndarray) -> RoadWidthResult:
        """
        Compute road width, road type, and summary statistics from a binary mask.

        Args:
            road_mask : (H, W) numpy array, dtype uint8, values 0 or 255.
                        Any non-zero value is treated as road.

        Returns:
            :class:`RoadWidthResult` containing all derived arrays and statistics.

        Raises:
            ValueError : If road_mask is not 2-D.
        """
        if road_mask.ndim != 2:
            raise ValueError(
                f"road_mask must be 2-D, got shape {road_mask.shape}")

        H, W = road_mask.shape
        binary = (road_mask > 0)

        # ── Edge case: empty or near-empty mask ──────────────────────────────
        if binary.sum() < self.cfg.min_road_pixels:
            return self._empty_result(H, W)

        # ── Medial axis + distance transform ─────────────────────────────────
        skeleton, dist = medial_axis(binary, return_distance=True)
        # skeleton : bool (H, W)
        # dist     : float64 (H, W) — distance from skeleton pixel to mask edge

        # Width = diameter = 2 × radius (in pixels)
        width_px = np.zeros((H, W), dtype=np.float32)
        width_px[skeleton] = (2.0 * dist[skeleton]).astype(np.float32)

        # Convert to metres
        width_m = width_px * self.cfg.gsd_m_per_px

        # ── Road type classification ──────────────────────────────────────────
        road_type_map = self._classify(skeleton, width_m)

        # ── Summary statistics (over skeleton pixels only) ────────────────────
        skel_widths_m = width_m[skeleton]
        if skel_widths_m.size > 0:
            summary_stats = {
                'mean_m':   float(np.mean(skel_widths_m)),
                'median_m': float(np.median(skel_widths_m)),
                'std_m':    float(np.std(skel_widths_m)),
                'min_m':    float(np.min(skel_widths_m)),
                'max_m':    float(np.max(skel_widths_m)),
            }
        else:
            summary_stats = {k: 0.0 for k in
                             ('mean_m', 'median_m', 'std_m', 'min_m', 'max_m')}

        # ── Class distribution ────────────────────────────────────────────────
        class_distribution = self._class_distribution(road_type_map, skeleton)

        # ── Colour heatmap ────────────────────────────────────────────────────
        heatmap_rgb = self._build_heatmap(skeleton, width_m)

        return RoadWidthResult(
            skeleton=skeleton,
            width_px=width_px,
            width_m=width_m,
            road_type_map=road_type_map,
            summary_stats=summary_stats,
            class_distribution=class_distribution,
            width_heatmap_rgb=heatmap_rgb,
            is_empty=False,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify(self,
                  skeleton:  np.ndarray,
                  width_m:   np.ndarray) -> np.ndarray:
        """
        Assign a road-type label to each skeleton pixel based on its width in metres.

        Args:
            skeleton : (H, W) bool
            width_m  : (H, W) float32

        Returns:
            road_type_map : (H, W) object array of str labels
        """
        H, W = skeleton.shape
        road_type_map = np.full((H, W), '', dtype=object)

        rows, cols = np.where(skeleton)
        cfg = self.cfg

        for r, c in zip(rows, cols):
            w = width_m[r, c]
            if w < cfg.footpath_max_m:
                label = 'footpath'
            elif w < cfg.single_lane_max_m:
                label = 'single_lane'
            elif w < cfg.standard_max_m:
                label = 'standard'
            else:
                label = 'highway'
            road_type_map[r, c] = label

        return road_type_map

    def _class_distribution(self,
                             road_type_map: np.ndarray,
                             skeleton:      np.ndarray) -> Dict[str, int]:
        """
        Count skeleton pixels per road type category.

        Args:
            road_type_map : (H, W) object array of str labels
            skeleton      : (H, W) bool

        Returns:
            dict mapping road type label to pixel count
        """
        dist: Dict[str, int] = {t: 0 for t in ROAD_TYPES}
        labels = road_type_map[skeleton]
        for label in labels:
            if label in dist:
                dist[label] += 1
        return dist

    def _build_heatmap(self,
                       skeleton: np.ndarray,
                       width_m:  np.ndarray) -> np.ndarray:
        """
        Render a plasma-colourmap heatmap of road widths on a black background.

        Args:
            skeleton : (H, W) bool
            width_m  : (H, W) float32

        Returns:
            (H, W, 3) uint8 RGB image
        """
        H, W = skeleton.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        if not skeleton.any():
            return rgb

        skel_vals = width_m[skeleton]
        v_min, v_max = skel_vals.min(), skel_vals.max()
        if v_min == v_max:
            v_max = v_min + 1.0  # avoid division by zero

        norm    = Normalize(vmin=v_min, vmax=v_max)
        cmap    = plt.get_cmap('plasma')

        # Vectorised colour lookup
        normalised = norm(skel_vals)          # (N,) float in [0, 1]
        colours    = cmap(normalised)         # (N, 4) RGBA float
        colours_u8 = (colours[:, :3] * 255).astype(np.uint8)

        rows, cols = np.where(skeleton)
        rgb[rows, cols] = colours_u8

        return rgb

    @staticmethod
    def _empty_result(H: int, W: int) -> RoadWidthResult:
        """Return a degenerate result for an empty/near-empty mask."""
        return RoadWidthResult(
            skeleton=np.zeros((H, W), dtype=bool),
            width_px=np.zeros((H, W), dtype=np.float32),
            width_m=np.zeros((H, W), dtype=np.float32),
            road_type_map=np.full((H, W), '', dtype=object),
            summary_stats={k: 0.0 for k in
                           ('mean_m', 'median_m', 'std_m', 'min_m', 'max_m')},
            class_distribution={t: 0 for t in ROAD_TYPES},
            width_heatmap_rgb=np.zeros((H, W, 3), dtype=np.uint8),
            is_empty=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (module-level)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_road_widths(road_mask: np.ndarray,
                         config:    WidthConfig = None) -> RoadWidthResult:
    """
    Module-level convenience wrapper around :class:`RoadWidthEstimator`.

    Args:
        road_mask : (H, W) uint8 binary road mask (0/255).
        config    : optional :class:`WidthConfig`.

    Returns:
        :class:`RoadWidthResult`
    """
    return RoadWidthEstimator(config).analyse(road_mask)
