# =============================================================================
# road_router.py  –  Tier 1 Module 3: Vehicle-Aware Road Routing
# =============================================================================
#
# Builds a cost surface from the road skeleton and finds the least-cost path
# between two pixel coordinates using skimage.graph.MCP_Geometric.
#
# Cost at each skeleton pixel combines:
#   • Base traversal cost            (constant 1.0)
#   • Width penalty                  (wider roads → lower cost)
#   • Surface penalty                (paved=1.0, unpaved=2.5, damaged=5.0)
#
# Vehicle type constraints:
#   • pedestrian : no width requirement; all surfaces passable
#   • motorcycle : ≥ 1.5 m; all surfaces passable
#   • car        : ≥ 3 m;   surface penalty ×1.5 on unpaved
#   • truck      : ≥ 6 m;   surface penalty ×2.0 on unpaved; refuses damaged
#
# Usage:
#   from road_router import RoadRouter
#   router = RoadRouter(road_mask, width_result, type_result)
#   result = router.find_route(src=(r0, c0), dst=(r1, c1), vehicle_type='car')
#
# All arrays are pure numpy — no torch, no file I/O.
# =============================================================================

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """
    Configuration for :class:`RoadRouter`.

    Attributes:
        gsd_m_per_px         : metres per pixel (matches pipeline GSD).
        base_cost            : constant traversal cost per skeleton pixel.
        off_road_cost        : cost for non-skeleton pixels (acts as barrier).
        width_cost_scale     : divisor that maps width_m to a cost reduction.
                               cost_width = 1 / (1 + width_m / width_cost_scale).
                               Larger scale → width matters more.
        surface_costs        : base cost multiplier per surface type.
    """
    gsd_m_per_px:    float = 0.5
    base_cost:       float = 1.0
    off_road_cost:   float = 1e6
    width_cost_scale: float = 12.0
    surface_costs:   Dict[str, float] = field(default_factory=lambda: {
        'paved':   1.0,
        'unpaved': 2.5,
        'damaged': 5.0,
        '':        1.5,   # unknown / unlabelled skeleton pixel
    })


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleProfile:
    """
    Encodes routing constraints and surface penalties for a vehicle type.

    Attributes:
        name                 : Human-readable name.
        min_width_m          : Minimum road width in metres; skeleton pixels
                               narrower than this are treated as impassable.
        surface_penalty_mult : Additional multiplier on surface cost
                               (applied on top of RouterConfig.surface_costs).
        refuses_damaged      : If True, damaged pixels are treated as barriers.
    """
    name:                 str
    min_width_m:          float
    surface_penalty_mult: float = 1.0
    refuses_damaged:      bool  = False


_VEHICLE_PROFILES: Dict[str, VehicleProfile] = {
    'pedestrian': VehicleProfile('pedestrian', min_width_m=0.0),
    'motorcycle': VehicleProfile('motorcycle', min_width_m=1.5),
    'car':        VehicleProfile('car',        min_width_m=3.0,
                                 surface_penalty_mult=1.5),
    'truck':      VehicleProfile('truck',      min_width_m=6.0,
                                 surface_penalty_mult=2.0,
                                 refuses_damaged=True),
}

VEHICLE_TYPES: List[str] = list(_VEHICLE_PROFILES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Route result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """
    Output of :meth:`RoadRouter.find_route`.

    Attributes:
        found             : True if a valid route was found.
        path              : list of (row, col) pixel coordinates along the route.
        total_cost        : accumulated MCP cost along the path.
        distance_m        : Euclidean path length in metres
                            (sum of pixel distances × GSD).
        mean_width_m      : mean road width along the path.
        dominant_surface  : most common surface type along the path.
        route_overlay_rgb : (H, W, 3) uint8 route drawn on satellite image.
        message           : human-readable status string.
    """
    found:             bool
    path:              List[Tuple[int, int]]
    total_cost:        float
    distance_m:        float
    mean_width_m:      float
    dominant_surface:  str
    route_overlay_rgb: np.ndarray
    message:           str = ''


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint picker (module-level helper, used by visualisers)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_route_endpoints(
        skeleton: np.ndarray,
        max_sample: int = 500,
        rng_seed: int = 0,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Find the two skeleton pixels that are furthest apart (Euclidean distance).

    For large skeletons (> *max_sample* pixels), a random subsample of
    *max_sample* points is used to avoid O(N²) memory allocation.

    Args:
        skeleton   : (H, W) bool skeleton array.
        max_sample : maximum number of points to include in the distance matrix.
        rng_seed   : seed for reproducible subsampling.

    Returns:
        ``(src_rc, dst_rc)`` as ``(row, col)`` int tuples, or
        ``(None, None)`` if the skeleton has fewer than 2 pixels.
    """
    rows, cols = np.where(skeleton)
    N = len(rows)

    if N < 2:
        return None, None

    pts = np.stack([rows, cols], axis=1).astype(np.float32)  # (N, 2)

    if N > max_sample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(N, size=max_sample, replace=False)
        pts = pts[idx]

    # Pairwise distance matrix on the (possibly subsampled) points
    diff      = pts[:, None, :] - pts[None, :, :]  # (M, M, 2)
    dist_mat  = np.sqrt((diff ** 2).sum(axis=-1))  # (M, M)

    flat_idx  = int(np.argmax(dist_mat))
    i, j      = divmod(flat_idx, len(pts))

    src_rc = (int(pts[i, 0]), int(pts[i, 1]))
    dst_rc = (int(pts[j, 0]), int(pts[j, 1]))
    return src_rc, dst_rc


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class RoadRouter:
    """
    Vehicle-aware road router built on :class:`skimage.graph.MCP_Geometric`.

    Builds a cost surface from the road skeleton, width map, and surface type
    map.  Off-skeleton pixels get a very high cost so MCP effectively constrains
    routing to the road network.

    Args:
        road_mask    : (H, W) uint8 binary road mask (0/255).
        width_result : dict or RoadWidthResult from ``road_width.py``.
                       Must expose ``.skeleton``, ``.width_m``.
        type_result  : dict from ``road_type_classifier.py``.
                       Must expose ``['surface_map']``.
        config       : :class:`RouterConfig` (optional).

    Raises:
        ValueError : If the road mask is completely empty (no road pixels).
    """

    def __init__(self,
                 road_mask:    np.ndarray,
                 width_result,
                 type_result:  dict,
                 config:       RouterConfig = None) -> None:

        self.cfg   = config or RouterConfig()
        H, W       = road_mask.shape[:2]
        self.shape = (H, W)

        # Extract skeleton and width map
        if hasattr(width_result, 'skeleton'):
            self.skeleton = width_result.skeleton
            self.width_m  = width_result.width_m
        else:
            self.skeleton = width_result['skeleton']
            self.width_m  = width_result['width_m']

        # Extract surface map
        self.surface_map = type_result['surface_map']   # (H, W) str object array

        # Pre-build a base cost surface (without vehicle-specific constraints)
        self._base_cost_surface = self._build_base_cost()

    # ── Cost surface ───────────────────────────────────────────────────────────

    def _build_base_cost(self) -> np.ndarray:
        """
        Construct the base cost grid shared across all vehicle types.

        Fully vectorised — no Python-level per-pixel loop.

        Cost at skeleton pixel (r, c):
            base_cost
            × (1 / (1 + width_m / width_cost_scale))   ← width reward
            × surface_cost[surface_map(r, c)]           ← surface penalty

        Off-skeleton pixels get ``off_road_cost``.

        Returns:
            (H, W) float64 cost surface.
        """
        H, W = self.shape
        cfg  = self.cfg
        cost = np.full((H, W), cfg.off_road_cost, dtype=np.float64)

        skel = self.skeleton
        if not skel.any():
            return cost

        # ── Width factor (vectorised) ─────────────────────────────────────────
        width_factor = 1.0 / (1.0 + self.width_m[skel] / cfg.width_cost_scale)

        # ── Surface factor (vectorised lookup) ────────────────────────────────
        surface_vals = self.surface_map[skel]          # 1-D object array of str
        surface_factor = np.array(
            [cfg.surface_costs.get(str(s), 1.5) for s in surface_vals],
            dtype=np.float64)

        cost[skel] = cfg.base_cost * width_factor * surface_factor
        return cost

    def _apply_vehicle_constraints(self,
                                   profile: VehicleProfile) -> np.ndarray:
        """
        Clone the base cost surface and apply vehicle-specific constraints.

        Fully vectorised — no Python-level per-pixel loop.

        Pixels that violate minimum width requirements or surface restrictions
        for the given vehicle profile are raised to ``off_road_cost`` (impassable).

        Args:
            profile : :class:`VehicleProfile`.

        Returns:
            (H, W) float64 vehicle-adjusted cost surface.
        """
        cost = self._base_cost_surface.copy()
        skel = self.skeleton
        if not skel.any():
            return cost

        cfg = self.cfg
        w_m_skel     = self.width_m[skel].astype(np.float64)
        surface_skel = self.surface_map[skel]   # 1-D object array of str

        # Boolean masks over skeleton pixels
        too_narrow    = w_m_skel < profile.min_width_m
        is_damaged    = np.array([str(s) == 'damaged' for s in surface_skel], dtype=bool)
        is_unpaved    = np.array([str(s) == 'unpaved' for s in surface_skel], dtype=bool)

        # Impassable mask: too narrow OR (refuses damaged AND is damaged)
        blocked = too_narrow | (profile.refuses_damaged & is_damaged)

        # Apply block
        rows, cols = np.where(skel)
        cost[rows[blocked], cols[blocked]] = cfg.off_road_cost

        # Apply surface penalty multiplier on unpaved pixels that are NOT blocked
        if profile.surface_penalty_mult != 1.0:
            apply_penalty = is_unpaved & ~blocked
            cost[rows[apply_penalty], cols[apply_penalty]] *= profile.surface_penalty_mult

        return cost

    # ── Snapping helper ────────────────────────────────────────────────────────

    def _snap_to_skeleton(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Snap a pixel coordinate to the nearest skeleton pixel.

        Returns None if the skeleton is empty.

        Args:
            point : (row, col) int tuple.

        Returns:
            Nearest skeleton pixel coordinate, or None.
        """
        rows, cols = np.where(self.skeleton)
        if len(rows) == 0:
            return None
        r, c  = point
        dists = (rows - r) ** 2 + (cols - c) ** 2
        idx   = int(np.argmin(dists))
        return (int(rows[idx]), int(cols[idx]))

    # ── Public API ─────────────────────────────────────────────────────────────

    def find_route(self,
                   src:          Tuple[int, int],
                   dst:          Tuple[int, int],
                   vehicle_type: str = 'car',
                   satellite_rgb: Optional[np.ndarray] = None
                   ) -> RouteResult:
        """
        Find the least-cost path from *src* to *dst* on the road skeleton.

        Both *src* and *dst* are snapped to the nearest skeleton pixel before
        routing.  If no valid path exists (disconnected network, vehicle width
        constraint isolates regions), a ``RouteResult`` with ``found=False``
        is returned.

        Args:
            src          : (row, col) source pixel coordinate.
            dst          : (row, col) destination pixel coordinate.
            vehicle_type : one of ``'pedestrian'``, ``'motorcycle'``,
                           ``'car'``, ``'truck'``.
            satellite_rgb: optional (H, W, 3) uint8 image to draw route on.
                           If None, a blank canvas is used.

        Returns:
            :class:`RouteResult`

        Raises:
            ValueError : If *vehicle_type* is not recognised.
        """
        if vehicle_type not in _VEHICLE_PROFILES:
            raise ValueError(
                f"Unknown vehicle_type '{vehicle_type}'. "
                f"Choose from: {VEHICLE_TYPES}")

        H, W    = self.shape
        profile = _VEHICLE_PROFILES[vehicle_type]

        # ── Edge case: empty skeleton ─────────────────────────────────────────
        if not self.skeleton.any():
            return self._no_route_result(
                H, W, satellite_rgb, "Empty road skeleton — no routes possible.")

        # ── Snap endpoints to skeleton ────────────────────────────────────────
        src_snap = self._snap_to_skeleton(src)
        dst_snap = self._snap_to_skeleton(dst)

        if src_snap is None or dst_snap is None:
            return self._no_route_result(
                H, W, satellite_rgb, "Could not snap endpoints to skeleton.")

        if src_snap == dst_snap:
            return self._trivial_route(src_snap, H, W, satellite_rgb)

        # ── Build vehicle-adjusted cost surface ───────────────────────────────
        cost_surface = self._apply_vehicle_constraints(profile)

        # ── MCP routing ───────────────────────────────────────────────────────
        try:
            mcp  = MCP_Geometric(cost_surface)
            costs_arr, _ = mcp.find_costs([src_snap])
            path_rc = mcp.traceback(dst_snap)   # list of (row, col) tuples
        except Exception as exc:
            return self._no_route_result(
                H, W, satellite_rgb, f"MCP routing failed: {exc}")

        if not path_rc:
            return self._no_route_result(
                H, W, satellite_rgb,
                f"No valid route for '{vehicle_type}' between {src} → {dst}.")

        # Check if the cost at destination is still effectively infinite
        total_cost = float(costs_arr[dst_snap[0], dst_snap[1]])
        if total_cost >= self.cfg.off_road_cost * 0.5:
            return self._no_route_result(
                H, W, satellite_rgb,
                f"No connected road path for '{vehicle_type}' "
                f"between {src} and {dst}.")

        # ── Route statistics ──────────────────────────────────────────────────
        path_arr = [(int(r), int(c)) for r, c in path_rc]

        # Euclidean length in metres
        distance_m = self._path_length_m(path_arr)

        # Mean width along path
        path_widths = [float(self.width_m[r, c]) for r, c in path_arr]
        mean_width_m = float(np.mean(path_widths)) if path_widths else 0.0

        # Dominant surface (vectorised)
        path_pts     = np.array(path_arr, dtype=np.int32)
        path_surfs   = self.surface_map[path_pts[:, 0], path_pts[:, 1]]
        path_surfs   = [str(s) for s in path_surfs if str(s) != '']
        dominant_surface = Counter(path_surfs).most_common(1)[0][0] \
                           if path_surfs else 'unknown'

        # ── Overlay ───────────────────────────────────────────────────────────
        overlay = self._draw_route(path_arr, H, W, satellite_rgb, src_snap, dst_snap)

        return RouteResult(
            found=True,
            path=path_arr,
            total_cost=total_cost,
            distance_m=distance_m,
            mean_width_m=mean_width_m,
            dominant_surface=dominant_surface,
            route_overlay_rgb=overlay,
            message=(f"Route found for '{vehicle_type}': "
                     f"{len(path_arr)} px | {distance_m:.1f} m | "
                     f"{dominant_surface} surface | "
                     f"avg width {mean_width_m:.1f} m"),
        )

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw_route(self,
                    path:         List[Tuple[int, int]],
                    H:            int,
                    W:            int,
                    satellite_rgb: Optional[np.ndarray],
                    src:          Tuple[int, int],
                    dst:          Tuple[int, int]) -> np.ndarray:
        """
        Draw the route path on a copy of the satellite image (or black canvas).

        Route pixels are coloured yellow; src is green; dst is red.

        Args:
            path          : list of (row, col) tuples.
            H, W          : image dimensions.
            satellite_rgb : optional (H, W, 3) uint8 image.
            src, dst      : snapped endpoint coordinates.

        Returns:
            (H, W, 3) uint8 RGB overlay.
        """
        if satellite_rgb is not None:
            canvas = satellite_rgb.copy().astype(np.uint8)
            # Darken slightly so route stands out
            canvas = (canvas * 0.6).astype(np.uint8)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        route_color  = np.array([255, 255,   0], dtype=np.uint8)  # yellow
        src_color    = np.array([  0, 255,   0], dtype=np.uint8)  # green
        dst_color    = np.array([255,  50,  50], dtype=np.uint8)  # red

        for r, c in path:
            canvas[r, c] = route_color

        # Draw marker crosses at endpoints (3-pixel radius)
        for pt, colour in [(src, src_color), (dst, dst_color)]:
            r, c = pt
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        canvas[nr, nc] = colour

        return canvas

    def _path_length_m(self, path: List[Tuple[int, int]]) -> float:
        """
        Compute the Euclidean path length in metres.

        Consecutive pixel steps can be 1-pixel (axial) or √2 (diagonal);
        both cases are handled correctly via Euclidean distance.

        Args:
            path : list of (row, col) tuples.

        Returns:
            Total path length in metres.
        """
        if len(path) < 2:
            return 0.0
        pts = np.array(path, dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        return float(step_lengths.sum() * self.cfg.gsd_m_per_px)

    # ── No-route helpers ───────────────────────────────────────────────────────

    def _no_route_result(self,
                         H:            int,
                         W:            int,
                         satellite_rgb: Optional[np.ndarray],
                         message:       str) -> RouteResult:
        """Return a degenerate RouteResult when routing fails."""
        canvas = (satellite_rgb.copy().astype(np.uint8)
                  if satellite_rgb is not None
                  else np.zeros((H, W, 3), dtype=np.uint8))
        return RouteResult(
            found=False,
            path=[],
            total_cost=float('inf'),
            distance_m=0.0,
            mean_width_m=0.0,
            dominant_surface='unknown',
            route_overlay_rgb=canvas,
            message=message,
        )

    def _trivial_route(self,
                       pt:           Tuple[int, int],
                       H:            int,
                       W:            int,
                       satellite_rgb: Optional[np.ndarray]) -> RouteResult:
        """Return a trivial single-point route when src == dst after snapping."""
        canvas = (satellite_rgb.copy().astype(np.uint8)
                  if satellite_rgb is not None
                  else np.zeros((H, W, 3), dtype=np.uint8))
        canvas[pt[0], pt[1]] = [0, 255, 0]
        return RouteResult(
            found=True,
            path=[pt],
            total_cost=0.0,
            distance_m=0.0,
            mean_width_m=float(self.width_m[pt[0], pt[1]]),
            dominant_surface=str(self.surface_map[pt[0], pt[1]]) or 'unknown',
            route_overlay_rgb=canvas,
            message="Source and destination are the same skeleton pixel.",
        )
