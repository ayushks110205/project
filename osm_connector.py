# =============================================================================
# osm_connector.py  —  OpenStreetMap Road Network Connector
# =============================================================================
#
# Fetches road network data from OpenStreetMap (via osmnx / Overpass API)
# for any lat/lon bounding box, renders it onto a 512×512 binary mask
# that is directly compatible with the DeepGlobe pipeline's Tier 1 & Tier 2.
#
# No account, no API key, no cost — OSM is completely free.
#
# Install:
#   pip install osmnx
#
# Usage:
#   from osm_connector import OSMConnector
#   conn   = OSMConnector()
#   result = conn.fetch(lat_min=12.90, lon_min=77.50,
#                       lat_max=13.10, lon_max=77.70)
#   mask   = result['mask']   # (512,512) uint8 — plug into _run_tier1/_run_tier2
#   print(result['highway_counts'])
# =============================================================================

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Optional osmnx import ──────────────────────────────────────────────────────
_OSM_OK = False
try:
    import osmnx as ox
    _OSM_OK = True
except ImportError:
    pass


# =============================================================================
# Constants
# =============================================================================

# Highway type → pixel draw width on the 512×512 canvas.
# Wider = more prominent road class (motorway visible, footpath thin).
_HIGHWAY_WIDTH_PX: Dict[str, int] = {
    'motorway':       8,
    'motorway_link':  5,
    'trunk':          7,
    'trunk_link':     5,
    'primary':        6,
    'primary_link':   4,
    'secondary':      5,
    'secondary_link': 3,
    'tertiary':       4,
    'tertiary_link':  3,
    'unclassified':   3,
    'residential':    3,
    'living_street':  2,
    'service':        2,
    'track':          2,
    'path':           1,
    'footway':        1,
    'cycleway':       1,
    'steps':          1,
}
_DEFAULT_WIDTH_PX = 2

# OSM surface tag → pipeline surface class (paved / unpaved / unknown)
_OSM_SURFACE_MAP: Dict[str, str] = {
    'paved':          'paved',
    'asphalt':        'paved',
    'concrete':       'paved',
    'concrete:plates':'paved',
    'cobblestone':    'paved',
    'sett':           'paved',
    'paving_stones':  'paved',
    'metal':          'paved',
    'unpaved':        'unpaved',
    'gravel':         'unpaved',
    'compacted':      'unpaved',
    'fine_gravel':    'unpaved',
    'dirt':           'unpaved',
    'earth':          'unpaved',
    'grass':          'unpaved',
    'ground':         'unpaved',
    'mud':            'unpaved',
    'sand':           'unpaved',
    'woodchips':      'unpaved',
}

# Highway type → assumed surface when the surface tag is absent
_HIGHWAY_DEFAULT_SURFACE: Dict[str, str] = {
    'motorway':    'paved',
    'trunk':       'paved',
    'primary':     'paved',
    'secondary':   'paved',
    'tertiary':    'paved',
    'residential': 'paved',
    'service':     'paved',
    'living_street':'paved',
    'unclassified':'paved',
    'track':       'unpaved',
    'path':        'unpaved',
    'footway':     'unpaved',
    'cycleway':    'unpaved',
}

# Default road types for Live Analysis — major driveable roads only.
# Excludes footway, steps, cycleway, service, path to keep the mask sparse
# and compatible with the pipeline's expected ~20-40% coverage.
_DEFAULT_ROAD_TYPES = [
    'motorway', 'motorway_link',
    'trunk',    'trunk_link',
    'primary',  'primary_link',
    'secondary','secondary_link',
    'tertiary', 'tertiary_link',
    'residential', 'unclassified', 'living_street',
    'track',
]

# =============================================================================

# OSMConnector
# =============================================================================

class OSMConnector:
    """
    Fetches the road network from OpenStreetMap for a bounding box
    and renders it as a binary mask compatible with the DeepGlobe pipeline.

    Args:
        output_size: Side length (pixels) of the returned square mask.
                     Must match the pipeline's expected input (default 512).
    """

    def __init__(self, output_size: int = 512) -> None:
        if not _OSM_OK:
            raise RuntimeError(
                "osmnx is not installed.\n"
                "Run:  pip install osmnx"
            )
        self._size = output_size
        # Suppress osmnx console chatter; enable local caching
        ox.settings.log_console = False
        ox.settings.use_cache   = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        lat_min:    float,
        lon_min:    float,
        lat_max:    float,
        lon_max:    float,
        road_types: Optional[List[str]] = None,
    ) -> dict:
        """
        Fetch OSM road network for a bounding box and render to a binary mask.

        Args:
            lat_min, lon_min : South-West corner (EPSG:4326).
            lat_max, lon_max : North-East corner.
            road_types       : List of OSM highway values to include.
                               ``None`` → major driveable roads only
                               (motorway / trunk / primary / secondary /
                               tertiary / residential / unclassified / track).
                               Pass ``['all']`` to include every highway type.

        Returns:
            dict with keys:
                mask            — (512,512) uint8 binary road mask (0/255)
                total_roads     — number of OSM way segments fetched
                total_length_m  — total road length in metres
                highway_counts  — {highway_type: count}
                surface_counts  — {'paved': N, 'unpaved': M, 'unknown': K}
                osm_roads       — list of per-road attribute dicts
                n_nodes         — OSM graph node count
                n_edges         — OSM graph edge count
        """
        self._validate_bbox(lat_min, lon_min, lat_max, lon_max)

        # Use major roads only by default to keep mask coverage ~20-40%
        if road_types is None:
            road_types = _DEFAULT_ROAD_TYPES
        elif road_types == ['all']:
            road_types = None   # no filter = all highway types

        print(f"[INFO] Querying OSM bbox=({lat_min:.3f},{lon_min:.3f},"
              f"{lat_max:.3f},{lon_max:.3f}) ...")

        # Build osmnx custom_filter from road_types list
        if road_types:
            hwy_filter = '["highway"~"' + '|'.join(road_types) + '"]'
        else:
            hwy_filter = '["highway"]'

        try:
            # osmnx 2.x: bbox=(left, bottom, right, top) = (lon_min, lat_min, lon_max, lat_max)
            G = ox.graph_from_bbox(
                bbox          = (lon_min, lat_min, lon_max, lat_max),
                custom_filter = hwy_filter,
                retain_all    = True,
            )
        except Exception as exc:
            raise ValueError(
                f"OSM query failed for bbox=({lat_min},{lon_min},"
                f"{lat_max},{lon_max}):\n{exc}\n"
                "Try a larger bbox or check your internet connection."
            ) from exc

        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"[OK]   OSM: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # ── Render binary mask ────────────────────────────────────────────────
        mask = self._render_mask(edges, lat_min, lon_min, lat_max, lon_max)

        # ── Collect per-road stats ────────────────────────────────────────────
        highway_counts: Dict[str, int] = {}
        surface_counts: Dict[str, int] = {'paved': 0, 'unpaved': 0, 'unknown': 0}
        osm_roads: List[dict] = []
        total_length_m = 0.0

        for _, row in edges.iterrows():
            hw  = row.get('highway', 'unknown')
            hw  = hw[0] if isinstance(hw, list) else str(hw)

            osm_surf = row.get('surface', None)
            osm_surf = osm_surf[0] if isinstance(osm_surf, list) else (
                str(osm_surf) if osm_surf else None)

            surf_class = (
                _OSM_SURFACE_MAP.get(osm_surf, None)
                or _HIGHWAY_DEFAULT_SURFACE.get(hw, 'unknown')
            )

            length = float(row.get('length', 0))
            total_length_m += length

            highway_counts[hw] = highway_counts.get(hw, 0) + 1
            surface_counts[surf_class] = surface_counts.get(surf_class, 0) + 1

            osm_roads.append({
                'highway':         hw,
                'surface':         surf_class,
                'osm_surface_tag': osm_surf,
                'length_m':        round(length, 1),
                'name':            row.get('name', None),
                'maxspeed':        row.get('maxspeed', None),
                'lanes':           row.get('lanes', None),
            })

        return {
            'mask':           mask,
            'total_roads':    len(osm_roads),
            'total_length_m': round(total_length_m, 1),
            'highway_counts': highway_counts,
            'surface_counts': surface_counts,
            'osm_roads':      osm_roads,
            'n_nodes':        len(G.nodes),
            'n_edges':        len(G.edges),
        }

    @staticmethod
    def is_available() -> bool:
        """Return True if osmnx is installed."""
        return _OSM_OK

    # ── Private helpers ────────────────────────────────────────────────────────

    def _render_mask(self, edges, lat_min, lon_min, lat_max, lon_max) -> np.ndarray:
        """
        Draw OSM road edges as white polylines on a black canvas.
        Road draw-width is proportional to the highway classification.
        """
        mask = np.zeros((self._size, self._size), dtype=np.uint8)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        if lat_range <= 0 or lon_range <= 0:
            return mask

        def to_px(lat: float, lon: float):
            """Convert geographic coordinates to canvas pixel (x, y)."""
            x = int((lon - lon_min) / lon_range * (self._size - 1))
            y = int((lat_max - lat) / lat_range * (self._size - 1))
            x = max(0, min(self._size - 1, x))
            y = max(0, min(self._size - 1, y))
            return (x, y)

        for _, row in edges.iterrows():
            hw    = row.get('highway', 'unknown')
            hw    = hw[0] if isinstance(hw, list) else str(hw)
            width = _HIGHWAY_WIDTH_PX.get(hw, _DEFAULT_WIDTH_PX)

            geom = row.get('geometry', None)
            if geom is None:
                continue

            coords = list(geom.coords)   # list of (lon, lat) tuples
            if len(coords) < 2:
                continue

            pts = [to_px(lat, lon) for lon, lat in coords]
            for i in range(len(pts) - 1):
                cv2.line(mask, pts[i], pts[i + 1], 255, thickness=width)

        print(f"[OK]   Mask rendered: road_px={int((mask > 0).sum())} "
              f"({int((mask > 0).mean() * 100)}% coverage)")
        return mask

    @staticmethod
    def _validate_bbox(lat_min, lon_min, lat_max, lon_max) -> None:
        if lat_min >= lat_max:
            raise ValueError(f"lat_min ({lat_min}) must be < lat_max ({lat_max})")
        if lon_min >= lon_max:
            raise ValueError(f"lon_min ({lon_min}) must be < lon_max ({lon_max})")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitudes must be in [-90, 90]")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError("Longitudes must be in [-180, 180]")


# =============================================================================
# Change detection helper (used by /live/change endpoint)
# =============================================================================

def compute_rgb_change(rgb_t1: np.ndarray, rgb_t2: np.ndarray,
                       threshold: float = 0.12,
                       min_area_px: int  = 80) -> dict:
    """
    Compute pixel-level change between two RGB images.

    Args:
        rgb_t1       : (H,W,3) uint8  — 'before' image
        rgb_t2       : (H,W,3) uint8  — 'after'  image
        threshold    : Normalised change magnitude considered significant (0–1).
                       Default 0.12 = 12% change.
        min_area_px  : Minimum contiguous changed-pixel blob area to keep.
                       Smaller blobs (noise, cloud artefacts) are discarded.

    Returns:
        dict with:
            magnitude       — (H,W) float32 normalised change [0,1]
            change_mask     — (H,W) bool   (after denoising)
            change_map_rgb  — (H,W,3) uint8 colour-coded change image
                              Orange = brighter (construction / bare soil)
                              Blue   = darker   (new vegetation / flooding)
            changed_pct     — percentage of pixels that changed
            increased_px    — count of orange (brighter) change pixels
            decreased_px    — count of blue   (darker)  change pixels
    """
    from scipy.ndimage import (gaussian_filter, binary_opening,
                                label as nd_label, sum as nd_sum)

    f1 = rgb_t1.astype(np.float32)
    f2 = rgb_t2.astype(np.float32)

    # ── Step 1: Smooth both images before diffing ─────────────────────────────
    # A 1-pixel Gaussian blur removes single-pixel sensor noise before we diff
    f1_s = gaussian_filter(f1, sigma=1.0)
    f2_s = gaussian_filter(f2, sigma=1.0)

    diff      = f2_s - f1_s                                       # (H,W,3)
    max_norm  = np.sqrt(3.0 * 255.0 ** 2)
    magnitude = np.linalg.norm(diff, axis=2) / max_norm           # (H,W) 0→1

    change_mask = magnitude > threshold

    # ── Step 2: Morphological opening — removes isolated pixel clumps ─────────
    # opening = erosion then dilation; destroys tiny blobs, keeps large ones
    struct = np.ones((3, 3), dtype=bool)
    change_mask = binary_opening(change_mask, structure=struct, iterations=2)

    # ── Step 3: Remove small connected components (cloud artefacts, noise) ────
    labeled, n_components = nd_label(change_mask)
    component_sizes = nd_sum(change_mask, labeled, range(1, n_components + 1))
    small = np.array(component_sizes) < min_area_px
    remove_pixels = small[labeled - 1]
    remove_pixels[labeled == 0] = False
    change_mask[remove_pixels] = False

    # ── Step 4: Direction map ─────────────────────────────────────────────────
    mean_diff = diff.mean(axis=2)
    increased = change_mask & (mean_diff > 0)   # orange = brighter
    decreased = change_mask & (mean_diff < 0)   # blue   = darker

    change_map = np.zeros_like(rgb_t1)
    change_map[increased] = [255, 100, 30]    # orange-red = construction
    change_map[decreased] = [30,  120, 255]   # blue       = vegetation

    return {
        'magnitude':      magnitude.astype(np.float32),
        'change_mask':    change_mask,
        'change_map_rgb': change_map,
        'changed_pct':    round(float(change_mask.mean()) * 100, 2),
        'increased_px':   int(increased.sum()),
        'decreased_px':   int(decreased.sum()),
    }



# =============================================================================
# Standalone test
# =============================================================================

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Test OSM road fetch')
    parser.add_argument('--lat-min', type=float, default=12.90)
    parser.add_argument('--lon-min', type=float, default=77.50)
    parser.add_argument('--lat-max', type=float, default=13.10)
    parser.add_argument('--lon-max', type=float, default=77.70)
    parser.add_argument('--out', default='osm_mask_test.png')
    args = parser.parse_args()

    if not _OSM_OK:
        print("ERROR: pip install osmnx")
        sys.exit(1)

    conn   = OSMConnector()
    result = conn.fetch(args.lat_min, args.lon_min, args.lat_max, args.lon_max)

    cv2.imwrite(args.out, result['mask'])
    print(f"\nSaved: {args.out}")
    print(f"Total roads  : {result['total_roads']}")
    print(f"Total length : {result['total_length_m']} m")
    print(f"Nodes/Edges  : {result['n_nodes']} / {result['n_edges']}")
    print(f"Highway types: {result['highway_counts']}")
    print(f"Surfaces     : {result['surface_counts']}")
