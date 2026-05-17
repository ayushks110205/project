# surface_urban_patch.py
# ======================
# Drop-in post-processing patch for the M2 KMeans surface classifier.
#
# PROBLEM
# -------
# The KMeans classifier was trained on rural DeepGlobe textures.
# In dense urban/campus scenes, tropical tree canopy casts shadows
# directly onto paved tarmac roads. The shadow texture is dark and
# irregular — identical to the "damaged" signature in rural training data.
#
# This causes two systematic errors on urban imagery:
#   1. Paved roads under tree canopy → classified as "damaged"
#   2. Wide well-connected boulevards → classified as "unpaved"
#
# SOLUTION
# --------
# A three-signal heuristic that fires AFTER M2 classification:
#
#   Signal 1 — junction_density   (from the graph you already build)
#               High junction density means a structured urban network,
#               not a rural damaged road.
#
#   Signal 2 — width_m            (from M1, already computed)
#               Wide roads in urban grids are almost never truly damaged.
#
#   Signal 3 — component_size     (from the NetworkX graph)
#               If the road segment belongs to a large connected component
#               it's more likely to be a maintained urban road.
#
# USAGE
# -----
# This module exposes two paths:
#
#   1. apply_urban_surface_correction()  — per-pixel correction (standalone)
#   2. apply_urban_correction_to_graph() — edge-level correction (Tier 2 integrated)
#
# The graph-level function is preferred when the NetworkX road graph is available
# because it also recomputes vehicle costs for corrected edges.

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple

# ── label constants (match M2 output strings) ─────────────────────────────────
PAVED   = "paved"
UNPAVED = "unpaved"
DAMAGED = "damaged"


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_junction_density(G: nx.Graph) -> float:
    """
    junction_density = n_junctions / n_nodes  (junction = degree >= 3).

    Returns 0.0 for empty graphs.

    Reference values:
        Chulalongkorn campus (urban) : 68/99 ≈ 0.687
        Typical rural DeepGlobe tile : 0.15 – 0.30
    """
    if G.number_of_nodes() == 0:
        return 0.0
    n_junctions = sum(1 for _, d in G.degree() if d >= 3)
    return n_junctions / G.number_of_nodes()


def get_node_component_sizes(G: nx.Graph) -> Dict[int, int]:
    """Return {node_id: size_of_its_connected_component}."""
    size_map: Dict[int, int] = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for node in comp:
            size_map[node] = sz
    return size_map


# ─────────────────────────────────────────────────────────────────────────────
# Path A — per-pixel correction (standalone, no road_graph dependency)
# ─────────────────────────────────────────────────────────────────────────────

def apply_urban_surface_correction(
    surface_labels: np.ndarray,
    width_m:        np.ndarray,
    node_ids:       np.ndarray,
    G:              nx.Graph,
    junction_density_threshold: float = 0.45,
    width_damaged_to_paved:     float = 4.5,
    width_unpaved_to_paved:     float = 6.0,
    min_component_size:         int   = 20,
) -> np.ndarray:
    """
    Post-process M2 surface labels (per skeleton pixel) to correct urban
    canopy shadow artefacts.

    Parameters
    ----------
    surface_labels : (N,) string array — one label per skeleton pixel
    width_m        : (N,) float array  — road width in metres per pixel
    node_ids       : (N,) int array    — graph node ID nearest to each pixel
    G              : NetworkX graph of the road skeleton
    junction_density_threshold : activate correction above this density
    width_damaged_to_paved     : damaged→paved when width >= this (metres)
    width_unpaved_to_paved     : unpaved→paved when width >= this (metres)
    min_component_size         : skip pixels in tiny isolated components

    Returns
    -------
    corrected : np.ndarray of strings, same shape as surface_labels
    """
    corrected = surface_labels.copy()

    jd = compute_junction_density(G)
    if jd < junction_density_threshold:
        return corrected   # rural scene — trust M2 as-is

    comp_sizes = get_node_component_sizes(G)

    for i in range(len(corrected)):
        if comp_sizes.get(int(node_ids[i]), 0) < min_component_size:
            continue
        w     = float(width_m[i])
        label = corrected[i]
        if label == DAMAGED and w >= width_damaged_to_paved:
            corrected[i] = PAVED
        elif label == UNPAVED and w >= width_unpaved_to_paved:
            corrected[i] = PAVED

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Path B — edge-level correction (preferred when Tier 2 graph is available)
# ─────────────────────────────────────────────────────────────────────────────

def apply_urban_correction_to_graph(
    G: nx.Graph,
    surface_multipliers: Dict[str, Dict[str, float]],
    min_width_m:         Dict[str, float],
    junction_density_threshold: float = 0.45,
    width_damaged_to_paved:     float = 4.5,
    width_unpaved_to_paved:     float = 6.0,
    min_component_size:         int   = 20,
) -> Tuple[nx.Graph, bool, float, int]:
    """
    Apply urban canopy shadow correction directly to NetworkX graph edge
    attributes, then recompute vehicle costs for corrected edges.

    This is the preferred integration point when the Tier 2 road graph is
    already available because:
      - No per-pixel iteration needed
      - Vehicle costs are automatically updated
      - Corrected dominant_surface propagates into JSON summary

    Parameters
    ----------
    G                    : NetworkX graph (modified IN-PLACE)
    surface_multipliers  : dict of {vehicle: {surface: cost_multiplier}}
                           — pass road_graph._SURFACE_MULTIPLIERS
    min_width_m          : dict of {vehicle: min_width_metres}
                           — pass road_graph._MIN_WIDTH_M
    junction_density_threshold : activate correction above this density
    width_damaged_to_paved     : damaged→paved for edges >= this width
    width_unpaved_to_paved     : unpaved→paved for edges >= this width
    min_component_size         : skip edges whose component is < this

    Returns
    -------
    G               : the same graph (modified in-place, returned for chaining)
    is_urban        : True if urban correction was applied
    junction_density: computed junction density value
    n_corrected     : number of edges whose surface label was changed
    """
    from road_graph import VEHICLE_TYPES  # local import to avoid circular dep

    jd       = compute_junction_density(G)
    is_urban = jd >= junction_density_threshold

    if not is_urban:
        return G, False, jd, 0

    comp_sizes = get_node_component_sizes(G)
    n_corrected = 0

    for u, v, data in G.edges(data=True):
        c_size = min(comp_sizes.get(u, 0), comp_sizes.get(v, 0))
        if c_size < min_component_size:
            continue

        old_surf = data.get('dominant_surface', '')
        w        = data.get('mean_width_m', 0.0)
        new_surf = old_surf

        if old_surf == DAMAGED and w >= width_damaged_to_paved:
            new_surf = PAVED
        elif old_surf == UNPAVED and w >= width_unpaved_to_paved:
            new_surf = PAVED

        if new_surf != old_surf:
            data['dominant_surface'] = new_surf
            # Recompute vehicle costs for this edge
            length_m     = data.get('length_m', 0.0)
            mean_width_m = data.get('mean_width_m', 0.0)
            for vtype in VEHICLE_TYPES:
                mult     = surface_multipliers[vtype].get(new_surf, 1.5)
                min_w    = min_width_m[vtype]
                if mult == float('inf'):
                    cost = float('inf')
                else:
                    if min_w > 0 and mean_width_m < min_w:
                        deficit_ratio = min_w / max(mean_width_m, 0.05)
                        mult = mult * min(deficit_ratio * 20.0, 1000.0)
                    cost = length_m * mult
                data[f'cost_{vtype}'] = cost
            n_corrected += 1

    return G, True, jd, n_corrected


def propagate_correction_to_surface_map(
    G:           nx.Graph,
    surface_map: np.ndarray,
) -> np.ndarray:
    """
    After edge-level urban correction, propagate corrected dominant_surface
    labels back to the pixel-level surface_map (H×W string object array).

    For each edge, all pixels in its pixel_path are relabelled with the
    (potentially corrected) dominant_surface of that edge.

    Args:
        G           : NetworkX graph with corrected edge attributes.
        surface_map : (H, W) object array of str — from M2 classifier.

    Returns:
        surface_map : same array, modified in-place and returned.
    """
    for _, _, data in G.edges(data=True):
        surf = data.get('dominant_surface', '')
        if not surf:
            continue
        for r, c in data.get('pixel_path', []):
            surface_map[r, c] = surf
    return surface_map


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_surface_stats(surface_labels: np.ndarray) -> dict:
    """Recompute surface_counts and dominant_surface after correction."""
    unique, counts = np.unique(surface_labels, return_counts=True)
    surface_counts = dict(zip(unique.tolist(), counts.tolist()))
    dominant = max(surface_counts, key=surface_counts.get) if surface_counts else ''
    return {
        'surface_counts':   surface_counts,
        'dominant_surface': dominant,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    G = nx.grid_2d_graph(10, 10)
    G = nx.convert_node_labels_to_integers(G)

    n   = 200
    rng = np.random.default_rng(42)
    labels = rng.choice([PAVED, UNPAVED, DAMAGED], size=n, p=[0.10, 0.54, 0.36])
    widths = rng.uniform(1.0, 10.0, size=n)
    nodes  = rng.integers(0, G.number_of_nodes(), size=n)

    jd = compute_junction_density(G)
    print(f"Junction density : {jd:.3f}  (urban threshold = 0.45)")
    print(f"Is urban scene   : {jd >= 0.45}")
    print()

    before = compute_surface_stats(labels)
    print("Before correction:", before['surface_counts'])

    corrected = apply_urban_surface_correction(labels, widths, nodes, G)
    after = compute_surface_stats(corrected)
    print("After  correction:", after['surface_counts'])
    print("Dominant surface :", after['dominant_surface'])
    print(f"Pixels corrected : {int((labels != corrected).sum())} / {n}")
