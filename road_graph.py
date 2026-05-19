# =============================================================================
# road_graph.py  –  Tier 2: Skeleton-to-NetworkX Graph + Vehicle-Aware Routing
# =============================================================================
# pip install networkx
#
# Replaces/extends the MCP_Geometric approach in road_router.py with a proper
# NetworkX graph built from the road skeleton.
#
# Pipeline:
#   1. skeleton → junction/endpoint nodes
#   2. Trace segments between nodes → edge attributes (length, width, surface)
#   3. Compute 4 vehicle-aware edge costs
#   4. find_top3_routes() — top-3 alternative paths per vehicle type
#   5. draw_routes()      — visualise routes on satellite RGB
#
# Usage:
#   from road_graph import RoadGraph, find_top3_routes, pick_src_dst_auto, draw_routes
#   rg     = RoadGraph(tier1_result)
#   src, dst = pick_src_dst_auto(rg.G)
#   routes = find_top3_routes(rg.G, src, dst, 'car')
#   viz    = draw_routes(satellite_rgb, rg.G, routes)
# =============================================================================

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation as _bin_dilate

# Urban canopy shadow correction (applied after graph build)
try:
    from surface_urban_patch import (
        apply_urban_correction_to_graph,
        propagate_correction_to_surface_map,
    )
    _URBAN_PATCH_OK = True
except ImportError:
    _URBAN_PATCH_OK = False

# GSD used throughout (metres per pixel, matches DeepGlobe pipeline)
_GSD_M_PER_PX: float = 0.5

# 8-connected neighbour offsets
_NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1),
                 (0,  -1),          (0,  1),
                 (1,  -1), (1,  0), (1,  1)]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphConfig:
    """
    Tuneable parameters for :class:`RoadGraph`.

    Attributes:
        gsd_m_per_px      : Ground sampling distance (metres per pixel).
        fallback_step_px  : Spacing for fallback node sampling when no
                            junctions/endpoints are detected.
    """
    gsd_m_per_px:     float = 0.5
    fallback_step_px: int   = 20


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle profiles
# ─────────────────────────────────────────────────────────────────────────────

_SURFACE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    'pedestrian': {'paved': 1.0, 'unpaved': 1.2, 'damaged': 1.5,  '': 1.2},
    'motorcycle': {'paved': 1.0, 'unpaved': 1.5, 'damaged': 2.5,  '': 1.5},
    'car':        {'paved': 1.0, 'unpaved': 2.0, 'damaged': 4.0,  '': 2.0},
    'truck':      {'paved': 1.0, 'unpaved': 3.0, 'damaged': float('inf'), '': 3.0},
}

# Minimum *mean* width (metres) for a segment to be traversable by a vehicle.
# We use mean_width_m (not min_width_m) so that a single narrow junction pixel
# doesn't block the whole segment — common at DeepGlobe's 0.5 m/px GSD.
# Thresholds are kept conservative but realistic for satellite road widths.
_MIN_WIDTH_M: Dict[str, float] = {
    'pedestrian': 0.0,   # footpaths / narrow alleys
    'motorcycle': 0.5,   # ~1 skeleton pixel wide
    'car':        1.5,   # ~3 skeleton pixels wide  (lowered from 2.0)
    'truck':      2.5,   # ~5 skeleton pixels wide  (lowered from 4.0)
}

VEHICLE_TYPES: List[str] = ['pedestrian', 'motorcycle', 'car', 'truck']


# ─────────────────────────────────────────────────────────────────────────────
# Route result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """
    A single found route between two graph nodes.

    Attributes:
        rank               : 1 = shortest, 2/3 = alternatives.
        vehicle_type       : vehicle profile used.
        node_path          : ordered list of node IDs.
        pixel_path         : concatenated (row, col) pixel coordinates.
        total_distance_m   : sum of edge length_m along route.
        total_cost         : sum of vehicle cost along route.
        mean_width_m       : mean of per-edge mean_width_m.
        dominant_surface   : mode of dominant_surface across edges.
        dominant_road_type : mode of dominant_road_type across edges.
    """
    rank:               int
    vehicle_type:       str
    node_path:          List[int]
    pixel_path:         List[Tuple[int, int]]
    total_distance_m:   float
    total_cost:         float
    mean_width_m:       float
    dominant_surface:   str
    dominant_road_type: str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_skel_neighbours(skeleton: np.ndarray) -> np.ndarray:
    """
    Count 8-connected skeleton neighbours for every skeleton pixel.

    Args:
        skeleton : (H, W) bool array.

    Returns:
        (H, W) uint8 neighbour-count array (0 for non-skeleton pixels).
    """
    skel_u8 = skeleton.astype(np.uint8)
    kernel  = np.ones((3, 3), dtype=np.uint8)
    # convolve counts ALL 9 cells; subtract self
    counts  = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel.astype(np.uint8))
    counts  = np.clip(counts.astype(np.int32) - skel_u8.astype(np.int32), 0, 8)
    return (counts * skel_u8).astype(np.uint8)


def _mode_str(values: List[str]) -> str:
    """Return most common non-empty string from *values*, or '' if all empty."""
    non_empty = [v for v in values if v and v != '']
    if not non_empty:
        return ''
    return Counter(non_empty).most_common(1)[0][0]


def _edge_length_m(pixel_path: List[Tuple[int, int]], gsd: float) -> float:
    """Euclidean path length in metres."""
    if len(pixel_path) < 2:
        return 0.0
    pts   = np.array(pixel_path, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum() * gsd)


def _vehicle_cost(length_m: float,
                  mean_width_m: float,
                  dominant_surface: str,
                  vehicle: str) -> float:
    """Compute a single vehicle cost for one edge.

    Uses *mean_width_m* (not min_width_m) for the traversability check so
    that isolated narrow junction pixels don't block an otherwise wide road
    segment — a common artefact of the medial-axis skeleton at 0.5 m/px GSD.

    Width handling:
        If the segment is narrower than the vehicle minimum, we apply a heavy
        **penalty multiplier** rather than returning ``float('inf')``.  This
        ensures a route is always found when the graph is topologically
        connected — the router simply strongly prefers wider roads and only
        uses narrow bottlenecks as a last resort.  A hard ``inf`` is still
        returned for truly impassable *surfaces* (e.g. truck on damaged road).
    """
    mult = _SURFACE_MULTIPLIERS[vehicle].get(dominant_surface, 1.5)
    if mult == float('inf'):
        return float('inf')   # truly impassable surface — keep as inf

    # Width penalty: graduated factor so wider roads are always preferred.
    # If below the minimum width, add a 20× surcharge per unit of deficit.
    min_w = _MIN_WIDTH_M[vehicle]
    if min_w > 0 and mean_width_m < min_w:
        # penalty grows as width shrinks; capped to avoid numerical issues
        deficit_ratio = min_w / max(mean_width_m, 0.05)
        width_penalty = min(deficit_ratio * 20.0, 1000.0)
        mult = mult * width_penalty

    return length_m * mult


# ─────────────────────────────────────────────────────────────────────────────
# Segment tracer
# ─────────────────────────────────────────────────────────────────────────────

def _trace_segments(
        skeleton:      np.ndarray,
        node_set:      set,
        width_m:       np.ndarray,
        road_type_map: np.ndarray,
        surface_map:   np.ndarray,
        gsd:           float,
) -> List[dict]:
    """
    Walk the skeleton and trace road segments between node pixels.

    For each node pixel, attempt a DFS walk along skeleton pixels that are
    NOT yet covered by a traced segment.  The walk terminates when another
    node pixel is reached.

    Args:
        skeleton      : (H, W) bool.
        node_set      : set of (row, col) tuples that are junction/endpoint nodes.
        width_m       : (H, W) float32.
        road_type_map : (H, W) str object array.
        surface_map   : (H, W) str object array.
        gsd           : metres per pixel.

    Returns:
        List of segment dicts, each with keys:
            src_rc, dst_rc, pixel_path, length_m,
            mean_width_m, min_width_m,
            dominant_road_type, dominant_surface.
    """
    H, W = skeleton.shape
    visited_edges: set = set()   # frozenset of (src_rc, dst_rc) pairs
    segments: List[dict] = []

    def _skel_neighbours(r: int, c: int) -> List[Tuple[int, int]]:
        result = []
        for dr, dc in _NEIGHBOURS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc]:
                result.append((nr, nc))
        return result

    for start in node_set:
        for nb in _skel_neighbours(*start):
            if nb in node_set:
                # Direct node–node edge
                key = frozenset([start, nb])
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                path = [start, nb]
            else:
                # Walk until we hit another node or dead-end
                key_candidate = frozenset([start, nb])
                if key_candidate in visited_edges:
                    continue

                path  = [start, nb]
                prev  = start
                curr  = nb
                found = False

                while True:
                    nbs = [n for n in _skel_neighbours(*curr) if n != prev]
                    if not nbs:
                        # dead end — only valid if curr itself is a node
                        if curr in node_set:
                            found = True
                        break
                    if len(nbs) == 1:
                        nxt = nbs[0]
                        if nxt in node_set:
                            path.append(nxt)
                            found = True
                            break
                        path.append(nxt)
                        prev, curr = curr, nxt
                    else:
                        # branching — we're at a junction that wasn't flagged
                        found = True
                        break

                if not found:
                    continue

                key = frozenset([path[0], path[-1]])
                if key in visited_edges or path[0] == path[-1]:
                    continue
                visited_edges.add(key)

            # Compute segment attributes
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            w_vals  = [float(width_m[r, c])        for r, c in path]
            rt_vals = [str(road_type_map[r, c])     for r, c in path]
            sf_vals = [str(surface_map[r, c])       for r, c in path]

            segments.append({
                'src_rc':            path[0],
                'dst_rc':            path[-1],
                'pixel_path':        path,
                'length_m':          _edge_length_m(path, gsd),
                'mean_width_m':      float(np.mean(w_vals)) if w_vals else 0.0,
                'min_width_m':       float(np.min(w_vals))  if w_vals else 0.0,
                'dominant_road_type': _mode_str(rt_vals),
                'dominant_surface':  _mode_str(sf_vals),
            })

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class RoadGraph:
    """
    Builds a vehicle-aware NetworkX graph from Tier 1 road intelligence results.

    Args:
        tier1_result : dict returned by ``SatellitePipeline.run_tier1()``.
                       Must contain ``width_result`` and ``type_result``.
        config       : :class:`GraphConfig` (optional).

    Attributes:
        G : ``nx.Graph`` — the constructed road network graph.
    """

    def __init__(self,
                 tier1_result: dict,
                 config: GraphConfig = None) -> None:

        self.cfg = config or GraphConfig()

        # Unpack Tier 1 arrays
        wr = tier1_result['width_result']
        tr = tier1_result['type_result']

        self.skeleton:      np.ndarray = wr.skeleton         # (H,W) bool
        self.width_m:       np.ndarray = wr.width_m          # (H,W) float32
        self.road_type_map: np.ndarray = wr.road_type_map    # (H,W) str object
        self.surface_map:   np.ndarray = tr['surface_map']   # (H,W) str object

        self.G: nx.Graph = self._build_graph()

        # ── Urban canopy shadow correction ────────────────────────────────────
        # After the graph is built we know junction_density, component sizes,
        # and per-edge mean_width_m — enough to detect urban scenes and fix
        # systematic M2 mis-classifications (shadow on paved → "damaged").
        self.is_urban_corrected: bool  = False
        self.junction_density:   float = 0.0
        self.n_urban_corrected:  int   = 0

        if _URBAN_PATCH_OK and self.G.number_of_edges() > 0:
            self.G, self.is_urban_corrected, self.junction_density, \
                self.n_urban_corrected = apply_urban_correction_to_graph(
                    self.G,
                    _SURFACE_MULTIPLIERS,
                    _MIN_WIDTH_M,
                )
            if self.is_urban_corrected and self.n_urban_corrected > 0:
                # Propagate corrected edge labels back to the pixel surface_map
                # so Tier 1 visualisations also reflect the correction.
                propagate_correction_to_surface_map(self.G, self.surface_map)
                print(f"  Urban correction ✓  "
                      f"jd={self.junction_density:.3f}  "
                      f"{self.n_urban_corrected} edge(s) relabelled "
                      f"(shadow-on-paved artefact fixed)")

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_graph(self) -> nx.Graph:
        """Construct and return the full road graph."""
        G   = nx.Graph()
        skel = self.skeleton

        if not skel.any():
            return G  # empty graph for empty skeleton

        # Bridge thin gaps caused by 1-2 px dirt paths / building subtraction.
        # Dilate by 2 iterations then re-skeletonize to reconnect fragments.
        # We re-import skeletonize here to avoid a top-level circular dep.
        try:
            from skimage.morphology import skeletonize as _skel_fn
            skel_dilated = _bin_dilate(skel, iterations=2)
            skel = _skel_fn(skel_dilated).astype(bool)
            print(f"  Skeleton bridging: "
                  f"{int(skel.sum())} px after gap-fill dilation")
        except Exception as _e:
            print(f"  Skeleton bridging skipped ({_e})")

        # Step 1: find node pixels
        nbr_counts = _count_skel_neighbours(skel)
        endpoint_mask  = skel & (nbr_counts == 1)
        junction_mask  = skel & (nbr_counts >= 3)

        node_pixels: Dict[Tuple[int, int], str] = {}
        for r, c in zip(*np.where(endpoint_mask)):
            node_pixels[(int(r), int(c))] = 'endpoint'
        for r, c in zip(*np.where(junction_mask)):
            node_pixels[(int(r), int(c))] = 'junction'

        # Fallback: sample every N pixels if no structural nodes
        if len(node_pixels) < 2:
            step = self.cfg.fallback_step_px
            rows, cols = np.where(skel)
            for r, c in zip(rows[::step], cols[::step]):
                node_pixels[(int(r), int(c))] = 'sampled'

        if len(node_pixels) < 2:
            return G  # single-pixel skeleton — nothing to route

        # Assign integer node IDs
        rc_to_id: Dict[Tuple[int, int], int] = {
            rc: i for i, rc in enumerate(node_pixels)}

        for rc, ntype in node_pixels.items():
            nid = rc_to_id[rc]
            G.add_node(nid, row=rc[0], col=rc[1], node_type=ntype)

        # Step 2: trace segments
        segments = _trace_segments(
            skel, set(node_pixels.keys()),
            self.width_m, self.road_type_map, self.surface_map,
            self.cfg.gsd_m_per_px)

        # Step 3: add edges with vehicle costs
        for seg in segments:
            src_id = rc_to_id.get(seg['src_rc'])
            dst_id = rc_to_id.get(seg['dst_rc'])
            if src_id is None or dst_id is None or src_id == dst_id:
                continue
            if G.has_edge(src_id, dst_id):
                # Keep the shorter segment if duplicate
                if G[src_id][dst_id]['length_m'] <= seg['length_m']:
                    continue

            # Use mean_width_m for cost computation — see _vehicle_cost() note.
            costs = {
                f'cost_{v}': _vehicle_cost(
                    seg['length_m'], seg['mean_width_m'],
                    seg['dominant_surface'], v)
                for v in VEHICLE_TYPES
            }
            G.add_edge(src_id, dst_id,
                       length_m=seg['length_m'],
                       mean_width_m=seg['mean_width_m'],
                       min_width_m=seg['min_width_m'],
                       dominant_road_type=seg['dominant_road_type'],
                       dominant_surface=seg['dominant_surface'],
                       pixel_path=seg['pixel_path'],
                       **costs)

        return G


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Top-3 route query
# ─────────────────────────────────────────────────────────────────────────────

def _route_stats(G: nx.Graph,
                 node_path: List[int],
                 vehicle_type: str) -> RouteResult:
    """Compute summary statistics for a node-level path."""
    pixel_path:    List[Tuple[int, int]] = []
    distances:     List[float] = []
    costs:         List[float] = []
    widths:        List[float] = []
    surfaces:      List[str]   = []
    road_types:    List[str]   = []
    cost_key = f'cost_{vehicle_type}'

    for u, v in zip(node_path[:-1], node_path[1:]):
        ed = G[u][v]
        pixel_path.extend(ed['pixel_path'])
        distances.append(ed['length_m'])
        costs.append(ed.get(cost_key, ed['length_m']))
        widths.append(ed['mean_width_m'])
        surfaces.append(ed['dominant_surface'])
        road_types.append(ed['dominant_road_type'])

    return RouteResult(
        rank=0,
        vehicle_type=vehicle_type,
        node_path=node_path,
        pixel_path=pixel_path,
        total_distance_m=float(sum(distances)),
        total_cost=float(sum(costs)),
        mean_width_m=float(np.mean(widths)) if widths else 0.0,
        dominant_surface=_mode_str(surfaces),
        dominant_road_type=_mode_str(road_types),
    )


def find_top3_routes(G: nx.Graph,
                     src_node: int,
                     dst_node: int,
                     vehicle_type: str) -> List[RouteResult]:
    """
    Find up to 3 alternative routes between *src_node* and *dst_node*.

    Route 1 is the shortest path weighted by ``cost_<vehicle_type>``.
    Routes 2 and 3 are found by temporarily removing the edges of previous
    routes and re-running shortest path.  Missing routes (disconnected graph
    for that vehicle profile) are silently skipped.

    Args:
        G            : NetworkX graph from :class:`RoadGraph`.
        src_node     : source node ID.
        dst_node     : destination node ID.
        vehicle_type : one of ``'pedestrian'``, ``'motorcycle'``,
                       ``'car'``, ``'truck'``.

    Returns:
        List of :class:`RouteResult` (length 0–3), ranked 1/2/3.

    Raises:
        ValueError : If *vehicle_type* is not recognised.
    """
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(
            f"Unknown vehicle_type '{vehicle_type}'. Choose from: {VEHICLE_TYPES}")

    if G.number_of_nodes() < 2:
        return []

    cost_key  = f'cost_{vehicle_type}'
    routes:   List[RouteResult] = []
    excluded: List[Tuple[int, int, dict]] = []  # (u, v, edge_data) removed so far

    for rank in range(1, 4):
        # Temporarily remove excluded edges
        for u, v, _ in excluded:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        try:
            node_path = nx.shortest_path(
                G, src_node, dst_node, weight=cost_key)
            result = _route_stats(G, node_path, vehicle_type)

            # Check the route is actually traversable (cost < inf)
            if result.total_cost == float('inf'):
                break

            result.rank = rank
            routes.append(result)

            # Queue this route's edges for exclusion in next iteration
            for u, v in zip(node_path[:-1], node_path[1:]):
                if G.has_edge(u, v):
                    excluded.append((u, v, dict(G[u][v])))

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        finally:
            # Always restore removed edges
            for u, v, data in excluded:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **data)

    # Restore any edges still missing after the loop
    for u, v, data in excluded:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **data)

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Pixel coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_nearest_node(G: nx.Graph, row: int, col: int) -> Optional[int]:
    """
    Find the graph node nearest to pixel coordinate *(row, col)*.

    Args:
        G   : NetworkX graph.
        row : pixel row.
        col : pixel column.

    Returns:
        Node ID of the nearest node, or None if the graph is empty.
    """
    if G.number_of_nodes() == 0:
        return None
    node_ids = list(G.nodes())
    coords   = np.array([[G.nodes[n]['row'], G.nodes[n]['col']]
                          for n in node_ids], dtype=np.float32)
    query    = np.array([row, col], dtype=np.float32)
    dists    = np.sqrt(((coords - query) ** 2).sum(axis=1))
    return node_ids[int(np.argmin(dists))]


def pick_src_dst_auto(G: nx.Graph,
                      max_sample: int = 500,
                      rng_seed:   int = 0
                      ) -> Tuple[Optional[int], Optional[int]]:
    """
    Automatically pick the two graph nodes that are furthest apart,
    restricted to the **largest connected component** so that a path
    is guaranteed to exist between src and dst.

    Preferentially selects from endpoint nodes within the largest
    component; falls back to all nodes in that component if fewer
    than 2 endpoints exist.  For large graphs, subsamples *max_sample*
    nodes to avoid O(N²) computation.

    Args:
        G          : NetworkX graph.
        max_sample : maximum nodes to include in distance computation.
        rng_seed   : seed for reproducible subsampling.

    Returns:
        ``(src_node_id, dst_node_id)`` or ``(None, None)`` if < 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return None, None

    # ── Restrict to largest connected component ───────────────────────────────
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)

    endpoints = [n for n, d in G.nodes(data=True)
                 if d.get('node_type') == 'endpoint' and n in largest_cc]
    candidates = endpoints if len(endpoints) >= 2 else [n for n in largest_cc]

    if len(candidates) < 2:
        return None, None

    node_ids = candidates
    coords   = np.array([[G.nodes[n]['row'], G.nodes[n]['col']]
                          for n in node_ids], dtype=np.float32)

    if len(node_ids) > max_sample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(node_ids), size=max_sample, replace=False)
        node_ids = [node_ids[i] for i in idx]
        coords   = coords[idx]

    # Pairwise distance matrix
    diff     = coords[:, None, :] - coords[None, :, :]  # (M, M, 2)
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
    flat_idx = int(np.argmax(dist_mat))
    i, j     = divmod(flat_idx, len(node_ids))
    return node_ids[i], node_ids[j]


def get_graph_summary(G: nx.Graph,
                      is_urban_scene:    bool  = False,
                      junction_density:  float = 0.0,
                      n_urban_corrected: int   = 0) -> dict:
    """
    Return a concise diagnostic summary of the road network graph.

    Args:
        G                 : NetworkX graph from :class:`RoadGraph`.
        is_urban_scene    : True if urban canopy correction was applied.
        junction_density  : graph junction density (n_junctions/n_nodes).
        n_urban_corrected : number of edges whose surface was corrected.

    Returns:
        dict with keys: n_nodes, n_edges, n_components,
        largest_component_size, n_endpoints, n_junctions,
        mean_width_m, width_p25/50/75_m, preferred_pct_<vehicle>,
        is_urban_scene, junction_density, urban_edges_corrected.
    """
    components = list(nx.connected_components(G))
    largest_cc_size = max((len(c) for c in components), default=0)
    n_endpoints = sum(
        1 for _, d in G.nodes(data=True) if d.get('node_type') == 'endpoint')
    n_junctions = sum(
        1 for _, d in G.nodes(data=True) if d.get('node_type') == 'junction')

    # Width statistics over all edges
    widths = [d.get('mean_width_m', 0.0) for _, _, d in G.edges(data=True)]
    if widths:
        w_arr  = np.array(widths, dtype=np.float32)
        mean_w = float(np.mean(w_arr))
        p25_w  = float(np.percentile(w_arr, 25))
        p50_w  = float(np.percentile(w_arr, 50))
        p75_w  = float(np.percentile(w_arr, 75))
    else:
        mean_w = p25_w = p50_w = p75_w = 0.0

    # % of edges at or above minimum width (no width penalty applied)
    n_edges  = G.number_of_edges()
    preferred: dict = {}
    for vtype in VEHICLE_TYPES:
        min_w_v = _MIN_WIDTH_M[vtype]
        if n_edges == 0:
            preferred[f'preferred_pct_{vtype}'] = 0.0
        else:
            ok = sum(1 for _, _, d in G.edges(data=True)
                     if d.get('mean_width_m', 0.0) >= min_w_v)
            preferred[f'preferred_pct_{vtype}'] = round(100.0 * ok / n_edges, 1)

    return {
        'n_nodes':                G.number_of_nodes(),
        'n_edges':                G.number_of_edges(),
        'n_components':           len(components),
        'largest_component_size': largest_cc_size,
        'n_endpoints':            n_endpoints,
        'n_junctions':            n_junctions,
        'mean_width_m':           round(mean_w, 2),
        'width_p25_m':            round(p25_w, 2),
        'width_p50_m':            round(p50_w, 2),
        'width_p75_m':            round(p75_w, 2),
        **preferred,
        # Urban canopy shadow correction metadata
        'is_urban_scene':         is_urban_scene,
        'junction_density':       round(junction_density, 3),
        'urban_edges_corrected':  n_urban_corrected,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Route visualisation
# ─────────────────────────────────────────────────────────────────────────────

_ROUTE_COLORS = {
    1: (0,   255, 255),   # cyan   — Route 1 (BGR for cv2)
    2: (0,   255, 255),   # yellow — Route 2
    3: (255, 0,   255),   # magenta — Route 3
}
_ROUTE_COLORS_BGR = {
    1: (255, 255, 0),     # cyan   in BGR
    2: (0,   255, 255),   # yellow in BGR
    3: (255, 0,   255),   # magenta in BGR
}
_ROUTE_THICKNESS = {1: 3, 2: 2, 3: 2}


def draw_routes(satellite_rgb: np.ndarray,
                G:             nx.Graph,
                routes:        List[RouteResult],
                node_size:     int = 3) -> np.ndarray:
    """
    Draw all skeleton edges, graph nodes, and up to 3 coloured routes on a
    copy of *satellite_rgb*.

    Colour scheme:
        - Skeleton edges (background) : thin grey (1 px)
        - All nodes                   : white dots
        - Route 1                     : cyan,    3 px thick
        - Route 2                     : yellow,  2 px thick
        - Route 3                     : magenta, 2 px thick
        - Source node                 : green filled circle (r=8)
        - Destination node            : red filled circle (r=8)
        - Legend                      : top-left text summary

    Args:
        satellite_rgb : (H, W, 3) uint8 RGB satellite image.
        G             : NetworkX graph from :class:`RoadGraph`.
        routes        : list of :class:`RouteResult` (up to 3).
        node_size     : radius of node markers in pixels.

    Returns:
        (H, W, 3) uint8 RGB annotated image.
    """
    # Work in BGR for cv2, convert back at end
    canvas = cv2.cvtColor(satellite_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # ── Draw all skeleton edges (grey) ────────────────────────────────────────
    for u, v, data in G.edges(data=True):
        path = data.get('pixel_path', [])
        if len(path) < 2:
            continue
        pts = np.array([[c, r] for r, c in path], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=(80, 80, 80), thickness=1)

    # ── Draw all nodes (white dots) ───────────────────────────────────────────
    for n, d in G.nodes(data=True):
        cv2.circle(canvas, (d['col'], d['row']), node_size,
                   (200, 200, 200), -1)

    # ── Draw routes ───────────────────────────────────────────────────────────
    src_node: Optional[int] = None
    dst_node: Optional[int] = None

    for route in routes:
        color = _ROUTE_COLORS_BGR.get(route.rank, (128, 128, 128))
        thick = _ROUTE_THICKNESS.get(route.rank, 2)
        path  = route.pixel_path
        if len(path) < 2:
            continue
        pts = np.array([[c, r] for r, c in path], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thick)

        if src_node is None and route.node_path:
            src_node = route.node_path[0]
            dst_node = route.node_path[-1]

    # ── Source / destination markers ──────────────────────────────────────────
    if src_node is not None and G.has_node(src_node):
        d = G.nodes[src_node]
        cv2.circle(canvas, (d['col'], d['row']), 8, (0, 255, 0), -1)   # green
    if dst_node is not None and G.has_node(dst_node):
        d = G.nodes[dst_node]
        cv2.circle(canvas, (d['col'], d['row']), 8, (0, 0, 255), -1)   # red

    # ── Legend ────────────────────────────────────────────────────────────────
    _LEGEND_LABELS = {1: 'cyan', 2: 'yellow', 3: 'magenta'}
    y0 = 18
    vehicle_shown = routes[0].vehicle_type if routes else 'n/a'
    # Header line — vehicle type
    cv2.putText(canvas, f"Vehicle: {vehicle_shown}", (8, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    y0 += 18
    for route in routes:
        label = (f"Route {route.rank} ({_LEGEND_LABELS.get(route.rank,'?')}): "
                 f"{route.total_distance_m:.0f}m, {route.dominant_surface}, "
                 f"w={route.mean_width_m:.1f}m")
        cv2.putText(canvas, label, (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    _ROUTE_COLORS_BGR.get(route.rank, (200, 200, 200)), 1,
                    cv2.LINE_AA)
        y0 += 18

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
