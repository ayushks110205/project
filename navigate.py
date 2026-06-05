# =============================================================================
# navigate.py  —  Real-World Geocoded Routing with ML Surface Analysis
# =============================================================================
#
# Orchestrates:
#   1. Geocoding (Nominatim / raw lat,lng)
#   2. osmnx road-network download (cached, keyword-only bbox for osmnx>=2.0)
#   3. Sentinel-2 imagery fetch (graceful fallback)
#   4. ML surface + width analysis (M2 + M1)
#   5. Paint ML results onto osmnx edges (with ±3 px radius search)
#   6. Dual Dijkstra routing (fastest + safest)
#   7. Response assembly (polyline, segments, warnings, time estimates)
#
# Usage (from app.py):
#   from navigate import navigate
#   result = navigate("Connaught Place, Delhi", "India Gate, Delhi", "car")
# =============================================================================

from __future__ import annotations

import concurrent.futures
import functools
import math
import os
import requests as _req
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Optional dependencies ─────────────────────────────────────────────────────
_NAV_DEPS_OK = False
try:
    import osmnx as ox
    import networkx as nx
    from geopy.geocoders import Nominatim
    _NAV_DEPS_OK = True
except ImportError:
    pass

# ML modules (guarded)
_ML_OK = False
try:
    from road_width import RoadWidthEstimator
    from road_type_classifier import RoadTypeClassifier
    _ML_OK = True
except ImportError:
    pass

# Sentinel (guarded)
_SENTINEL_OK = False
try:
    from sentinel_connector import SentinelConnector
    _SENTINEL_OK = True
except ImportError:
    pass

# OSM raster mask (guarded)
_OSM_MASK_OK = False
try:
    from osm_connector import OSMConnector
    _OSM_MASK_OK = True
except ImportError:
    pass


# =============================================================================
# Constants
# =============================================================================

EARTH_RADIUS_KM = 6371.0
MAX_ROUTE_KM = 50.0

# Vehicle minimum road widths (metres) for passability.
# More conservative than road_graph.py values because these apply to real-world
# OSM roads, not satellite-derived skeleton widths at 0.5 m/px GSD.
_NAV_MIN_WIDTH: Dict[str, float] = {
    'pedestrian': 0.0,
    'motorcycle': 1.5,
    'car':        3.0,
    'truck':      6.0,
}

# Damage multiplier for safest-route weighting.
# ONLY used for path selection — never shown to user.
_DAMAGE_MULTIPLIER: Dict[str, float] = {
    'paved':   1.0,
    'unpaved': 1.4,
    'damaged': 2.5,
    '':        1.4,   # unknown → unpaved-like
}

# Default speeds (km/h) by OSM highway type.
_SPEED_DEFAULTS_KMH: Dict[str, float] = {
    'motorway': 80, 'motorway_link': 80,
    'trunk': 80,    'trunk_link': 60,
    'primary': 50,  'primary_link': 50,
    'secondary': 50, 'secondary_link': 40,
    'tertiary': 30,  'tertiary_link': 30,
    'residential': 30, 'living_street': 20,
    'unclassified': 20, 'service': 15,
    'track': 15, 'path': 5, 'footway': 5, 'cycleway': 15,
}

# Vehicle speed caps (km/h) — applied AFTER highway-type lookup.
_VEHICLE_SPEED_CAP_KMH: Dict[str, float] = {
    'pedestrian': 5,
    'motorcycle': 60,
    'car':        80,
    'truck':      60,
}

# osmnx network_type per vehicle.
_NETWORK_TYPE: Dict[str, str] = {
    'car':        'drive',
    'truck':      'drive',
    'pedestrian': 'all',
    'motorcycle': 'bike',
}


# =============================================================================
# Step 1 — Geocoding
# =============================================================================

def geocode(place: str) -> Tuple[float, float]:
    """
    Convert a place name or ``'lat,lng'`` string to ``(lat, lng)`` floats.

    Tries raw coordinate parsing first (fast path — always used when the
    frontend autocomplete has cached coordinates).  Falls back to the
    Mappls Geocoding REST API for plain text queries.

    Raises:
        ValueError: if the place cannot be geocoded.
    """
    # ── Guard: key must be present before anything else ──────────────────────
    mappls_key = os.environ.get("MAPPLS_KEY", "").strip()
    if not mappls_key:
        raise ValueError("Mappls API key not configured.")

    # ── Fast path: raw "lat,lng" string (sent by frontend after autocomplete) ──
    parts = place.strip().split(',')
    if len(parts) == 2:
        try:
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return (lat, lng)
        except ValueError:
            pass

    # ── Mappls Geocoding REST API ─────────────────────────────────────────────
    try:
        # Exchange static key for OAuth access_token
        tok_resp = _req.post(
            "https://outpost.mappls.com/api/security/oauth/token",
            data={"grant_type": "client_credentials",
                  "client_id": mappls_key,
                  "client_secret": mappls_key},
            timeout=10,
        )
        if not tok_resp.ok:
            raise ValueError(f"Mappls auth failed {tok_resp.status_code}: {tok_resp.text[:100]}")
        access_token = tok_resp.json()["access_token"]

        resp = _req.get(
            "https://atlas.mappls.com/api/places/geocode",
            params={"address": place, "region": "IND", "access_token": access_token},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Mappls returns results under 'copResults' (single) or 'results' (list)
        cop = data.get("copResults") or {}
        if not cop and data.get("results"):
            cop = data["results"][0]

        # ── Guard: empty response or missing latitude ─────────────────────────
        if not cop or "latitude" not in cop:
            raise ValueError(f"Could not locate: {place}")

        lat = float(cop.get("latitude")  or cop.get("lat", 0))
        lon = float(cop.get("longitude") or cop.get("lng") or cop.get("lon", 0))

        if lat == 0.0 and lon == 0.0:
            raise ValueError(f"Could not locate: {place}")

        print(f"  📍 Mappls geocode: '{place}' → ({lat:.5f}, {lon:.5f})")
        return (lat, lon)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not geocode '{place}': {exc}") from exc



# =============================================================================
# Step 2 — Distance guard
# =============================================================================

def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# =============================================================================
# Step 3 — osmnx graph download (LRU-cached, osmnx>=2.0 keyword bbox)
# =============================================================================

@functools.lru_cache(maxsize=8)
def _cached_graph_from_bbox(north: float, south: float,
                            east: float, west: float,
                            network_type: str):
    """LRU-cached osmnx download.  Caller rounds bbox to 2 dp."""
    print(f"  📡 Downloading OSM graph: "
          f"bbox=({north},{south},{east},{west}), type={network_type}")
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),   # osmnx >= 2.0: (left, bottom, right, top)
        network_type=network_type,
    )
    return G


def build_road_graph(
        origin_ll:  Tuple[float, float],
        dest_ll:    Tuple[float, float],
        vehicle:    str,
        buffer_m:   float = 500.0,
):
    """
    Download the osmnx road graph for the bbox containing both points
    plus a buffer.

    Returns:
        ``(G, (north, south, east, west))``
    """
    lat1, lon1 = origin_ll
    lat2, lon2 = dest_ll

    # ≈ 111 km per degree latitude
    avg_lat = (lat1 + lat2) / 2
    lat_buf = buffer_m / 111_000
    lon_buf = buffer_m / (111_000 * max(math.cos(math.radians(avg_lat)), 0.01))

    north = round(max(lat1, lat2) + lat_buf, 2)
    south = round(min(lat1, lat2) - lat_buf, 2)
    east  = round(max(lon1, lon2) + lon_buf, 2)
    west  = round(min(lon1, lon2) - lon_buf, 2)

    net_type = _NETWORK_TYPE.get(vehicle, 'drive')

    try:
        G = _cached_graph_from_bbox(north, south, east, west, net_type)
    except Exception:
        # Fallback to 'all' if requested type yields nothing
        if net_type != 'all':
            print(f"  ⚠️  network_type='{net_type}' failed — falling back to 'all'")
            G = _cached_graph_from_bbox(north, south, east, west, 'all')
        else:
            raise

    return G, (north, south, east, west)


# =============================================================================
# Step 4 — Sentinel-2 imagery (graceful fallback)
# =============================================================================

def fetch_satellite(
        bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, bool]:
    """
    Fetch Sentinel-2 RGB for *bbox*.

    Returns:
        ``(rgb_512x512, ml_active)`` — ml_active is False if dummy image.
    """
    north, south, east, west = bbox

    if not _SENTINEL_OK:
        print("  ⚠️  GEE not available — using dummy image")
        return np.zeros((512, 512, 3), dtype=np.uint8), False

    try:
        end_date   = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=90)).strftime('%Y-%m-%d')

        conn = SentinelConnector()
        rgb, meta = conn.fetch(
            bbox=[west, south, east, north],   # lon_min, lat_min, lon_max, lat_max
            start_date=start_date,
            end_date=end_date,
            cloud_pct=20,
        )
        print(f"  🛰️  Sentinel-2: {meta.get('image_count', '?')} images composited")
        return rgb, True

    except Exception as exc:
        print(f"  ⚠️  Sentinel-2 failed: {exc} — using dummy image")
        return np.zeros((512, 512, 3), dtype=np.uint8), False


# =============================================================================
# Step 5 — ML analysis (M1 width + M2 surface)
# =============================================================================

def run_ml_analysis(
        rgb:       np.ndarray,
        road_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns ``(surface_map, width_m)`` — both (H, W) arrays.

    ``surface_map`` is a str-object array with values
    ``'paved'`` / ``'unpaved'`` / ``'damaged'`` / ``''``.
    ``width_m`` is float32 road widths in metres.
    """
    H, W = road_mask.shape[:2]

    if not _ML_OK:
        return (np.full((H, W), '', dtype=object),
                np.zeros((H, W), dtype=np.float32))

    wr   = RoadWidthEstimator()
    wres = wr.analyse(road_mask)

    clf  = RoadTypeClassifier()
    tres = clf.predict(rgb, road_mask, width_result=wres)

    surface_map = tres['surface_map']       # (H, W) str object array
    width_m     = wres.width_m              # (H, W) float32

    print(f"  🔬 ML: dominant={tres['summary']['dominant_type']}, "
          f"mean_w={wres.summary_stats['mean_m']:.1f}m")
    return surface_map, width_m


# =============================================================================
# Step 6 — Paint ML onto osmnx edges
# =============================================================================

def _latlon_to_pixel(lat: float, lon: float,
                     bbox: Tuple[float, float, float, float],
                     size: int = 512) -> Tuple[int, int]:
    """Map lat/lon → pixel ``(row, col)`` on *size × size* raster."""
    north, south, east, west = bbox
    col = int((lon - west) / max(east - west, 1e-9) * (size - 1))
    row = int((north - lat) / max(north - south, 1e-9) * (size - 1))
    return (max(0, min(size - 1, row)),
            max(0, min(size - 1, col)))


def _sample_road_pixel(
        surface_map: np.ndarray,
        width_m:     np.ndarray,
        row: int, col: int,
        radius: int = 3,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Sample surface label and width at ``(row, col)``.

    If the exact pixel is a non-road background pixel, searches within a
    ``±radius`` square for the nearest road pixel (fix 1).  Returns
    ``(None, None)`` only if no road pixel exists within the radius.
    """
    H, W = surface_map.shape[:2]

    # ── Check exact pixel first ───────────────────────────────────────────────
    if 0 <= row < H and 0 <= col < W:
        s = str(surface_map[row, col])
        if s and s != '':
            return s, float(width_m[row, col])

    # ── Search ±radius square for nearest road pixel ──────────────────────────
    best_dist = float('inf')
    best_s: Optional[str]   = None
    best_w: Optional[float] = None

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < H and 0 <= nc < W:
                sv = str(surface_map[nr, nc])
                if sv and sv != '':
                    d2 = dr * dr + dc * dc
                    if d2 < best_dist:
                        best_dist = d2
                        best_s = sv
                        best_w = float(width_m[nr, nc])

    return best_s, best_w


def paint_ml_on_edges(
        G,
        surface_map: np.ndarray,
        width_m:     np.ndarray,
        bbox:        Tuple[float, float, float, float],
        vehicle:     str,
):
    """
    For every osmnx edge, sample ML surface/width at ~10 points along
    the geometry and assign ``ml_surface``, ``ml_width_m``, ``passable``
    attributes.
    """
    min_w = _NAV_MIN_WIDTH.get(vehicle, 3.0)

    for u, v, key, data in G.edges(keys=True, data=True):
        # ── Get edge geometry ─────────────────────────────────────────────────
        if 'geometry' in data:
            coords = list(data['geometry'].coords)     # [(lon, lat), ...]
        else:
            u_d, v_d = G.nodes[u], G.nodes[v]
            coords = [(u_d['x'], u_d['y']), (v_d['x'], v_d['y'])]

        # ── Sample ~10 evenly-spaced points ───────────────────────────────────
        n_pts   = min(max(len(coords), 2), 10)
        indices = np.linspace(0, len(coords) - 1, n_pts, dtype=int)

        surfaces: List[str]   = []
        widths:   List[float] = []

        for idx in indices:
            lon, lat = coords[idx]
            row, col = _latlon_to_pixel(lat, lon, bbox)
            s, w = _sample_road_pixel(surface_map, width_m, row, col, radius=3)
            if s is not None:
                surfaces.append(s)
            if w is not None:
                widths.append(w)

        # ── Assign attributes ─────────────────────────────────────────────────
        data['ml_surface']  = (Counter(surfaces).most_common(1)[0][0]
                               if surfaces else 'unpaved')
        data['ml_width_m']  = float(np.mean(widths)) if widths else 3.0
        data['passable']    = data['ml_width_m'] >= min_w

    return G


# =============================================================================
# Step 7 — Route computation
# =============================================================================

def _edge_length(data: dict) -> float:
    return float(data.get('length', 0.0))


def _parse_maxspeed(val) -> Optional[float]:
    """Parse osmnx ``maxspeed`` attribute → km/h float (or None)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        for v in val:
            r = _parse_maxspeed(v)
            if r is not None:
                return r
        return None
    if isinstance(val, str):
        s = val.strip().lower().replace('km/h', '').replace('kph', '')
        if 'mph' in s:
            s = s.replace('mph', '').strip()
            try:
                return float(s) * 1.609
            except ValueError:
                return None
        try:
            return float(s.strip())
        except ValueError:
            return None
    return None


def _highway_type(data: dict) -> str:
    hw = data.get('highway', '')
    if isinstance(hw, list):
        hw = hw[0] if hw else ''
    return str(hw)


def _edge_speed_kmh(data: dict, vehicle: str) -> float:
    """
    Effective edge speed capped by vehicle type (fix 5).

    ``speed = min(osm_or_default_speed, vehicle_cap)``
    """
    cap = _VEHICLE_SPEED_CAP_KMH.get(vehicle, 80)

    # Try OSM maxspeed tag
    ms = _parse_maxspeed(data.get('maxspeed'))
    if ms is not None and ms > 0:
        return min(ms, cap)

    # Highway-type default
    hw = _highway_type(data)
    default = _SPEED_DEFAULTS_KMH.get(hw, 20)
    return min(default, cap)


def compute_route(G, orig_node, dest_node, weight_key: str):
    """
    Dijkstra shortest path.  Impassable edges have weight set to a very
    large number so they're avoided but not hard-blocked (graceful
    degradation).

    Returns list of node IDs or ``None``.
    """
    try:
        return nx.shortest_path(G, orig_node, dest_node, weight=weight_key)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# =============================================================================
# Step 8 — Build route response
# =============================================================================

def build_route_info(G, node_path: List[int], vehicle: str) -> dict:
    """
    Walk *node_path* and assemble distance, time, polyline, segments,
    and damage warnings.
    """
    total_dist_m = 0.0
    total_time_s = 0.0
    good_m       = 0.0
    unpaved_m    = 0.0
    damaged_m    = 0.0
    
    # Track damage per road name to aggregate warnings
    damage_by_road: Dict[str, float] = {}
    polyline:  List[List[float]] = []

    for i, (u, v) in enumerate(zip(node_path[:-1], node_path[1:])):
        if not G.has_edge(u, v):
            continue

        edges    = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get('length', float('inf')))
        data     = edges[best_key]
        length_m = _edge_length(data)

        # ── Polyline (include full geometry) ──────────────────────────────────
        if 'geometry' in data:
            coords = list(data['geometry'].coords)    # [(lon, lat), ...]
            # Orient from u → v
            u_d = G.nodes[u]
            first = coords[0]
            if (abs(first[0] - u_d['x']) + abs(first[1] - u_d['y']) >
                abs(coords[-1][0] - u_d['x']) + abs(coords[-1][1] - u_d['y'])):
                coords = coords[::-1]
            start_idx = 1 if i > 0 else 0      # skip duplicate junction point
            for lon, lat in coords[start_idx:]:
                polyline.append([lat, lon])
        else:
            nd = G.nodes[v] if i > 0 else G.nodes[u]
            if i == 0:
                polyline.append([G.nodes[u]['y'], G.nodes[u]['x']])
            polyline.append([G.nodes[v]['y'], G.nodes[v]['x']])

        # ── Accumulate stats ──────────────────────────────────────────────────
        total_dist_m += length_m

        speed_kmh = _edge_speed_kmh(data, vehicle)
        speed_ms  = speed_kmh / 3.6
        if speed_ms > 0:
            total_time_s += length_m / speed_ms

        surface = data.get('ml_surface', 'unpaved')
        if surface == 'paved':
            good_m += length_m
        elif surface == 'damaged':
            damaged_m += length_m
            name = data.get('name', '')
            if isinstance(name, list):
                name = name[0] if name else ''
            if not name:
                name = 'unnamed road'
            
            damage_by_road[name] = damage_by_road.get(name, 0.0) + length_m
        else:
            unpaved_m += length_m

    # Add last node if polyline is still empty
    if not polyline and node_path:
        nd = G.nodes[node_path[0]]
        polyline.append([nd['y'], nd['x']])

    # ── Format warnings ───────────────────────────────────────────────────────
    warnings: List[str] = []
    # Sort by longest damaged stretch first
    for name, length in sorted(damage_by_road.items(), key=lambda x: x[1], reverse=True):
        if length >= 1000:
            dist_str = f"{length/1000:.1f} km"
        else:
            dist_str = f"{int(length)} m"
        warnings.append(f"Damaged surface on {name} (~{dist_str} total)")

    # ── Count road segments and junctions ────────────────────────────────────
    road_count     = len(node_path) - 1 if len(node_path) > 1 else 0
    junction_count = sum(
        1 for n in node_path[1:-1]
        if G.degree(n) > 2
    )

    # ── Time rounding: 1 dp below 10 min, integer above ───────────────────────
    raw_min = total_time_s / 60
    if raw_min < 10:
        est_min = round(raw_min, 1)
    else:
        est_min = round(raw_min)

    return {
        'distance_km':       round(total_dist_m / 1000, 1),
        'estimated_minutes': est_min,
        'polyline':          polyline,
        'segments': {
            'good_km':    round(good_m / 1000, 2),
            'unpaved_km': round(unpaved_m / 1000, 2),
            'damaged_km': round(damaged_m / 1000, 2),
        },
        'warnings':      warnings,
        'road_count':    road_count,
        'junction_count': junction_count,
    }


# =============================================================================
# Orchestrator
# =============================================================================

_IMPASSABLE_COST = 1_000_000.0   # 1 000 km equivalent

# Full-result cache: (origin, dest, vehicle) → response dict
_navigate_cache: dict = {}

def navigate(origin: str, destination: str, vehicle: str = 'car') -> dict:
    """
    Full navigate pipeline.

    Args:
        origin:      Place name or ``'lat,lng'`` string.
        destination: Place name or ``'lat,lng'`` string.
        vehicle:     ``'car'`` | ``'motorcycle'`` | ``'truck'`` | ``'pedestrian'``

    Returns:
        dict ready for JSON serialisation with ``fastest``, ``safest``,
        ``origin_coords``, ``destination_coords``, ``ml_active`` keys.

    Raises:
        ValueError: geocoding failure, route too long, or no route found.
        RuntimeError: missing dependencies.
    """
    if not _NAV_DEPS_OK:
        raise RuntimeError(
            "Navigation dependencies (osmnx, geopy, networkx) not installed.")

    # ── 1. Geocode ────────────────────────────────────────────────────────────
    print(f"🧭 Navigate: '{origin}' → '{destination}' ({vehicle})")
    origin_ll = geocode(origin)
    dest_ll   = geocode(destination)
    print(f"  📍 Origin : {origin_ll}")
    print(f"  📍 Dest   : {dest_ll}")

    # ── Result cache ───────────────────────────────────────────────────────────────────
    cache_key = f"{origin_ll}|{dest_ll}|{vehicle}"
    if cache_key in _navigate_cache:
        print("  ⚡ Returning cached result")
        return _navigate_cache[cache_key]

    # ── 2. Distance guard ───────────────────────────────────────────────────────────────────
    dist_km = _haversine_km(origin_ll[0], origin_ll[1],
                            dest_ll[0],   dest_ll[1])
    print(f"  📏 Straight-line: {dist_km:.1f} km")
    if dist_km > MAX_ROUTE_KM:
        raise ValueError(
            "Route too long — keep both points within 50 km of each other.")

    # ── 3 + 4. Parallel: osmnx graph download  +  Sentinel-2 ─────────────────
    # build_road_graph is LRU-cached so it's fast on repeat calls.
    # fetch_satellite is GEE I/O — fire it in a thread to overlap with graph.
    print("  ⏳ Fetching OSM graph + Sentinel-2 in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
        _fut_graph     = _pool.submit(build_road_graph, origin_ll, dest_ll, vehicle)
        G, bbox        = _fut_graph.result()          # wait for bbox first
        _fut_sat       = _pool.submit(fetch_satellite, bbox)   # then fire GEE
        rgb, ml_active = _fut_sat.result()

    print(f"  🗺️  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


    # ── 5. OSM raster mask for ML ─────────────────────────────────────────────
    north, south, east, west = bbox
    road_mask = np.zeros((512, 512), dtype=np.uint8)
    if _OSM_MASK_OK:
        try:
            osm = OSMConnector()
            osm_result = osm.fetch(
                lat_min=south, lon_min=west,
                lat_max=north, lon_max=east,
                road_types=['all'],
            )
            road_mask = osm_result['mask']
            print(f"  🛤️  OSM mask: {int((road_mask > 0).sum())} road px")
        except Exception as exc:
            print(f"  ⚠️  OSM mask failed: {exc}")

    # ── 6. ML analysis ────────────────────────────────────────────────────────
    surface_map, width_m_map = run_ml_analysis(rgb, road_mask)

    # ── 7. Paint ML onto edges ────────────────────────────────────────────────
    G = paint_ml_on_edges(G, surface_map, width_m_map, bbox, vehicle)

    # ── 8. Set routing weights ────────────────────────────────────────────────
    for u, v, key, data in G.edges(keys=True, data=True):
        length  = _edge_length(data)
        surface = data.get('ml_surface', 'unpaved')
        passable = data.get('passable', True)

        # Fastest: raw length (huge penalty if impassable)
        data['weight_fastest'] = (length if passable
                                  else _IMPASSABLE_COST + length)

        # Safest: damage-weighted (huge penalty if impassable)
        mult = _DAMAGE_MULTIPLIER.get(surface, 1.4)
        data['weight_safest'] = (length * mult if passable
                                 else _IMPASSABLE_COST + length * mult)

    # ── 9. Find nearest nodes ─────────────────────────────────────────────────
    orig_node = ox.nearest_nodes(G, origin_ll[1], origin_ll[0])
    dest_node = ox.nearest_nodes(G, dest_ll[1],   dest_ll[0])

    # ── 10. Compute routes ────────────────────────────────────────────────────
    fastest_path = compute_route(G, orig_node, dest_node, 'weight_fastest')
    safest_path  = compute_route(G, orig_node, dest_node, 'weight_safest')

    if fastest_path is None and safest_path is None:
        raise ValueError("No route found between these locations.")

    # If one fails, mirror the other
    if fastest_path is None:
        fastest_path = safest_path
    if safest_path is None:
        safest_path = fastest_path

    # ── 11. Build response ────────────────────────────────────────────────────
    fastest_info = build_route_info(G, fastest_path, vehicle)
    safest_info  = build_route_info(G, safest_path,  vehicle)

    print(f"  ✅ Fastest: {fastest_info['distance_km']} km, "
          f"{fastest_info['estimated_minutes']} min, "
          f"{fastest_info['road_count']} segs, "
          f"{fastest_info['junction_count']} junctions")
    print(f"  ✅ Safest:  {safest_info['distance_km']} km, "
          f"{safest_info['estimated_minutes']} min")

    result = {
        'fastest':            fastest_info,
        'safest':             safest_info,
        'origin_coords':      list(origin_ll),
        'destination_coords': list(dest_ll),
        'ml_active':          ml_active,
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
        },
    }
    _navigate_cache[cache_key] = result
    return result
