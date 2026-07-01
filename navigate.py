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
    
    # Overpass API mirrors — we cycle through these on connection failure.
    _OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.openstreetmap.fr/api",
    ]
    ox.settings.overpass_endpoint = _OVERPASS_ENDPOINTS[0]
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
MAX_ROUTE_KM = 200.0

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
# Shared Mappls OAuth token cache
# =============================================================================

_mappls_token_cache: dict = {"token": None, "expires_at": 0.0, "backoff_until": 0.0}


def _get_mappls_token(client_id: str, client_secret: str) -> str:
    """Exchange Mappls OAuth credentials for an access_token (raw, no cache)."""
    resp = _req.post(
        "https://outpost.mappls.com/api/security/oauth/token",
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_cached_mappls_token() -> str:
    """
    Return a cached Mappls OAuth token, refreshing when it expires.
    Reads MAPPLS_CLIENT_ID + MAPPLS_CLIENT_SECRET (preferred) or falls
    back to MAPPLS_KEY for both if the dedicated vars aren’t set.
    Backs off 60 s on auth failure to avoid hammering the endpoint.
    """
    import time
    c = _mappls_token_cache

    if c["token"] and time.time() < c["expires_at"]:
        return c["token"]

    if time.time() < c["backoff_until"]:
        raise ValueError("Mappls auth in backoff — skipping retry")

    static_key    = os.environ.get("MAPPLS_KEY",           "").strip()
    client_id     = os.environ.get("MAPPLS_CLIENT_ID",    "").strip() or static_key
    client_secret = os.environ.get("MAPPLS_CLIENT_SECRET", "").strip() or static_key

    if not client_id or not client_secret:
        raise ValueError("Mappls credentials not configured.")

    try:
        token = _get_mappls_token(client_id, client_secret)
    except Exception as exc:
        c["backoff_until"] = time.time() + 60
        raise ValueError(f"Mappls token fetch failed: {exc}") from exc

    c["token"]         = token
    c["expires_at"]    = time.time() + 21_000   # 5.8 h
    c["backoff_until"] = 0.0
    print("  🔑 Mappls token refreshed")
    return token


# =============================================================================
# Step 1 — Geocoding
# =============================================================================

def geocode(place: str) -> Tuple[float, float]:
    """
    Convert a place name or ``'lat,lng'`` string to ``(lat, lng)`` floats.

    Priority:
      1. Raw ``'lat,lng'`` string  — fast path (always used after autocomplete)
      2. Mappls Place Search → eLoc → Place Details  — works for POI names
      3. Mappls Geocoding API  — fallback for address-style queries

    Raises:
        ValueError: if the place cannot be geocoded.
    """
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

    # ── Get OAuth token (shared cache) ───────────────────────────────────────
    try:
        token = _get_cached_mappls_token()
    except Exception as exc:
        raise ValueError(f"Mappls auth failed: {exc}") from exc

    # ── Path 2: Place Search → address string → Geocode API ─────────────────
    # eLoc param returns 400; pass placeName+placeAddress as address string instead
    try:
        search_resp = _req.get(
            "https://atlas.mappls.com/api/places/search/json",
            params={"query": place, "access_token": token},
            timeout=8,
        )
        if search_resp.ok:
            raw = (search_resp.json().get("suggestedLocations") or
                   search_resp.json().get("results") or [])
            print(f"  🔍 Place Search: '{place}' → {len(raw)} raw results")
            for loc in raw[:3]:
                place_name = loc.get("placeName") or loc.get("name") or ""
                place_addr = loc.get("placeAddress") or loc.get("address") or ""
                full_addr  = f"{place_name}, {place_addr}".strip(", ") if place_addr else place_name
                if not full_addr:
                    continue
                try:
                    det = _req.get(
                        "https://atlas.mappls.com/api/places/geocode",
                        params={"address": full_addr, "region": "IND", "access_token": token},
                        timeout=5,
                    )
                    print(f"    Geocode '{place_name}' → {det.status_code}: {det.text[:200]}")
                    if det.ok:
                        cop = det.json().get("copResults") or {}
                        if isinstance(cop, list):
                            cop = cop[0] if cop else {}
                        print(f"    copResults keys: {list(cop.keys())}")
                        lat_s = cop.get("latitude")  or cop.get("lat")  or ""
                        lon_s = cop.get("longitude") or cop.get("lng")  or cop.get("lon") or ""
                        print(f"    lat={lat_s!r}, lng={lon_s!r}")
                        lat_f = float(lat_s) if lat_s else 0.0
                        lon_f = float(lon_s) if lon_s else 0.0
                        if lat_f and lon_f:
                            print(f"  📍 Geocode (via Place Search): '{place}' → ({lat_f:.5f}, {lon_f:.5f})")
                            return (lat_f, lon_f)
                except Exception as de:
                    print(f"    Geocode error for '{place_name}': {de}")
    except Exception as exc:
        print(f"  ⚠️  Place Search/Geocode failed: {exc}")

    # ── Path 3: Geocode API by address name (works for some queries) ──────────
    try:
        resp = _req.get(
            "https://atlas.mappls.com/api/places/geocode",
            params={"address": place, "region": "IND", "access_token": token},
            timeout=10,
        )
        print(f"  🔍 Geocode API (address): {resp.status_code} → {resp.text[:200]}")
        if resp.ok:
            data = resp.json()
            cop = data.get("copResults") or {}
            print(f"    copResults keys: {list(cop.keys())}")
            lat_s = cop.get("latitude")  or cop.get("lat")  or ""
            lon_s = cop.get("longitude") or cop.get("lng")  or cop.get("lon") or ""
            print(f"    lat={lat_s!r}, lng={lon_s!r}")
            lat_f = float(lat_s) if lat_s else 0.0
            lon_f = float(lon_s) if lon_s else 0.0
            if lat_f and lon_f:
                print(f"  📍 Geocoding API: '{place}' → ({lat_f:.5f}, {lon_f:.5f})")
                return (lat_f, lon_f)
    except Exception as exc:
        print(f"  ⚠️  Geocoding API failed: {exc}")

    # ── Path 4: External geocoders cascade (Nominatim → Photon → Geoapify) ────
    # Try each in order; all are free, different infra, different rate limits

    # 4a: Nominatim (OSM)
    try:
        print(f"  🌐 Nominatim: '{place}, India'")
        nom = _req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{place}, India", "format": "json",
                    "countrycodes": "in", "limit": "1"},
            headers={"User-Agent": "RoadSense/2.0 (+huggingface.co)"},
            timeout=10,
        )
        print(f"    status={nom.status_code}, body={nom.text[:200]}")
        if nom.ok and nom.json():
            hit = nom.json()[0]
            lat, lon = float(hit["lat"]), float(hit["lon"])
            print(f"  📍 Nominatim: '{place}' → ({lat:.5f}, {lon:.5f})")
            return (lat, lon)
    except Exception as exc:
        print(f"  ⚠️  Nominatim failed: {exc}")

    # 4b: Photon (komoot) — different OSM mirror, no rate-limit header
    try:
        print(f"  🌐 Photon: '{place}'")
        ph = _req.get(
            "https://photon.komoot.io/api/",
            params={"q": f"{place}, India", "limit": "1", "lang": "en"},
            timeout=10,
        )
        print(f"    status={ph.status_code}, body={ph.text[:200]}")
        if ph.ok:
            features = ph.json().get("features", [])
            if features:
                coords = features[0]["geometry"]["coordinates"]
                lat, lon = float(coords[1]), float(coords[0])
                print(f"  📍 Photon: '{place}' → ({lat:.5f}, {lon:.5f})")
                return (lat, lon)
    except Exception as exc:
        print(f"  ⚠️  Photon failed: {exc}")

    # 4c: Geoapify free geocoding (no key needed for /v1/geocode/search)
    try:
        print(f"  🌐 Geoapify: '{place}'")
        geo = _req.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": f"{place}, India", "filter": "countrycode:in",
                    "format": "json", "apiKey": "open"},
            timeout=10,
        )
        print(f"    status={geo.status_code}, body={geo.text[:200]}")
        if geo.ok:
            results = geo.json().get("results", [])
            if results:
                lat = float(results[0].get("lat", 0))
                lon = float(results[0].get("lon", 0))
                if lat and lon:
                    print(f"  📍 Geoapify: '{place}' → ({lat:.5f}, {lon:.5f})")
                    return (lat, lon)
    except Exception as exc:
        print(f"  ⚠️  Geoapify failed: {exc}")

    raise ValueError(f"Could not locate: {place}")




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
    """LRU-cached osmnx download with Overpass endpoint cycling."""
    endpoints = getattr(ox.settings, '_OVERPASS_ENDPOINTS', None)
    if endpoints is None:
        try:
            endpoints = _OVERPASS_ENDPOINTS
        except NameError:
            endpoints = [ox.settings.overpass_endpoint]

    last_err = None
    for ep in endpoints:
        ox.settings.overpass_endpoint = ep
        try:
            print(f"  \U0001F4E1 Downloading OSM graph via {ep}: "
                  f"bbox=({north},{south},{east},{west}), type={network_type}")
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
            )
            return G
        except Exception as e:
            last_err = e
            print(f"  \u26a0\ufe0f  Overpass endpoint {ep} failed: {type(e).__name__}")
            continue
    # All endpoints exhausted
    raise last_err


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


def _osm_surface_label(data: dict):
    """Classify road surface from OSM tags. Returns label or None if unknown."""
    highway = data.get('highway', '')
    if isinstance(highway, list):
        highway = highway[0] if highway else ''
    highway = str(highway)

    surface = data.get('surface', '')
    if isinstance(surface, list):
        surface = surface[0] if surface else ''
    surface = str(surface)

    tracktype = data.get('tracktype', '')
    if isinstance(tracktype, list):
        tracktype = tracktype[0] if tracktype else ''
    tracktype = str(tracktype)

    # Damaged: rough tracks, poor surface tags
    if highway in ('track', 'path', 'bridleway'):
        return 'damaged'
    if tracktype in ('grade3', 'grade4', 'grade5'):
        return 'damaged'
    if surface in ('dirt', 'ground', 'grass', 'mud',
                   'sand', 'gravel', 'compacted'):
        return 'unpaved'
    if surface in ('unpaved', 'fine_gravel', 'pebblestone'):
        return 'unpaved'

    # Paved: confirmed good surfaces
    if surface in ('asphalt', 'concrete', 'paved',
                   'concrete:plates', 'concrete:lanes'):
        return 'paved'
    if highway in ('motorway', 'trunk', 'primary',
                   'motorway_link', 'trunk_link', 'primary_link'):
        return 'paved'

    # No OSM tag available
    return None


def paint_ml_on_edges(
        G,
        surface_map: np.ndarray,
        width_m:     np.ndarray,
        bbox:        Tuple[float, float, float, float],
        vehicle:     str,
):
    """
    For every osmnx edge, first try OSM tag-based surface classification.
    If no OSM tag, fall back to ML pixel sampling at ~10 points along
    the geometry.  Assigns ``ml_surface``, ``ml_width_m``, ``passable``,
    ``surface_src`` ('osm' or 'ml') attributes.
    """
    min_w = _NAV_MIN_WIDTH.get(vehicle, 3.0)
    osm_count = 0
    ml_count  = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        # ── 1. Try OSM tags first (primary source) ──────────────────────────
        osm_label = _osm_surface_label(data)
        if osm_label is not None:
            data['ml_surface']  = osm_label
            data['ml_width_m']  = 3.5 if osm_label == 'paved' else 3.0
            data['passable']    = True  # OSM roads are mapped → passable
            data['surface_src'] = 'osm'
            osm_count += 1
            continue

        # ── 2. Fall back to ML pixel sampling ────────────────────────────────
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
        else:
            u_d, v_d = G.nodes[u], G.nodes[v]
            coords = [(u_d['x'], u_d['y']), (v_d['x'], v_d['y'])]

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

        data['ml_surface']  = (Counter(surfaces).most_common(1)[0][0]
                               if surfaces else 'unpaved')
        data['ml_width_m']  = float(np.mean(widths)) if widths else 3.0
        data['passable']    = data['ml_width_m'] >= min_w
        data['surface_src'] = 'ml'
        ml_count += 1

    # Store counts on graph for later use in response
    G.graph['_osm_surface_count'] = osm_count
    G.graph['_ml_surface_count']  = ml_count

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
# Step 7 — Fetch traffic multipliers from Mappls
# =============================================================================

_TRAFFIC_MAP: Dict[str, float] = {
    "free_flow":  1.0,
    "normal":     1.0,
    "light":      1.2,
    "moderate":   1.5,
    "heavy":      2.0,
    "severe":     2.5,
    "standstill": 2.5,
}


def fetch_traffic_multipliers(
        polyline: List[Tuple[float, float]],
        vehicle:  str,
) -> List[float]:
    """
    Query the Mappls live traffic flow API for a route polyline.

    Samples the polyline to ≤20 points, then maps congestion labels
    to time multipliers.  Always returns a list of floats ≥1.0.
    Returns ``[1.0]`` on any failure (never crashes the route).
    """
    # Pedestrians are unaffected by vehicle traffic congestion
    if vehicle == 'pedestrian' or not polyline:
        return [1.0]

    key = os.environ.get("MAPPLS_KEY", "").strip()
    if not key:
        return [1.0]

    try:
        token = _get_cached_mappls_token()

        # Sample ≤20 waypoints to stay within API limits
        N       = max(1, len(polyline) // 20)
        sampled = polyline[::N]
        path    = "|".join(f"{lat},{lng}" for lat, lng in sampled)

        resp = _req.get(
            "https://atlas.mappls.com/api/live_traffic/flow",
            params={"path": path, "access_token": token},
            timeout=8,
        )

        if not resp.ok:
            print(f"  ⚠️  Traffic API {resp.status_code} — using no-traffic multipliers")
            return [1.0]

        data = resp.json()

        # Parse whichever response shape Mappls returns
        segments = (
            data.get("traffic_data")
            or data.get("segments")
            or data.get("results")
            or []
        )

        if not segments:
            # Maybe top-level has a single condition key
            cond = (
                data.get("traffic_condition")
                or data.get("congestion")
                or data.get("status")
                or ""
            ).lower()
            m = _TRAFFIC_MAP.get(cond, 1.0)
            print(f"  🚦 Traffic: single condition='{cond}' → {m}x")
            return [m]

        mults = []
        for seg in segments:
            cond = (
                seg.get("traffic_condition")
                or seg.get("congestion")
                or seg.get("status")
                or seg.get("flow")
                or ""
            ).lower()
            mults.append(_TRAFFIC_MAP.get(cond, 1.0))

        avg = sum(mults) / len(mults) if mults else 1.0
        print(f"  🚦 Traffic: {len(mults)} segments, avg_mult={avg:.2f}")
        return mults if mults else [1.0]

    except Exception as exc:
        print(f"  ⚠️  Traffic fetch failed ({exc}) — continuing without traffic data")
        return [1.0]


# =============================================================================
# Step 8 — Build route response
# =============================================================================

def _compute_surface_source(G, node_path) -> str:
    """Determine if a route is primarily OSM-sourced, ML-sourced, or mixed."""
    osm_edges = 0
    ml_edges  = 0
    
    for u, v in zip(node_path[:-1], node_path[1:]):
        if not G.has_edge(u, v):
            continue
        edges = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get('length', float('inf')))
        src = edges[best_key].get('surface_src', 'ml')
        if src == 'osm':
            osm_edges += 1
        else:
            ml_edges += 1
            
    total = osm_edges + ml_edges
    if total == 0:
        return 'unknown'
        
    osm_ratio = osm_edges / total
    if osm_ratio >= 0.7:
        return 'osm'
    elif osm_ratio <= 0.3:
        return 'ml'
    else:
        return 'mixed'


def _build_colored_segments(G, node_path: List[int]) -> List[dict]:
    """
    Walk *node_path* and group consecutive edges by surface type
    into coordinate runs for colour-coded polyline rendering.

    Returns a list of dicts:
        [{ 'surface': 'paved', 'coords': [[lat,lng], ...] }, ...]
    """
    segments: List[dict] = []
    current_surface: Optional[str] = None
    current_coords: List[List[float]] = []

    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        if not G.has_edge(u, v):
            continue

        edges    = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get('length', float('inf')))
        data     = edges[best_key]
        surface  = data.get('ml_surface', 'unpaved')

        u_coord = [G.nodes[u]['y'], G.nodes[u]['x']]
        v_coord = [G.nodes[v]['y'], G.nodes[v]['x']]

        if surface != current_surface:
            # Flush previous run
            if current_coords:
                current_coords.append(u_coord)          # bridge to new segment
                segments.append({
                    'surface': current_surface,
                    'coords':  current_coords,
                })
            current_surface = surface
            current_coords  = [u_coord, v_coord]
        else:
            current_coords.append(v_coord)

    # Flush final run
    if current_coords and current_surface:
        segments.append({
            'surface': current_surface,
            'coords':  current_coords,
        })

    return segments


def build_route_info(
        G,
        node_path: List[int],
        vehicle:   str,
        traffic_mults: Optional[List[float]] = None,
) -> dict:
    """
    Walk *node_path* and assemble distance, time, polyline, segments,
    damage warnings, and traffic status.

    *traffic_mults* is a list of per-segment time multipliers produced by
    :func:`fetch_traffic_multipliers`.  Pass ``None`` to have this function
    call it internally (used when traffic data isn’t pre-fetched).
    """
    total_dist_m = 0.0
    total_time_s = 0.0
    base_time_s  = 0.0          # time without traffic, for delay calc
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

        # ── Polyline (include full geometry) ───────────────────────────────────
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

        # ── Accumulate stats ──────────────────────────────────────────────────────
        total_dist_m += length_m

        speed_kmh = _edge_speed_kmh(data, vehicle)
        speed_ms  = speed_kmh / 3.6
        if speed_ms > 0:
            edge_base = length_m / speed_ms
            base_time_s += edge_base
            # Traffic multiplier mapped proportionally across mults list
            # (applied after polyline is built, so we use edge index i)
            # We defer the actual mult lookup until mults are available.
            total_time_s += edge_base   # placeholder; corrected below

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

    # ── Fetch / apply traffic multipliers ───────────────────────────────────────
    if traffic_mults is None:
        traffic_mults = fetch_traffic_multipliers(
            [(pt[0], pt[1]) for pt in polyline], vehicle
        )

    # Re-apply edge times with traffic multipliers
    n_edges = max(len(node_path) - 1, 1)
    total_time_s = 0.0
    for i, (u, v) in enumerate(zip(node_path[:-1], node_path[1:])):
        if not G.has_edge(u, v):
            continue
        edges    = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get('length', float('inf')))
        data     = edges[best_key]
        length_m = _edge_length(data)
        speed_kmh = _edge_speed_kmh(data, vehicle)
        speed_ms  = speed_kmh / 3.6
        if speed_ms > 0:
            edge_base = length_m / speed_ms
            t_idx = int(i / n_edges * len(traffic_mults))
            t_idx = min(t_idx, len(traffic_mults) - 1)
            total_time_s += edge_base * traffic_mults[t_idx]

    # ── Traffic summary ────────────────────────────────────────────────────────────
    avg_mult = (sum(traffic_mults) / len(traffic_mults)) if traffic_mults else 1.0
    if avg_mult <= 1.1:
        traffic_status = "free_flow"
    elif avg_mult <= 1.3:
        traffic_status = "light"
    elif avg_mult <= 1.7:
        traffic_status = "moderate"
    elif avg_mult <= 2.1:
        traffic_status = "heavy"
    else:
        traffic_status = "severe"
    traffic_delay_minutes = round((total_time_s - base_time_s) / 60)

    # ── Format warnings ──────────────────────────────────────────────────────────────
    warnings: List[str] = []
    for name, length in sorted(damage_by_road.items(), key=lambda x: x[1], reverse=True):
        if length >= 1000:
            dist_str = f"{length/1000:.1f} km"
        else:
            dist_str = f"{int(length)} m"
        warnings.append(f"Damaged surface on {name} (~{dist_str} total)")

    # ── Count road segments and junctions ───────────────────────────────────
    road_count     = len(node_path) - 1 if len(node_path) > 1 else 0
    junction_count = sum(1 for n in node_path[1:-1] if G.degree(n) > 2)

    # ── Time rounding: 1 dp below 10 min, integer above ───────────────────────
    raw_min = total_time_s / 60
    if raw_min < 10:
        est_min = round(raw_min, 1)
    else:
        est_min = round(raw_min)

    # ── Build colored segments for surface inspection ─────────────────────────
    colored_segments = _build_colored_segments(G, node_path)

    return {
        'distance_km':          round(total_dist_m / 1000, 1),
        'estimated_minutes':    est_min,
        'polyline':             polyline,
        'segments': {
            'good_km':    round(good_m / 1000, 2),
            'unpaved_km': round(unpaved_m / 1000, 2),
            'damaged_km': round(damaged_m / 1000, 2),
        },
        'warnings':             warnings,
        'road_count':           road_count,
        'junction_count':       junction_count,
        'traffic_status':       traffic_status,
        'traffic_delay_minutes': traffic_delay_minutes,
        'surface_source':       _compute_surface_source(G, node_path),
        'colored_segments':     colored_segments,
    }


# =============================================================================
# Edge-based coordinate snapping
# =============================================================================

def _snap_to_graph(G, lat: float, lng: float):
    """
    Snap a lat/lng coordinate to the nearest point on the road network.
    Uses nearest_edges() for precision, then picks the endpoint node
    of that edge that is closer to the coordinate.
    Falls back to nearest_nodes() if nearest_edges fails.

    Returns:
        ``(node_id, snap_gap_metres)`` — gap is the haversine distance
        between the actual coordinate and the snapped node.
    """
    try:
        # ox.nearest_edges returns (u, v, key) of the closest edge
        u, v, key = ox.nearest_edges(G, lng, lat)  # X=lng, Y=lat

        # Pick the endpoint node closest to our coordinate
        u_d, v_d = G.nodes[u], G.nodes[v]
        dist_u = ((u_d['y'] - lat)**2 + (u_d['x'] - lng)**2)**0.5
        dist_v = ((v_d['y'] - lat)**2 + (v_d['x'] - lng)**2)**0.5

        node = u if dist_u <= dist_v else v
    except Exception:
        # Fallback to nearest_nodes if nearest_edges fails
        node = ox.nearest_nodes(G, lng, lat)

    snapped_lat = G.nodes[node]['y']
    snapped_lng = G.nodes[node]['x']
    snap_gap_m  = _haversine_km(lat, lng, snapped_lat, snapped_lng) * 1000
    return node, snap_gap_m


_GAP_THRESHOLD_M = 300   # only expand bbox if snap gap exceeds this


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
            "Route too long — keep both points within 200 km of each other.")

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

        # Fastest: time-based weight (length / speed) so it minimizes travel TIME
        highway = data.get('highway', 'residential')
        if isinstance(highway, list):
            highway = highway[0]
        speed_kmh = _SPEED_DEFAULTS_KMH.get(highway, 20)
        speed_kmh = min(speed_kmh, _VEHICLE_SPEED_CAP_KMH.get(vehicle, 60))
        speed_ms  = speed_kmh / 3.6
        data['weight_fastest'] = ((length / speed_ms) if passable
                                  else _IMPASSABLE_COST + length)

        # Safest: damage-weighted (huge penalty if impassable)
        mult = _DAMAGE_MULTIPLIER.get(surface, 1.4)
        data['weight_safest'] = (length * mult if passable
                                 else _IMPASSABLE_COST + length * mult)

    # ── 9. Snap origin/destination to nearest edges ───────────────────────────
    orig_node, orig_gap = _snap_to_graph(G, origin_ll[0], origin_ll[1])
    dest_node, dest_gap = _snap_to_graph(G, dest_ll[0],   dest_ll[1])
    print(f"  📍 Snap gaps: origin={orig_gap:.0f}m, dest={dest_gap:.0f}m")

    # ── 9b. Expand bbox toward sparse endpoints if gap is large ───────────
    def _try_expand(G_base, coord_ll, current_node, current_gap, label):
        """Expand bbox toward coord_ll, re-download with network_type='all',
        and merge into G_base if it reduces the snap gap."""
        if current_gap <= _GAP_THRESHOLD_M:
            return G_base, current_node, current_gap
        try:
            n, s, e, w = bbox
            mid_lat = (n + s) / 2
            mid_lng = (e + w) / 2
            # Expand 0.005° (~550m) past the coordinate in its direction
            if coord_ll[0] > mid_lat:
                n = max(n, coord_ll[0] + 0.005)
            else:
                s = min(s, coord_ll[0] - 0.005)
            if coord_ll[1] > mid_lng:
                e = max(e, coord_ll[1] + 0.005)
            else:
                w = min(w, coord_ll[1] - 0.005)
            G_exp = ox.graph_from_bbox(bbox=(n, s, e, w),
                                       network_type='all', retain_all=True)
            new_node, new_gap = _snap_to_graph(G_exp, coord_ll[0], coord_ll[1])
            if new_gap < current_gap * 0.6:
                G_merged = nx.compose(G_base, G_exp)
                print(f"  📍 Expansion reduced {label} gap: "
                      f"{current_gap:.0f}m → {new_gap:.0f}m")
                return G_merged, new_node, new_gap
            else:
                print(f"  📍 Expansion didn't help {label} "
                      f"({new_gap:.0f}m), keeping original")
        except Exception as exc:
            print(f"  ⚠️ Bbox expansion for {label} failed: {exc}")
        return G_base, current_node, current_gap

    G, dest_node, dest_gap = _try_expand(G, dest_ll, dest_node, dest_gap, 'dest')
    G, orig_node, orig_gap = _try_expand(G, origin_ll, orig_node, orig_gap, 'origin')

    # ── 9c. Set weights on any new edges from expansion ───────────────────
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'weight_fastest' not in data:
            length  = _edge_length(data)
            surface = data.get('ml_surface', 'unpaved')
            passable = data.get('passable', True)
            highway = data.get('highway', 'residential')
            if isinstance(highway, list):
                highway = highway[0]
            speed_kmh = _SPEED_DEFAULTS_KMH.get(highway, 20)
            speed_kmh = min(speed_kmh, _VEHICLE_SPEED_CAP_KMH.get(vehicle, 60))
            speed_ms  = speed_kmh / 3.6
            data['weight_fastest'] = ((length / speed_ms) if passable
                                      else _IMPASSABLE_COST + length)
            mult = _DAMAGE_MULTIPLIER.get(surface, 1.4)
            data['weight_safest'] = (length * mult if passable
                                     else _IMPASSABLE_COST + length * mult)

    if orig_node not in G.nodes:
        raise ValueError("Could not snap origin to road network.")
    if dest_node not in G.nodes:
        raise ValueError("Could not snap destination to road network.")
    if orig_node == dest_node and _haversine_km(*origin_ll, *dest_ll) > 0.5:
        raise ValueError(
            "Origin and destination snapped to the same road node — "
            "try more specific addresses or points further apart.")

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

    # ── 11. Traffic fetch for both routes in parallel ─────────────────────────
    def _get_polyline_pts(path):
        """Quick polyline extraction for traffic sampling (node coords only)."""
        pts = []
        for n in path:
            nd = G.nodes[n]
            pts.append((nd['y'], nd['x']))
        return pts

    print("  🚦 Fetching traffic data for both routes in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _tpool:
        _ft_fast = _tpool.submit(
            fetch_traffic_multipliers,
            _get_polyline_pts(fastest_path), vehicle
        )
        _ft_safe = _tpool.submit(
            fetch_traffic_multipliers,
            _get_polyline_pts(safest_path), vehicle
        )
        fast_traffic = _ft_fast.result()
        safe_traffic = _ft_safe.result()

    # ── 12. Build response (traffic already fetched) ──────────────────────────
    fastest_info = build_route_info(G, fastest_path, vehicle, traffic_mults=fast_traffic)
    safest_info  = build_route_info(G, safest_path,  vehicle, traffic_mults=safe_traffic)

    # ── 12b. Enforce label invariant: "fastest" must have fewer minutes ────────
    if safest_info['estimated_minutes'] < fastest_info['estimated_minutes']:
        fastest_info, safest_info = safest_info, fastest_info
        print("  🔄 Swapped: safest was actually faster — labels corrected")

    # If fastest has more damage but time diff ≤ 5 min, prefer the safer option
    fast_damaged = fastest_info['segments']['damaged_km']
    safe_damaged = safest_info['segments']['damaged_km']
    time_diff = safest_info['estimated_minutes'] - fastest_info['estimated_minutes']
    if fast_damaged > safe_damaged and time_diff <= 5:
        fastest_info, safest_info = safest_info, fastest_info
        print(f"  🔄 Swapped: fastest had more damage ({fast_damaged:.1f} vs {safe_damaged:.1f} km) "
              f"with only {time_diff} min advantage — preferring safer option")

    print(f"  ✅ Fastest: {fastest_info['distance_km']} km, "
          f"{fastest_info['estimated_minutes']} min "
          f"[{fastest_info['traffic_status']}], "
          f"{fastest_info['road_count']} segs, "
          f"{fastest_info['junction_count']} junctions")
    print(f"  ✅ Safest:  {safest_info['distance_km']} km, "
          f"{safest_info['estimated_minutes']} min "
          f"[{safest_info['traffic_status']}]")

    result = {
        'fastest':            fastest_info,
        'safest':             safest_info,
        'origin_coords':      list(origin_ll),
        'destination_coords': list(dest_ll),
        'ml_active':          ml_active,
        'origin_snap_gap_m':  round(orig_gap),
        'dest_snap_gap_m':    round(dest_gap),
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
        },
    }
    _navigate_cache[cache_key] = result
    return result
