"""
app.py  —  DeepGlobe Satellite Road Intelligence  FastAPI Server
================================================================
Endpoints
---------
GET  /health          model load status
POST /analyze         road extraction + Tier 1 (width + surface)
POST /route           road extraction + Tier 1 + Tier 2 graph routing
POST /full            all 4 stages (road + landcover + building) + Tier 1/2

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

On Kaggle (background cell):
    import subprocess, threading
    threading.Thread(
        target=lambda: subprocess.run(
            ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        )
    ).start()
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from PIL import Image
from pydantic import BaseModel

# ── project imports ────────────────────────────────────────────────────────────
from models import get_road_model, get_landcover_model, get_building_model
from dataset import val_transform, building_val_transform

# Tier 1
_TIER1_OK = False
try:
    from road_width          import RoadWidthEstimator
    from road_type_classifier import RoadTypeClassifier
    from vizualize_road_tier1 import save_tier1_figure
    _TIER1_OK = True
except ImportError:
    pass

# Tier 2
_TIER2_OK = False
try:
    from road_graph import (
        RoadGraph, find_top3_routes, pick_src_dst_auto,
        draw_routes, VEHICLE_TYPES, get_graph_summary,
        find_nearest_node,
    )
    _TIER2_OK = True
except ImportError:
    pass

# Sentinel-2 / GEE
_GEE_OK = False
try:
    from sentinel_connector import SentinelConnector, _GEE_OK
except ImportError:
    pass

# OSM road network
_OSM_OK = False
try:
    from osm_connector import OSMConnector, compute_rgb_change, _OSM_OK
except ImportError:
    pass

def _resolve_weight(env_key: str, filename: str) -> str:
    """
    Resolve model weight path.
    Priority:  env var  →  /app/<filename>  →  /app/Best path/<filename>
    """
    if env_key in os.environ:
        return os.environ[env_key]
    flat_path   = f'/app/{filename}'
    nested_path = f'/app/Best path/{filename}'
    if os.path.isfile(flat_path):
        return flat_path
    if os.path.isfile(nested_path):
        return nested_path
    return flat_path  # return anyway — lifespan will log the missing-file error

_W = {
    'road':      _resolve_weight('ROAD_WEIGHTS',      'road_model_best.pth'),
    'landcover': _resolve_weight('LANDCOVER_WEIGHTS', 'landcover_best.pth'),
    'building':  _resolve_weight('BUILDING_WEIGHTS',  'building_model_best.pth'),
}
RESULTS_DIR = os.getenv('RESULTS_DIR', '/tmp/results')

# ── global model registry ──────────────────────────────────────────────────────
_models:  Dict[str, torch.nn.Module] = {}
_device:  torch.device                = torch.device('cpu')

LANDCOVER_COLORS = {
    0: [0,255,255], 1: [255,255,0], 2: [255,0,255],
    3: [0,255,0],   4: [0,0,255],   5: [255,255,255], 6: [0,0,0],
}


# ══════════════════════════════════════════════════════════════════════════════
# Startup / shutdown
# ══════════════════════════════════════════════════════════════════════════════

def _load_weights(model, path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    return model.to(device).eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {_device}")

    factories = {
        'road':      get_road_model,
        'landcover': get_landcover_model,
        'building':  get_building_model,
    }
    for name, factory in factories.items():
        path = _W[name]
        if os.path.exists(path):
            try:
                _models[name] = _load_weights(factory(), path, _device)
                print(f"✅  {name} model loaded  ←  {path}")
            except Exception as exc:
                print(f"❌  {name} failed: {exc}")
        else:
            print(f"⚠️   {name} weights not found at {path}  (endpoint disabled)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    yield
    _models.clear()
    print("🛑  Models unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "DeepGlobe Satellite Road Intelligence API",
    description = (
        "Road extraction, width estimation, surface classification, "
        "vehicle-aware graph routing, land cover and building detection "
        "from a single satellite image upload."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

# ── 1. CORS  (required for any browser frontend / HuggingFace Spaces UI) ──────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── 2. Upload size cap  (20 MB — prevents silent OOM on HF Spaces) ────────────
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB

class _MaxUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        cl = request.headers.get('content-length')
        if cl and int(cl) > _MAX_UPLOAD_BYTES:
            return JSONResponse(
                {'error': f'File too large (max {_MAX_UPLOAD_BYTES // 1024 // 1024} MB)'},
                status_code=413,
            )
        return await call_next(request)

app.add_middleware(_MaxUploadSize)


# ── 3. Root endpoint  (HuggingFace Spaces health check hits / not /health) ────
@app.get('/', tags=['System'])
def root():
    """API root — used by HuggingFace Spaces liveness probe."""
    return {
        'message': 'DeepGlobe Satellite Road Intelligence API',
        'docs':    '/docs',
        'health':  '/health',
    }

# ── helpers ───────────────────────────────────────────────────────────────────

def _require_model(name: str):
    if name not in _models:
        raise HTTPException(
            status_code = 503,
            detail      = f"'{name}' model not loaded. "
                          f"Check weights path: {_W.get(name, '?')}",
        )
    return _models[name]


def _img_to_b64(arr: np.ndarray, ext: str = '.png') -> str:
    """Encode a numpy BGR/RGB uint8 array to a base64 PNG/JPEG string."""
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode()


def _save_and_b64(arr_rgb: np.ndarray, path: str) -> str:
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    return _img_to_b64(bgr)


async def _read_upload(file: UploadFile) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Save upload to a temp file, read back as RGB + store path.
    Returns (rgb_uint8, bgr_uint8, tmp_path).
    """
    suffix = Path(file.filename or 'upload.jpg').suffix or '.jpg'
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.flush()
    tmp.close()

    bgr = cv2.imread(tmp.name)
    if bgr is None:
        os.unlink(tmp.name)
        raise HTTPException(400, "Cannot decode uploaded image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bgr, tmp.name


def _road_predict(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Stage 1 road model with three post-processing improvements:

    1. TTA (Test Time Augmentation) — 4 geometric variants averaged:
       original, horizontal flip, vertical flip, 90° rotation.
       Typically adds +1–2 IoU without any retraining.

    2. Adaptive threshold — instead of a fixed 0.5, the threshold
       scales with the scene's mean probability so low-contrast arid
       images recover thin roads that would otherwise fall below 0.5.
       Floor is 0.35 to prevent over-detection.

    3. Morphological closing — applied before AND after building
       subtraction to bridge gaps caused by skeletonization artefacts
       and the building-mask subtraction itself.

    Returns (road_mask uint8 0/255, prob float32 0-1) both (H, W).
    """
    from scipy.ndimage import binary_dilation, binary_closing

    model = _require_model('road')

    # ── 1. TTA: run 4 geometric variants, average the prob maps ──────────────
    def _infer(img_np: np.ndarray) -> np.ndarray:
        t = val_transform(image=img_np)['image'].unsqueeze(0).to(_device)
        with torch.no_grad():
            return torch.sigmoid(model(t)).squeeze().cpu().numpy()  # (H, W)

    prob_orig  = _infer(rgb)
    prob_hflip = np.fliplr(_infer(np.fliplr(rgb)))
    prob_vflip = np.flipud(_infer(np.flipud(rgb)))
    prob_rot90 = np.rot90(_infer(np.rot90(rgb, k=1)), k=-1)   # rotate → infer → un-rotate

    prob: np.ndarray = (prob_orig + prob_hflip + prob_vflip + prob_rot90) / 4.0
    print(f"🔄  TTA: mean prob={prob.mean():.4f}  max={prob.max():.4f}")

    # ── 2. Adaptive threshold ─────────────────────────────────────────────────
    # Scale threshold with scene mean so low-contrast arid scenes recover
    # thin roads. Floor at 0.35 prevents over-detection, ceiling at 0.50
    # keeps high-confidence predictions unchanged.
    scene_mean = float(prob.mean())
    thresh = float(np.clip(scene_mean * 0.80, 0.35, 0.50))
    mask   = (prob > thresh).astype(np.uint8) * 255
    print(f"🎯  Adaptive threshold: {thresh:.3f}  "
          f"(scene_mean={scene_mean:.4f})  "
          f"road_px={int((mask > 0).sum())}")

    # ── 3. Morphological closing (pre-subtraction) ────────────────────────────
    # Bridges tiny gaps in the detected network before building subtraction
    # so the skeleton tracer sees a more continuous mask.
    closed = binary_closing(mask > 0, iterations=2)
    mask   = closed.astype(np.uint8) * 255
    print(f"🔲  Closing (pre):  road_px after={int((mask > 0).sum())}")

    # ── Building mask subtraction ─────────────────────────────────────────────
    if 'building' in _models:
        building_mask = _building_predict(rgb)               # (640, 640) 0/255
        road_h, road_w = mask.shape                          # 512 × 512
        building_mask_resized = cv2.resize(
            building_mask,
            (road_w, road_h),
            interpolation=cv2.INTER_NEAREST,
        )
        building_binary  = (building_mask_resized > 0)
        building_dilated = binary_dilation(building_binary, iterations=1)
        mask[building_dilated] = 0
        prob[building_dilated] = 0.0
        n_removed = int(building_dilated.sum())
        print(f"🏢  Building subtraction: {n_removed} px removed")

        # ── 3b. Morphological closing (post-subtraction) ──────────────────────
        # Re-bridge gaps that the building mask cut through.
        closed = binary_closing(mask > 0, iterations=2)
        mask   = closed.astype(np.uint8) * 255
        print(f"🔲  Closing (post): road_px after={int((mask > 0).sum())}")

    return mask, prob



def _landcover_predict(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns ((H,W,3) RGB colour map, (H,W) class-ID array)."""
    model = _require_model('landcover')
    tensor = val_transform(image=rgb)['image'].unsqueeze(0).to(_device)
    with torch.no_grad():
        ids = torch.argmax(model(tensor), dim=1).squeeze().cpu().numpy()
    h, w  = ids.shape
    out   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in LANDCOVER_COLORS.items():
        out[ids == cls_id] = colour
    return out, ids


def _building_predict(rgb: np.ndarray) -> np.ndarray:
    """Returns (H,W) uint8 mask 0/255."""
    model  = _require_model('building')
    dummy  = np.zeros(rgb.shape[:2], dtype=np.float32)
    aug    = building_val_transform(image=rgb, mask=dummy,
                                    edge_mask=dummy, dist_map=dummy)
    tensor = aug['image'].unsqueeze(0).to(_device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8) * 255


def _run_tier1(rgb: np.ndarray, road_mask: np.ndarray,
               stem: str, include_images: bool) -> dict:
    """Run Tier 1 modules; returns summary dict + optional base64 images."""
    if not _TIER1_OK:
        return {'error': 'Tier 1 modules not installed.'}

    # ── Resize RGB to match road_mask coordinate space (512×512) ─────────────
    # road_mask is always 512×512 (model output resolution).  All Tier 1
    # overlays (width heatmap, surface overlay, tier1 figure) draw on top of
    # road pixels — using the original full-res rgb causes annotations to
    # appear only in the top-left quadrant of the image.
    mask_h, mask_w = road_mask.shape[:2]
    rgb_h,  rgb_w  = rgb.shape[:2]
    if rgb_h != mask_h or rgb_w != mask_w:
        rgb = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            (mask_w, mask_h),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    wr  = RoadWidthEstimator()
    wres = wr.analyse(road_mask)

    clf   = RoadTypeClassifier()
    tres  = clf.predict(rgb, road_mask, width_result=wres)

    saved = {}
    if include_images:
        heat_path = os.path.join(RESULTS_DIR, f'{stem}_width_heatmap.png')
        surf_path = os.path.join(RESULTS_DIR, f'{stem}_surface_overlay.png')
        fig_path  = os.path.join(RESULTS_DIR, f'{stem}_tier1_figure.png')
        saved['width_heatmap_b64'] = _save_and_b64(wres.width_heatmap_rgb,  heat_path)
        saved['surface_overlay_b64'] = _save_and_b64(tres['overlay_rgb'],   surf_path)
        tier1_bundle = {'width_result': wres, 'type_result': tres}
        save_tier1_figure(rgb, road_mask, tier1_bundle, fig_path,
                          title=f'Tier 1  |  {stem}')
        with open(fig_path, 'rb') as fh:
            saved['tier1_figure_b64'] = base64.b64encode(fh.read()).decode()

    return {
        'width_result': wres,
        'type_result':  tres,
        'summary': {
            'mean_width_m':    wres.summary_stats['mean_m'],
            'median_width_m':  wres.summary_stats['median_m'],
            'skeleton_pixels': int(wres.skeleton.sum()),
            'is_empty':        bool(wres.is_empty),
            'dominant_surface': tres['summary']['dominant_type'],
            'surface_counts':   tres['summary']['type_counts'],
            'width_class_dist': wres.class_distribution,
        },
        **saved,
    }


def _run_tier2(rgb: np.ndarray, tier1_result: dict,
               vehicle: str, stem: str, include_images: bool,
               src_rc=None, dst_rc=None) -> dict:
    """Run Tier 2 graph routing; returns summary + optional route viz."""
    if not _TIER2_OK:
        return {'error': 'Tier 2 (networkx) not installed.'}

    rg = RoadGraph(tier1_result)
    G  = rg.G
    graph_summary = get_graph_summary(
        G,
        is_urban_scene    = rg.is_urban_corrected,
        junction_density  = rg.junction_density,
        n_urban_corrected = rg.n_urban_corrected,
    )

    # ── Src / Dst resolution ─────────────────────────────────────────────────
    # If the caller provides pixel coordinates (in the original image space),
    # snap to the nearest graph node.  Otherwise fall back to auto-pick.
    if src_rc is not None and dst_rc is not None:
        s_r, s_c = src_rc
        d_r, d_c = dst_rc
        src = find_nearest_node(G, s_r, s_c)
        dst = find_nearest_node(G, d_r, d_c)
        print(f"📍 User src=({s_r},{s_c})→node {src}  dst=({d_r},{d_c})→node {dst}")
    else:
        src, dst = pick_src_dst_auto(G)

    routes_by_vehicle: dict = {}
    for vtype in VEHICLE_TYPES:
        routes_by_vehicle[vtype] = (
            find_top3_routes(G, src, dst, vtype)
            if src is not None and dst is not None else []
        )

    # Best vehicle fallback
    order = [vehicle, 'car', 'motorcycle', 'pedestrian', 'truck']
    best  = next((v for v in order if routes_by_vehicle.get(v)), 'pedestrian')

    # ── CRITICAL: graph node (row, col) are in 512×512 space (model output). ──
    # Drawing them on the original rgb (e.g. 1024×1024) places every route in
    # the top-left quadrant.  Resize to 512×512 so pixel coordinates align.
    h, w = rgb.shape[:2]
    if h != 512 or w != 512:
        viz_rgb = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )
        viz_rgb = cv2.cvtColor(viz_rgb, cv2.COLOR_BGR2RGB)
    else:
        viz_rgb = rgb

    route_viz_rgb = draw_routes(viz_rgb, G, routes_by_vehicle.get(best, []))

    def _route_dict(r):
        cost = r.total_cost
        return {
            'rank':               r.rank,
            'total_distance_m':   round(r.total_distance_m, 2),
            'total_cost':         round(cost, 2) if cost != float('inf') else None,
            'mean_width_m':       round(r.mean_width_m, 2),
            'dominant_surface':   r.dominant_surface,
            'dominant_road_type': r.dominant_road_type,
        }

    out: dict = {
        'graph_summary':     graph_summary,
        'best_vehicle':      best,
        'routes_by_vehicle': {
            v: [_route_dict(r) for r in routes_by_vehicle[v]]
            for v in VEHICLE_TYPES
        },
    }

    if include_images:
        viz_path = os.path.join(RESULTS_DIR, f'{stem}_route_viz.png')
        out['route_viz_b64'] = _save_and_b64(route_viz_rgb, viz_path)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:          str
    device:          str
    models_loaded:   List[str]
    tier1_available: bool
    tier2_available: bool
    gee_available:   bool
    osm_available:   bool


class SentinelRequest(BaseModel):
    """Request body for POST /sentinel."""
    lat_min:        float
    lat_max:        float
    lon_min:        float
    lon_max:        float
    start_date:     str           = '2024-01-01'
    end_date:       str           = '2024-03-31'
    vehicle:        str           = 'car'
    include_images: bool          = True
    cloud_pct:      float         = 20.0
    project_id:     Optional[str] = None


class LiveRequest(BaseModel):
    """Request body for POST /live — OSM roads + Sentinel-2 land cover."""
    lat_min:        float
    lat_max:        float
    lon_min:        float
    lon_max:        float
    start_date:     str           = '2024-01-01'
    end_date:       str           = '2024-03-31'
    vehicle:        str           = 'car'
    include_images: bool          = True
    cloud_pct:      float         = 20.0
    project_id:     Optional[str] = None
    road_types:     Optional[List[str]] = None


class LiveChangeRequest(BaseModel):
    """Request body for POST /live/change — temporal change detection."""
    lat_min:        float
    lat_max:        float
    lon_min:        float
    lon_max:        float
    before_start:   str           = '2022-01-01'
    before_end:     str           = '2022-06-30'
    after_start:    str           = '2024-01-01'
    after_end:      str           = '2024-06-30'
    include_images: bool          = True
    cloud_pct:      float         = 20.0
    project_id:     Optional[str] = None
    change_threshold: float       = 0.10



# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get('/health', response_model=HealthResponse, tags=['System'])
def health():
    """Return current model load status and device."""
    return HealthResponse(
        status          = 'ok' if _models else 'no models loaded',
        device          = str(_device),
        models_loaded   = list(_models.keys()),
        tier1_available = _TIER1_OK,
        tier2_available = _TIER2_OK,
        gee_available   = _GEE_OK,
        osm_available   = _OSM_OK,
    )


@app.post('/analyze', tags=['Inference'])
async def analyze(
    file:           UploadFile = File(..., description='Satellite image (JPG/PNG/TIFF)'),
    include_images: bool = Query(True,  description='Include base64 PNG outputs in response'),
    run_tier1:      bool = Query(True,  description='Run Tier 1 width + surface analysis'),
):
    """
    **Road extraction + optional Tier 1 intelligence.**

    - Runs Stage 1 road segmentation (DeepLabV3+)
    - Optionally runs M1 (width estimation) and M2 (surface classification)
    - Returns JSON summary + base64-encoded visualisations
    """
    try:
        rgb, bgr, tmp = await _read_upload(file)
        stem = Path(file.filename or 'upload').stem

        # Stage 1
        road_mask, road_prob = _road_predict(rgb)

        response: dict = {
            'filename':    file.filename,
            'image_shape': list(rgb.shape[:2]),
            'road': {
                'road_pixels': int((road_mask > 0).sum()),
                'road_pct':    round(float((road_mask > 0).mean()) * 100, 2),
            },
        }

        if include_images:
            # Road prob heatmap
            heat = cv2.applyColorMap(
                (road_prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            response['road']['prob_heatmap_b64'] = _img_to_b64(heat)
            # Binary mask
            response['road']['mask_b64'] = _img_to_b64(road_mask)

        if run_tier1 and _TIER1_OK:
            t1 = _run_tier1(rgb, road_mask, stem, include_images)
            response['tier1'] = t1.get('summary', {})
            if include_images:
                for k in ('width_heatmap_b64', 'surface_overlay_b64', 'tier1_figure_b64'):
                    if k in t1:
                        response['tier1'][k] = t1[k]

        os.unlink(tmp)
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, traceback.format_exc())


@app.post('/route', tags=['Inference'])
async def route(
    file:           UploadFile = File(..., description='Satellite image'),
    vehicle:        str  = Query('car', enum=['pedestrian','motorcycle','car','truck'],
                                 description='Preferred vehicle type'),
    include_images: bool = Query(True,  description='Include route visualisation'),
    src_x: Optional[float] = Query(None, description='Source X as fraction of image width (0-1)'),
    src_y: Optional[float] = Query(None, description='Source Y as fraction of image height (0-1)'),
    dst_x: Optional[float] = Query(None, description='Destination X as fraction of image width (0-1)'),
    dst_y: Optional[float] = Query(None, description='Destination Y as fraction of image height (0-1)'),
):
    """
    **Road extraction + Tier 1 + Tier 2 vehicle-aware graph routing.**

    Builds a NetworkX skeleton graph and finds top-3 routes for all
    4 vehicle types. Returns the route for the requested vehicle type
    (with automatic fallback to the most permissive available mode).
    """
    try:
        rgb, bgr, tmp = await _read_upload(file)
        stem = Path(file.filename or 'upload').stem

        if not _TIER1_OK or not _TIER2_OK:
            raise HTTPException(503, 'Tier 1 or Tier 2 modules not available.')

        road_mask, _ = _road_predict(rgb)
        t1           = _run_tier1(rgb, road_mask, stem, include_images=False)
        tier1_result = {
            'width_result': t1['width_result'],
            'type_result':  t1['type_result'],
        }

        # Map fractional display coords → 512×512 model pixel coords
        src_rc = None
        dst_rc = None
        if src_x is not None and src_y is not None and dst_x is not None and dst_y is not None:
            src_rc = (int(src_y * 512), int(src_x * 512))
            dst_rc = (int(dst_y * 512), int(dst_x * 512))

        t2 = _run_tier2(rgb, tier1_result, vehicle, stem, include_images,
                        src_rc=src_rc, dst_rc=dst_rc)

        response = {
            'filename': file.filename,
            'road_pixels': int((road_mask > 0).sum()),
            'tier1_summary': t1['summary'],
            'tier2': {
                'graph_summary':     t2['graph_summary'],
                'best_vehicle':      t2['best_vehicle'],
                'routes_by_vehicle': t2['routes_by_vehicle'],
            },
        }
        if include_images and 'route_viz_b64' in t2:
            response['tier2']['route_viz_b64'] = t2['route_viz_b64']

        os.unlink(tmp)
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, traceback.format_exc())


@app.post('/full', tags=['Inference'])
async def full_pipeline(
    file:           UploadFile = File(..., description='Satellite image'),
    vehicle:        str  = Query('car', enum=['pedestrian','motorcycle','car','truck']),
    include_images: bool = Query(True),
):
    """
    **Full 4-stage pipeline: Road + Land Cover + Building + Tier 1 + Tier 2.**

    Runs all available models (skips gracefully if weights missing).
    Returns a unified JSON summary with per-stage results and optional
    base64-encoded visualisations.
    """
    try:
        rgb, bgr, tmp = await _read_upload(file)
        stem = Path(file.filename or 'upload').stem

        response: dict = {
            'filename': file.filename,
            'image_shape': list(rgb.shape[:2]),
        }

        # ── Stage 1: Road ──────────────────────────────────────────────────────
        if 'road' in _models:
            road_mask, road_prob = _road_predict(rgb)
            response['road'] = {
                'road_pixels': int((road_mask > 0).sum()),
                'road_pct':    round(float((road_mask > 0).mean()) * 100, 2),
            }
            if include_images:
                response['road']['mask_b64'] = _img_to_b64(road_mask)

        # ── Stage 3: Land Cover ────────────────────────────────────────────────
        if 'landcover' in _models:
            lc_map, ids = _landcover_predict(rgb)   # one forward pass, get both
            class_names = ['urban', 'agriculture', 'rangeland',
                           'forest', 'water', 'barren', 'unknown']
            counts = {class_names[i]: int((ids == i).sum()) for i in range(7)}
            response['landcover'] = {'pixel_counts': counts}
            if include_images:
                response['landcover']['map_b64'] = _img_to_b64(
                    cv2.cvtColor(lc_map, cv2.COLOR_RGB2BGR))

        # ── Stage 4: Building ──────────────────────────────────────────────────
        if 'building' in _models:
            bld_mask = _building_predict(rgb)
            response['building'] = {
                'building_pixels': int((bld_mask > 0).sum()),
                'building_pct':    round(float((bld_mask > 0).mean()) * 100, 2),
            }
            if include_images:
                response['building']['mask_b64'] = _img_to_b64(bld_mask)

        # ── Tier 1 ─────────────────────────────────────────────────────────────
        if 'road' in _models and _TIER1_OK:
            t1 = _run_tier1(rgb, road_mask, stem, include_images)
            response['tier1'] = t1.get('summary', {})
            if include_images:
                for k in ('width_heatmap_b64', 'surface_overlay_b64', 'tier1_figure_b64'):
                    if k in t1:
                        response['tier1'][k] = t1[k]

        # ── Tier 2 ─────────────────────────────────────────────────────────────
        if 'road' in _models and _TIER1_OK and _TIER2_OK:
            tier1_result = {
                'width_result': t1['width_result'],
                'type_result':  t1['type_result'],
            }
            t2 = _run_tier2(rgb, tier1_result, vehicle, stem, include_images)
            response['tier2'] = {
                'graph_summary':     t2['graph_summary'],
                'best_vehicle':      t2['best_vehicle'],
                'routes_by_vehicle': t2['routes_by_vehicle'],
            }
            if include_images and 'route_viz_b64' in t2:
                response['tier2']['route_viz_b64'] = t2['route_viz_b64']

        os.unlink(tmp)
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, traceback.format_exc())


# =============================================================================
# Sentinel-2 Live Endpoint
# =============================================================================

@app.post('/sentinel', tags=['Sentinel-2'])
async def sentinel_analyze(req: SentinelRequest):
    """
    **Fetch live Sentinel-2 imagery from Google Earth Engine and run the full pipeline.**

    Provide a lat/lon bounding box + date range. The server will:
    1. Query the GEE Sentinel-2 SR Harmonized collection
    2. Cloud-filter and take a median composite
    3. Download the RGB thumbnail (upsampled to 512x512)
    4. Run road extraction + Tier 1 (width & surface) + Tier 2 (routing)
    5. Return the same JSON schema as /full, plus `sentinel_metadata`

    **One-time setup required on the server:**
    ```
    pip install earthengine-api requests
    earthengine authenticate
    ```
    """
    if not _GEE_OK:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "Google Earth Engine not available. "
                "Install with: pip install earthengine-api requests  "
                "then run: earthengine authenticate"
            )
        )

    if 'road' not in _models:
        raise HTTPException(503, "Road model not loaded.")

    # Validate vehicle
    valid_vehicles = ['pedestrian', 'motorcycle', 'car', 'truck']
    if req.vehicle not in valid_vehicles:
        raise HTTPException(422, f"vehicle must be one of {valid_vehicles}")

    try:
        # ── 1. Fetch from GEE ─────────────────────────────────────────────────
        bbox = [req.lon_min, req.lat_min, req.lon_max, req.lat_max]
        conn = SentinelConnector(project_id=req.project_id)
        rgb, sentinel_meta = conn.fetch(
            bbox       = bbox,
            start_date = req.start_date,
            end_date   = req.end_date,
            cloud_pct  = req.cloud_pct,
        )

        stem = (f"sentinel_{req.lat_min:.3f}_{req.lon_min:.3f}_"
                f"{req.start_date}_{req.end_date}").replace('-', '')

        # ── 2. Road extraction (Sentinel-aware) ──────────────────────────────
        # NOTE: Our model was trained on 0.5m/pixel DeepGlobe imagery.
        # Sentinel-2 is 10m/pixel (20x lower res). The model outputs ~0.5
        # everywhere (maximum uncertainty). We use a high threshold (0.65)
        # to return only the very highest-confidence detections, avoiding
        # false positives. Major highways may still be partially visible.
        road_mask, road_prob = _road_predict(rgb)
        # Override with a strict Sentinel threshold
        road_mask = (road_prob > 0.65).astype(np.uint8) * 255

        response: dict = {
            'source':            'sentinel-2',
            'sentinel_metadata': sentinel_meta,
            'road_model_warning': (
                'Road model trained on 0.5m/pixel data; Sentinel-2 is 10m/pixel. '
                'Only very high-confidence detections shown (threshold=0.65). '
                'Land cover classification is the primary use case for Sentinel-2.'
            ),
            'image_shape':      list(rgb.shape[:2]),
            'road': {
                'road_pixels': int((road_mask > 0).sum()),
                'road_pct':    round(float((road_mask > 0).mean()) * 100, 2),
            },
        }

        if req.include_images:
            response['road']['mask_b64'] = _img_to_b64(road_mask)
            # Save the raw Sentinel RGB so the user can see what was fetched
            sentinel_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            response['sentinel_rgb_b64'] = _img_to_b64(sentinel_bgr)

        # ── 3. Land cover ─────────────────────────────────────────────────────
        if 'landcover' in _models:
            lc_map, ids = _landcover_predict(rgb)
            class_names = ['urban', 'agriculture', 'rangeland',
                           'forest', 'water', 'barren', 'unknown']
            counts = {class_names[i]: int((ids == i).sum()) for i in range(7)}
            response['landcover'] = {'pixel_counts': counts}
            if req.include_images:
                response['landcover']['map_b64'] = _img_to_b64(
                    cv2.cvtColor(lc_map, cv2.COLOR_RGB2BGR))

        # ── 4. Building ───────────────────────────────────────────────────────
        if 'building' in _models:
            bld_mask = _building_predict(rgb)
            response['building'] = {
                'building_pixels': int((bld_mask > 0).sum()),
                'building_pct':    round(float((bld_mask > 0).mean()) * 100, 2),
            }
            if req.include_images:
                response['building']['mask_b64'] = _img_to_b64(bld_mask)

        # ── 5. Tier 1 ─────────────────────────────────────────────────────────
        t1 = None
        if _TIER1_OK:
            t1 = _run_tier1(rgb, road_mask, stem, req.include_images)
            response['tier1'] = t1.get('summary', {})
            if req.include_images:
                for k in ('width_heatmap_b64', 'surface_overlay_b64', 'tier1_figure_b64'):
                    if k in t1:
                        response['tier1'][k] = t1[k]

        # ── 6. Tier 2 ─────────────────────────────────────────────────────────
        if t1 and _TIER2_OK:
            tier1_result = {
                'width_result': t1['width_result'],
                'type_result':  t1['type_result'],
            }
            t2 = _run_tier2(rgb, tier1_result, req.vehicle, stem,
                            req.include_images)
            response['tier2'] = {
                'graph_summary':     t2['graph_summary'],
                'best_vehicle':      t2['best_vehicle'],
                'routes_by_vehicle': t2['routes_by_vehicle'],
            }
            if req.include_images and 'route_viz_b64' in t2:
                response['tier2']['route_viz_b64'] = t2['route_viz_b64']

        return JSONResponse(response)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        raise HTTPException(500, traceback.format_exc())


# =============================================================================
# Live Road Analysis  —  OSM + Sentinel-2
# =============================================================================

@app.post('/live', tags=['Live Analysis'])
async def live_analyze(req: LiveRequest):
    """
    **Fetch live road network from OSM + land cover from Sentinel-2 and run the full pipeline.**

    - Road mask comes from OpenStreetMap (no image needed, no resolution limit)
    - Land cover comes from Sentinel-2 via Google Earth Engine (optional)
    - Surface classification, width measurement, and routing all run unchanged
    - Works for any location on Earth

    **Requirements:**
    - `pip install osmnx`  (always needed)
    - `pip install earthengine-api && earthengine authenticate`  (for land cover only)
    """
    if not _OSM_OK:
        raise HTTPException(503,
            "osmnx not installed. Run: pip install osmnx")
    if 'road' not in _models:
        raise HTTPException(503, "Road model not loaded.")

    valid_vehicles = ['pedestrian', 'motorcycle', 'car', 'truck']
    if req.vehicle not in valid_vehicles:
        raise HTTPException(422, f"vehicle must be one of {valid_vehicles}")

    try:
        # ── 1. OSM road mask ──────────────────────────────────────────────────
        osm = OSMConnector()
        osm_result = osm.fetch(
            lat_min    = req.lat_min,
            lon_min    = req.lon_min,
            lat_max    = req.lat_max,
            lon_max    = req.lon_max,
            road_types = req.road_types,
        )
        road_mask = osm_result['mask']   # (512,512) uint8 — ready for pipeline

        stem = (f"live_{req.lat_min:.3f}_{req.lon_min:.3f}").replace('.', 'd')

        response: dict = {
            'source':      'osm+sentinel-2',
            'osm_metadata': {
                'total_roads':    osm_result['total_roads'],
                'total_length_m': osm_result['total_length_m'],
                'highway_counts': osm_result['highway_counts'],
                'surface_counts': osm_result['surface_counts'],
                'n_nodes':        osm_result['n_nodes'],
                'n_edges':        osm_result['n_edges'],
            },
            'road': {
                'road_pixels': int((road_mask > 0).sum()),
                'road_pct':    round(float((road_mask > 0).mean()) * 100, 2),
            },
        }

        if req.include_images:
            response['road']['mask_b64'] = _img_to_b64(road_mask)

        # ── 2. Sentinel-2 RGB + land cover (optional) ─────────────────────────
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)   # fallback grey canvas
        if _GEE_OK:
            try:
                bbox = [req.lon_min, req.lat_min, req.lon_max, req.lat_max]
                conn = SentinelConnector(project_id=req.project_id)
                rgb, sentinel_meta = conn.fetch(
                    bbox       = bbox,
                    start_date = req.start_date,
                    end_date   = req.end_date,
                    cloud_pct  = req.cloud_pct,
                )
                response['sentinel_metadata'] = sentinel_meta
                if req.include_images:
                    response['sentinel_rgb_b64'] = _img_to_b64(
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            except Exception as gee_err:
                response['sentinel_warning'] = (
                    f"Sentinel-2 fetch skipped: {gee_err}. "
                    "Road analysis continues with OSM only."
                )
        else:
            response['sentinel_warning'] = (
                "GEE not available — land cover skipped. "
                "Run: pip install earthengine-api && earthengine authenticate"
            )

        # Land cover on Sentinel-2 RGB
        if 'landcover' in _models and rgb.mean() > 1:
            lc_map, ids = _landcover_predict(rgb)
            class_names = ['urban', 'agriculture', 'rangeland',
                           'forest', 'water', 'barren', 'unknown']
            counts = {class_names[i]: int((ids == i).sum()) for i in range(7)}
            response['landcover'] = {'pixel_counts': counts}
            if req.include_images:
                response['landcover']['map_b64'] = _img_to_b64(
                    cv2.cvtColor(lc_map, cv2.COLOR_RGB2BGR))

        # ── 3. Tier 1 — width + surface on OSM mask ───────────────────────────
        t1 = None
        if _TIER1_OK:
            t1 = _run_tier1(rgb, road_mask, stem, req.include_images)
            response['tier1'] = t1.get('summary', {})
            if req.include_images:
                for k in ('width_heatmap_b64', 'surface_overlay_b64',
                          'tier1_figure_b64'):
                    if k in t1:
                        response['tier1'][k] = t1[k]

        # ── 4. Tier 2 — graph + routing on OSM mask ───────────────────────────
        if t1 and _TIER2_OK:
            tier1_result = {
                'width_result': t1['width_result'],
                'type_result':  t1['type_result'],
            }
            t2 = _run_tier2(rgb, tier1_result, req.vehicle, stem,
                            req.include_images)
            response['tier2'] = {
                'graph_summary':     t2['graph_summary'],
                'best_vehicle':      t2['best_vehicle'],
                'routes_by_vehicle': t2['routes_by_vehicle'],
            }
            if req.include_images and 'route_viz_b64' in t2:
                response['tier2']['route_viz_b64'] = t2['route_viz_b64']

        return JSONResponse(response)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        raise HTTPException(500, traceback.format_exc())


# =============================================================================
# Live Change Detection  —  Two-epoch Sentinel-2 + OSM road overlay
# =============================================================================

@app.post('/live/change', tags=['Live Analysis'])
async def live_change(req: LiveChangeRequest):
    """
    **Detect land cover and visual change between two time periods using Sentinel-2.**

    Compares two Sentinel-2 median composites (before / after) for the same
    bounding box and returns:
    - RGB change magnitude map (orange = new construction, blue = new vegetation)
    - Land cover class change table (urban↑, forest↓, etc.)
    - Current OSM road network overlaid on the change map

    **Requirements:**
    - `pip install earthengine-api osmnx`
    - `earthengine authenticate`
    """
    if not _GEE_OK:
        raise HTTPException(503,
            "GEE not available. Run: pip install earthengine-api "
            "then: earthengine authenticate")

    try:
        bbox = [req.lon_min, req.lat_min, req.lon_max, req.lat_max]
        conn = SentinelConnector(project_id=req.project_id)

        # ── 1. Fetch before + after from GEE ─────────────────────────────────
        print("[INFO] Fetching T1 (before)...")
        rgb_t1, meta_t1 = conn.fetch(bbox, req.before_start,
                                     req.before_end, req.cloud_pct)
        print("[INFO] Fetching T2 (after)...")
        rgb_t2, meta_t2 = conn.fetch(bbox, req.after_start,
                                     req.after_end, req.cloud_pct)

        # ── 2. RGB change map ─────────────────────────────────────────────────
        change = compute_rgb_change(rgb_t1, rgb_t2,
                                    threshold=req.change_threshold)

        response: dict = {
            'source':         'sentinel-2-change',
            'before_metadata': meta_t1,
            'after_metadata':  meta_t2,
            'change': {
                'changed_pct':    change['changed_pct'],
                'increased_px':   change['increased_px'],   # brighter (construction)
                'decreased_px':   change['decreased_px'],   # darker   (vegetation)
                'threshold_used': req.change_threshold,
            },
        }

        if req.include_images:
            response['before_rgb_b64']    = _img_to_b64(
                cv2.cvtColor(rgb_t1, cv2.COLOR_RGB2BGR))
            response['after_rgb_b64']     = _img_to_b64(
                cv2.cvtColor(rgb_t2, cv2.COLOR_RGB2BGR))
            response['change_map_b64']    = _img_to_b64(
                cv2.cvtColor(change['change_map_rgb'], cv2.COLOR_RGB2BGR))

        # ── 3. Land cover change ──────────────────────────────────────────────
        if 'landcover' in _models:
            class_names = ['urban', 'agriculture', 'rangeland',
                           'forest', 'water', 'barren', 'unknown']
            _, ids_t1 = _landcover_predict(rgb_t1)
            _, ids_t2 = _landcover_predict(rgb_t2)

            lc_change_table = []
            for i, cls in enumerate(class_names):
                before_px = int((ids_t1 == i).sum())
                after_px  = int((ids_t2 == i).sum())
                lc_change_table.append({
                    'class':     cls,
                    'before_px': before_px,
                    'after_px':  after_px,
                    'delta_px':  after_px - before_px,
                    'delta_pct': round(
                        (after_px - before_px) / max(before_px, 1) * 100, 1),
                })

            # Dominant change = class with largest absolute delta
            dominant = max(lc_change_table,
                           key=lambda x: abs(x['delta_px']), default=None)
            response['change']['lc_change_table']  = lc_change_table
            response['change']['dominant_change']  = (
                dominant['class'] if dominant else 'none')
            response['change']['dominant_direction'] = (
                'increase' if dominant and dominant['delta_px'] > 0
                else 'decrease')

            if req.include_images:
                lc_map_t1, _ = _landcover_predict(rgb_t1)
                lc_map_t2, _ = _landcover_predict(rgb_t2)
                response['before_lc_b64'] = _img_to_b64(
                    cv2.cvtColor(lc_map_t1, cv2.COLOR_RGB2BGR))
                response['after_lc_b64']  = _img_to_b64(
                    cv2.cvtColor(lc_map_t2, cv2.COLOR_RGB2BGR))

        # ── 4. OSM road overlay on change map ─────────────────────────────────
        if _OSM_OK:
            try:
                osm = OSMConnector()
                osm_result = osm.fetch(req.lat_min, req.lon_min,
                                       req.lat_max, req.lon_max)
                road_mask = osm_result['mask']
                response['osm_metadata'] = {
                    'total_roads':    osm_result['total_roads'],
                    'total_length_m': osm_result['total_length_m'],
                    'highway_counts': osm_result['highway_counts'],
                }
                if req.include_images:
                    # Overlay OSM roads (yellow) on change map
                    overlay = change['change_map_rgb'].copy()
                    overlay[road_mask > 0] = [255, 230, 0]   # yellow roads
                    response['change_with_roads_b64'] = _img_to_b64(
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as osm_err:
                response['osm_warning'] = f"OSM overlay skipped: {osm_err}"

        return JSONResponse(response)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        raise HTTPException(500, traceback.format_exc())
