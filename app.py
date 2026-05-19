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
    )
    _TIER2_OK = True
except ImportError:
    pass

# ── weight paths  (env-overridable; defaults match HuggingFace Spaces /app layout) ──
_W = {
    'road':      os.getenv('ROAD_WEIGHTS',      '/app/road_model_best.pth'),
    'landcover': os.getenv('LANDCOVER_WEIGHTS', '/app/landcover_best.pth'),
    'building':  os.getenv('BUILDING_WEIGHTS',  '/app/building_model_best.pth'),
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
    Run Stage 1 road model, then subtract Stage 4 building footprints
    to remove false positives caused by building roof textures.
    Returns (road_mask uint8 0/255, prob float32 0-1) both (H,W).
    """
    from scipy.ndimage import binary_dilation

    model  = _require_model('road')
    tensor = val_transform(image=rgb)['image'].unsqueeze(0).to(_device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255

    # ── Building mask subtraction ─────────────────────────────────────────────
    # Only runs if the building model is loaded; gracefully skips if not.
    # Building model outputs 640×640; road mask is 512×512 — resize first.
    # INTER_NEAREST preserves hard binary edges (no blurred intermediate values).
    if 'building' in _models:
        building_mask = _building_predict(rgb)              # (640,640) 0/255
        road_h, road_w = mask.shape                         # always 512×512
        building_mask_resized = cv2.resize(
            building_mask,
            (road_w, road_h),
            interpolation=cv2.INTER_NEAREST,
        )                                                    # now (512,512)
        building_binary  = (building_mask_resized > 0)
        building_dilated = binary_dilation(building_binary, iterations=1)   # was 3 — reduced to avoid eating road pixels along compound walls
        mask[building_dilated] = 0
        prob[building_dilated] = 0.0
        n_removed = int(building_dilated.sum())
        print(f"🏢  Building subtraction: {n_removed} road pixels removed")

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
               vehicle: str, stem: str, include_images: bool) -> dict:
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
    route_viz_rgb = draw_routes(rgb, G, routes_by_vehicle.get(best, []))

    def _route_dict(r):
        return {
            'rank':               r.rank,
            'total_distance_m':   round(r.total_distance_m, 2),
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
    status:       str
    device:       str
    models_loaded: List[str]
    tier1_available: bool
    tier2_available: bool


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

        t2 = _run_tier2(rgb, tier1_result, vehicle, stem, include_images)

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
