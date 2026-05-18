---
title: DeepGlobe Satellite Road Intelligence API
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🛰️ DeepGlobe Satellite Image Analysis Pipeline


A four-stage end-to-end pipeline for satellite imagery analysis, trained on the **DeepGlobe 2018** and **Massachusetts Buildings** datasets — extended with a **Tier 1 Road Intelligence Layer** for per-pixel width estimation, surface type classification, and vehicle-aware routing.

---

## 🗺️ Pipeline Overview

```
Satellite Image
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
┌──────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  Stage 1 │ Road Extraction           │  │  Stage 4 │ Building Detection           │
│  DeepLabV3+ ResNet34 │ 512×512       │  │  UnetPlusPlus ResNet50+scse │ 640×640  │
└──────────────┬───────────────────────┘  └────────────────────┬────────────────────┘
               │   ◄── building mask subtraction (dilate ×3) ──┘
               │         (removes roof-texture false positives)
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Corrected Road Mask                                                 │
│  ├─ Tier 1 M1 │ Width Estimation    │ Medial Axis DT                │
│  ├─ Tier 1 M2 │ Surface Classifier  │ KMeans on GLCM patches        │
│  └─ Tier 1 M3 │ Vehicle Router      │ NetworkX graph routing        │
│  Stage 2 │ Road Inpainting          │ Partial Conv U-Net │ 512×512  │
│  Stage 3 │ Land Cover (7-class)     │ DeepLabV3+ ResNet34│ 512×512  │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼
  5-Panel Composite Visualization  +  5-Panel Tier 1 Intelligence Map
```

> **Cross-stage feedback:** Stage 4 (building detection) runs in parallel with Stage 1 and its output is used to *correct* the road mask before any downstream analysis — see [Building-Mask Road Correction](#-building-mask-road-correction) below.

---

## 📊 Results

### Stage 1 – Road Extraction (DeepGlobe)

| Metric    | Value  |
|-----------|--------|
| IoU       | ~0.60  |
| F1        | ~0.75  |
| Precision | ~0.74  |
| Recall    | ~0.76  |

### Stage 2 – Road Inpainting

| Metric | Value |
|--------|-------|
| PSNR   | ~30dB |
| SSIM   | ~0.92 |

### Stage 3 – Land Cover Classification (DeepGlobe, 7-class)

| Metric       | Value  |
|--------------|--------|
| mIoU         | ~0.55  |
| Pixel Acc.   | ~0.82  |

### Stage 4 – Building Detection (Massachusetts)

| Metric    | Standard | TTA (×8) |
|-----------|----------|-----------|
| IoU       | 0.5552   | 0.5663    |
| Dice      | 0.7136   | 0.7224    |
| Precision | 0.6957   | 0.7172    |
| Recall    | 0.7357   | 0.7312    |
| F1        | 0.7151   | 0.7241    |

> Model: UnetPlusPlus + ResNet50 + scse attention | Trained on 137 images, stopped at epoch 38 (early stopping)

---

## 🏢 Building-Mask Road Correction

Building rooftops share many visual features with paved roads (grey tones, rectangular edges, smooth texture) and are a common source of **false-positive road pixels**. To eliminate this artefact, `_road_predict()` in `app.py` automatically applies a building-mask correction whenever the Stage 4 model is loaded:

```python
if 'building' in _models:
    building_mask    = _building_predict(rgb)              # (H,W) 0/255
    building_binary  = (building_mask > 0)
    building_dilated = binary_dilation(building_binary, iterations=3)
    mask[building_dilated] = 0     # zero binary road mask
    prob[building_dilated] = 0.0   # zero confidence map (fixes heatmap)
    n_removed = int(building_dilated.sum())
    print(f"🏢  Building subtraction: {n_removed} road pixels removed")
```

| Detail | Value |
|--------|-------|
| Dilation radius | 3 px — creates a small buffer so road pixels *hugging* building edges are also removed |
| Both `mask` and `prob` zeroed | The prob map feeds the confidence heatmap visualisation; zeroing only the mask would leave ghost activations |
| Graceful degradation | The `if 'building' in _models` guard means `/analyze`, `/route`, and `/full` still work correctly even when building weights are absent |
| Zero retraining | Pure post-processing patch — no model changes, no extra training |
| Endpoints fixed | `_road_predict()` is the single call point for all three endpoints, so the fix applies everywhere automatically |

---

## 🧠 Tier 1 Road Intelligence Layer

Built on top of the Stage 1 binary road mask, the Tier 1 layer extracts rich per-pixel road attributes — all in pure NumPy, no additional model training required.

### Module 1 — Road Width Estimation (`road_width.py`)

Uses the **medial axis distance transform** to derive road width at every skeleton pixel.

| Road Class    | Width Range | Colour        |
|---------------|-------------|---------------|
| Footpath      | < 3 m       | Gold          |
| Single Lane   | 3 – 6 m     | Dark Orange   |
| Standard Road | 6 – 12 m    | Dodger Blue   |
| Highway       | > 12 m      | Crimson       |

- GSD = 0.5 m/pixel (matches DeepGlobe tile resolution)
- Outputs: `skeleton`, `width_px`, `width_m`, `road_type_map`, plasma heatmap RGB

### Module 2 — Road Surface Type Classifier (`road_type_classifier.py`)

Unsupervised **KMeans (k=3)** on 47-dimensional texture/colour/edge features extracted from 16×16 patches centred on sampled skeleton points.

| Surface  | Feature Heuristic              | Colour  |
|----------|-------------------------------|---------|
| Paved    | Lowest GLCM contrast          | Green   |
| Unpaved  | Intermediate contrast          | Orange  |
| Damaged  | Highest edge density (Sobel)  | Crimson |

- Feature vector: 40 GLCM values + 6 RGB channel stats + 1 Sobel edge density
- KDTree propagation fills all skeleton pixels from the sampled subset
- `fit()` accepts a batch of images; `predict()` auto-fits if unfitted

### Module 3 — Vehicle-Aware Road Router (`road_router.py`)

Finds the **least-cost path** between two pixel coordinates using `skimage.graph.MCP_Geometric` on a cost surface derived from the skeleton, width map, and surface type.

**Vehicle profiles:**

| Vehicle    | Min Width | Surface Penalty | Refuses Damaged |
|------------|-----------|-----------------|-----------------|
| Pedestrian | 0 m       | ×1.0            | No              |
| Motorcycle | 1.5 m     | ×1.0            | No              |
| Car        | 3 m       | ×1.5 (unpaved)  | No              |
| Truck      | 6 m       | ×2.0 (unpaved)  | Yes             |

- Cost surface: `base_cost × width_reward × surface_penalty`
- Endpoints are auto-snapped to the nearest skeleton pixel
- Returns: path pixels, distance in metres, dominant surface, mean width, route overlay RGB

---

## 📸 Visual Results

### Stage 1 — Road Extraction

5-panel visualization: Satellite · Ground Truth · Confidence Heatmap · Binary Prediction · TP/FP/FN Overlay

![Road extraction — Val #639](results%20of%20training%20road/road_viz_20260509_132833_sample01_val639.png)
![Road extraction — Val #915](results%20of%20training%20road/road_viz_20260509_132833_sample02_val915.png)
![Road extraction — Val #147](results%20of%20training%20road/road_viz_20260509_132833_sample03_val147.png)
![Road extraction — Val #77](results%20of%20training%20road/road_viz_20260509_132833_sample04_val77.png)
![Road extraction — Val #845](results%20of%20training%20road/road_viz_20260509_132833_sample05_val845.png)

---

### Tier 1 Road Intelligence — 5-Panel Output

Satellite · Binary Mask · Width Heatmap (plasma) · Surface Type Overlay · Combined Intelligence Map

> Generated by `vizualize_road_tier1.py` — road-type colour blended with surface-type colour on a darkened satellite base.

---

### Stage 2 — Road Inpainting

8-panel: Original Mask · Corrupted · Inpainted Prediction · Ground Truth · Error Map · Width Heatmap · Surface Type · Vehicle Route

> Model trained to epoch 40 | Best Val L_hole = 0.0302 | Best Hole IoU = 0.4675

![Inpainting — Val #1011](results%20of%20road%20inpainting/inpaint_20260513_113911_01_val1011.png)
![Inpainting — Val #319](results%20of%20road%20inpainting/inpaint_20260513_113911_02_val319.png)
![Inpainting — Val #349](results%20of%20road%20inpainting/inpaint_20260513_113911_03_val349.png)
![Inpainting — Val #751](results%20of%20road%20inpainting/inpaint_20260513_113911_04_val751.png)
![Inpainting — Val #91](results%20of%20road%20inpainting/inpaint_20260513_113911_05_val91.png)

---

### Stage 3 — Land Cover Classification

4-panel: Satellite · Ground Truth · Prediction · Error Map  *(Best val mIoU = 0.5952)*

![Land cover predictions — full validation panel](results%20of%20training%20landcover/landcover_visual_20260427_093916.png)

**Confusion Matrix**

![Land cover confusion matrix](results%20of%20training%20landcover/landcover_confusion_matrix.png)

---

### Stage 4 — Building Detection (Massachusetts)

7-panel: Satellite · Ground Truth · Raw Prediction · Postprocessed · Confidence Heatmap · Boundary · Instance Map

**Best predictions (IoU ≥ 0.75)**

![Best — IoU=0.8168](results%20of%20training%20building%20%28masachusses%20dataset%29/24479005_15_best.png)
![Best — IoU=0.7543](results%20of%20training%20building%20%28masachusses%20dataset%29/23728840_15_best.png)
![Best — IoU=0.7375](results%20of%20training%20building%20%28masachusses%20dataset%29/23878945_15_best.png)

**Mid-range predictions (IoU 0.49–0.54) — dense urban grid**

![Mid-High — IoU=0.5437](results%20of%20training%20building%20%28masachusses%20dataset%29/23579080_15_mid_high.png)
![Mid-Low — IoU=0.4872](results%20of%20training%20building%20%28masachusses%20dataset%29/24329020_15_mid_low.png)

**Hardest case (IoU=0.21) — tiny dot-like footprints**

![Worst — IoU=0.2146](results%20of%20training%20building%20%28masachusses%20dataset%29/23129125_15_worst.png)

---

## 📁 Project Structure

```
deepglobe-part-2/
│
├── models.py                  # All model factory functions (Road / LC / Building)
├── dataset.py                 # Dataset classes for all 4 tasks
├── partial_conv.py            # Partial convolution layers (Stage 2)
├── inpainting_model.py        # Partial Conv U-Net architecture
├── inpainting_losses.py       # Multi-component inpainting loss
├── inpainting_dataset.py      # Inpainting dataset with dynamic hole generation
│
├── train_road.py              # Stage 1 training script
├── train_inpainting.py        # Stage 2 training script
├── train_landcover.py         # Stage 3 training script
├── train_building.py          # Stage 4 training script
│
├── evaluate_road.py           # Stage 1 evaluation
├── evaluate_inpainting.py     # Stage 2 evaluation
├── evaluate_landcover.py      # Stage 3 evaluation
├── evaluate_building.py       # Stage 4 evaluation + TTA
│
├── vizualize_road.py          # Stage 1 visualization (5-panel)
├── visualize_landcover.py     # Stage 3 visualization
├── visualize_building.py      # Stage 4 visualization (7-panel)
├── infer_inpainting.py        # Stage 2 inference + visualization
│
│── ── Tier 1 Road Intelligence ──────────────────────────────────────
├── road_width.py              # Module 1: Width estimation (medial axis DT)
├── road_type_classifier.py    # Module 2: Surface type (KMeans on GLCM patches)
├── road_router.py             # Module 3: Vehicle-aware routing (MCP_Geometric)
├── vizualize_road_tier1.py    # 5-panel Tier 1 intelligence visualiser
│
├── inference.py               # Single-image inference (road / lc / building)
├── pipeline.py                # Unified 4-stage pipeline + RoadInpaintingPipeline
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

**Recommended environment:** Kaggle P100 (16 GB VRAM) or dual T4 (30 GB total)

---

## 🚀 Usage

### Single-Image Inference

```bash
# Road extraction
python inference.py --image sat.jpg --model road

# Land cover classification
python inference.py --image sat.jpg --model landcover

# Building footprint detection
python inference.py --image tile.tif --model building
```

### 2-Stage Road Pipeline (Road + Inpainting)

```bash
python pipeline.py road \
  --image sat.jpg \
  --s1 /kaggle/working/road_model_best.pth \
  --s2 /kaggle/working/inpainting_model_best.pth
```

### Full 4-Stage Pipeline

```bash
python pipeline.py full \
  --image sat.jpg \
  --road     /kaggle/working/road_model_best.pth \
  --inpaint  /kaggle/working/inpainting_model_best.pth \
  --lc       /kaggle/working/landcover_model_best.pth \
  --building /kaggle/working/building_model_best.pth \
  --outdir   /kaggle/working/results/pipeline
```

### Tier 1 Road Intelligence (standalone)

```python
from road_width import RoadWidthEstimator
from road_type_classifier import RoadTypeClassifier
from road_router import RoadRouter
from vizualize_road_tier1 import save_tier1_figure

# Module 1 — Width estimation
width_result = RoadWidthEstimator().analyse(road_mask_np)
# width_result.skeleton, .width_m, .road_type_map, .width_heatmap_rgb

# Module 2 — Surface type classification
type_result = RoadTypeClassifier().predict(image_np, road_mask_np, width_result)
# type_result['surface_map'], ['overlay_rgb'], ['summary']

# Module 3 — Vehicle-aware routing
router = RoadRouter(road_mask_np, width_result, type_result)
route  = router.find_route(src=(r0, c0), dst=(r1, c1), vehicle_type='car')
# route.found, .distance_m, .dominant_surface, .route_overlay_rgb

# Tier 1 5-panel visualisation
tier1 = {'width_result': width_result, 'type_result': type_result}
save_tier1_figure(image_np, road_mask_np, tier1, 'output/tier1.png')
```

### Tier 1 — One-liner convenience

```python
from vizualize_road_tier1 import visualise_tier1

path = visualise_tier1(image_np, road_mask_np, save_dir='./results/tier1', stem='sample01')
# runs Module 1 + 2 internally, saves 5-panel PNG, returns absolute path
```

### Python API (full pipeline)

```python
from pipeline import SatellitePipeline

pipe = SatellitePipeline(
    road_path     = 'road_model_best.pth',
    inpaint_path  = 'inpainting_model_best.pth',
    lc_path       = 'landcover_model_best.pth',
    building_path = 'building_model_best.pth',
)
result = pipe.run('satellite.jpg', save_dir='./output')
# result['figure_path'] → 5-panel composite PNG
```

### REST API (FastAPI server)

```bash
# Start locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# → Interactive docs at http://localhost:8000/docs
```

| Endpoint | Description |
|----------|-------------|
| `GET  /`        | Liveness probe (HuggingFace health check) |
| `GET  /health`  | Model load status + device + Tier availability |
| `POST /analyze` | Road mask + Tier 1 width/surface analysis |
| `POST /route`   | + Tier 2 vehicle-aware graph routing |
| `POST /full`    | All 4 stages + Tier 1 + Tier 2 |

```bash
# Example — road extraction + Tier 2 routing for a truck
curl -X POST "http://localhost:8000/route?vehicle=truck&include_images=true" \
     -F "file=@741856_sat.jpg" | python -m json.tool
```

All image outputs are returned as **base64-encoded PNGs** in the JSON response.  
Pass `?include_images=false` to get JSON-only (faster, smaller payload).

#### Docker / HuggingFace Spaces

```bash
# Build and run locally
docker build -t deepglobe-api .
docker run -p 7860:7860 \
  -e ROAD_WEIGHTS=/app/weights/road_model_best.pth \
  -e LANDCOVER_WEIGHTS=/app/weights/landcover_best.pth \
  -e BUILDING_WEIGHTS=/app/weights/building_model_best.pth \
  -v /path/to/weights:/app/weights \
  deepglobe-api
```

> On HuggingFace Spaces (Docker SDK): push the repo, set weight paths as  
> **Space Secrets**, and the server starts automatically on port 7860.

---

## 🧠 Model Architecture Details

### Stage 1 & 3 — DeepLabV3+ (ResNet34)
- **ASPP** (Atrous Spatial Pyramid Pooling) for multi-scale context
- `encoder_output_stride=16` for finer boundary detail in LC
- Input: 512×512 RGB | Output: logits (B,1,H,W) or (B,7,H,W)

### Stage 2 — Partial Convolution U-Net
- Partial convolutions handle arbitrary hole masks without border artifacts
- **Loss:** Valid loss + Hole loss + Perceptual (VGG16) + Connectivity
- Input: (corrupted_mask, hole_mask) | Output: completed road mask

### Stage 4 — UnetPlusPlus (ResNet50 + scse)
- **Nested dense skip connections** for multi-scale building structure
- **scse attention** suppresses shadow false-positives in decoder
- Input: 640×640 RGB | Output: logits (B,1,H,W)
- **TTA:** 8 augmented variants (flips + rotations) averaged at inference

### Tier 1 — Road Intelligence Modules (NumPy-only)
- **Module 1** (`road_width.py`): `skimage.morphology.medial_axis` with `return_distance=True`; width = 2 × radius × GSD (0.5 m/px); fully vectorised classification into 4 road types
- **Module 2** (`road_type_classifier.py`): 47-dim feature vector (GLCM × 40 + RGB stats × 6 + Sobel edge × 1); `StandardScaler` + `KMeans(k=3)`; heuristic cluster labelling; KDTree propagation to all skeleton pixels
- **Module 3** (`road_router.py`): `skimage.graph.MCP_Geometric` on a cost surface blending width reward and surface penalty; 4 vehicle profiles with minimum-width and surface-type constraints; endpoints auto-snapped to nearest skeleton pixel

---

## 📦 Datasets

| Stage | Dataset | Size | Source |
|-------|---------|------|--------|
| 1 & 3 | DeepGlobe 2018 | 803 images | [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) |
| 2     | Synthetic holes on Stage-1 output | — | Generated |
| 4     | Massachusetts Buildings | 151 images (137 train / 4 val / 10 test) | [Kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) |

---

## 🎨 Land Cover Classes

| ID | Class       | Color   |
|----|-------------|---------|
| 0  | Urban       | Cyan    |
| 1  | Agriculture | Yellow  |
| 2  | Rangeland   | Magenta |
| 3  | Forest      | Green   |
| 4  | Water       | Blue    |
| 5  | Barren      | White   |
| 6  | Unknown     | Black   |

---

## 🔑 Key Engineering Decisions

- **AMP (fp16)** throughout for memory efficiency on P100
- **Extension-agnostic mask lookup** in building dataset to prevent silent `.tif`/`.tiff` mismatches
- **Honest IoU** — `compute_iou_batch` skips empty-union samples instead of returning false 1.0
- **Cosine Annealing** LR schedule with warm restarts for building training
- **Morphological post-processing** (opening + closing) on building predictions to clean noise
- **Instance segmentation** via connected components for per-building instance maps
- **Cross-stage building subtraction** — `_road_predict()` calls `_building_predict()` on the same image and removes dilated building footprints from the road mask *and* probability map before returning, eliminating roof-texture false positives with zero retraining
- **Dual-array zeroing** — both the binary `mask` and the float `prob` array are zeroed at building pixels; zeroing only the mask would leave ghost activations in the confidence heatmap
- **Vectorised cost surface** — Tier 1 router builds the entire MCP cost grid with NumPy array ops; no Python-level per-pixel loop
- **KDTree label propagation** — Tier 1 classifier samples up to 200 skeleton points then propagates labels to all remaining skeleton pixels via nearest-neighbour lookup, avoiding an O(N) prediction loop
- **Auto-snap endpoints** — router snaps arbitrary pixel coordinates to the nearest skeleton pixel before MCP traversal, preventing "no path found" failures on off-skeleton inputs
- **Zero-torch Tier 1** — all three Tier 1 modules are pure NumPy / scikit-learn / scikit-image; they run on CPU with no GPU dependency

---

## 📄 License

MIT
