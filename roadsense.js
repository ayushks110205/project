const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, ExternalHyperlink,
  TableOfContents
} = require('docx');
const fs = require('fs');

const ACCENT = "1B4F8A";       // deep blue
const ACCENT2 = "2E7D32";      // forest green
const LIGHT_BG = "EBF2FA";     // light blue tint
const GREEN_BG = "E8F5E9";
const AMBER_BG = "FFF8E1";
const GREY_TEXT = "555555";
const DARK = "1A1A2E";
const TABLE_HEADER = "1B4F8A";
const TABLE_STRIPE = "F0F6FF";
const CODE_BG = "F4F4F8";

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 6 } },
    children: [new TextRun({ text, font: "Arial", size: 32, bold: true, color: ACCENT })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 26, bold: true, color: ACCENT })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: "333366" })]
  });
}

function heading4(text) {
  return new Paragraph({
    spacing: { before: 180, after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: "1B4F8A", italics: false })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22, color: DARK, ...opts })]
  });
}

function bodyMixed(runs) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    children: runs.map(r => new TextRun({ font: "Arial", size: 22, color: DARK, ...r }))
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 22, color: DARK })]
  });
}

function bulletMixed(runs, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ font: "Arial", size: 22, color: DARK, ...r }))
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 22, color: DARK })]
  });
}

function spacer(lines = 1) {
  return Array.from({ length: lines }, () => new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun("")] }));
}

function code(text) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    shading: { fill: CODE_BG, type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "333333" })]
  });
}

function noteBox(text, type = "info") {
  const fill = type === "warning" ? AMBER_BG : type === "success" ? GREEN_BG : LIGHT_BG;
  const color = type === "warning" ? "5D4037" : type === "success" ? "1B5E20" : "0D2B5E";
  const icon = type === "warning" ? "⚠ " : type === "success" ? "✓ " : "ℹ ";
  return new Paragraph({
    spacing: { before: 100, after: 100 },
    indent: { left: 360, right: 360 },
    shading: { fill, type: ShadingType.CLEAR },
    border: {
      left: { style: BorderStyle.SINGLE, size: 16, color: type === "warning" ? "F57C00" : type === "success" ? "2E7D32" : ACCENT }
    },
    children: [
      new TextRun({ text: icon, font: "Arial", size: 20, bold: true, color: type === "warning" ? "E65100" : type === "success" ? "1B5E20" : ACCENT }),
      new TextRun({ text, font: "Arial", size: 20, color })
    ]
  });
}

function divider() {
  return new Paragraph({
    spacing: { before: 180, after: 180 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
    children: [new TextRun("")]
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerRow(cells, widths) {
  return new TableRow({
    tableHeader: true,
    children: cells.map((text, i) => new TableCell({
      borders,
      width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: TABLE_HEADER, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 140, right: 140 },
      children: [new Paragraph({ children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: "FFFFFF" })] })]
    }))
  });
}

function dataRow(cells, widths, stripe = false) {
  const fill = stripe ? TABLE_STRIPE : "FFFFFF";
  return new TableRow({
    children: cells.map((text, i) => new TableCell({
      borders,
      width: { size: widths[i], type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 140, right: 140 },
      children: [new Paragraph({ children: [new TextRun({ text: String(text ?? ""), font: "Arial", size: 20, color: DARK })] })]
    }))
  });
}

function dataRowMixed(cells, widths, stripe = false) {
  const fill = stripe ? TABLE_STRIPE : "FFFFFF";
  return new TableRow({
    children: cells.map((runs, i) => new TableCell({
      borders,
      width: { size: widths[i], type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 140, right: 140 },
      children: [new Paragraph({
        children: (typeof runs === "string"
          ? [new TextRun({ text: runs, font: "Arial", size: 20, color: DARK })]
          : runs.map(r => new TextRun({ font: "Arial", size: 20, color: DARK, ...r })))
      })]
    }))
  });
}

// === DOCUMENT CONTENT ===
const children = [

  // Cover section
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 720, after: 120 },
    children: [new TextRun({ text: "🛰️  RoadSense", font: "Arial", size: 64, bold: true, color: ACCENT })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "Full System Architecture & Technical Reference", font: "Arial", size: 32, color: GREY_TEXT })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text: "Satellite-powered road safety routing engine built on OSM, Sentinel-2 imagery, and ML surface classification", font: "Arial", size: 22, italics: true, color: GREY_TEXT })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: "Live Demo: ", font: "Arial", size: 22, bold: true, color: ACCENT }), new TextRun({ text: "https://huggingface.co/spaces/Ayushks07/updated_road_extraction", font: "Arial", size: 22, color: ACCENT2 })]
  }),
  divider(),

  // Table of Contents (Word auto-generates)
  heading1("Table of Contents"),
  new TableOfContents("Table of Contents", {
    hyperlink: true,
    headingStyleRange: "1-3",
  }),
  divider(),

  // Section 1
  heading1("1. Project Overview"),
  body("RoadSense is an end-to-end road safety routing system that combines satellite remote sensing, machine learning, and graph-based routing to provide both a fastest and a safest driving route for Indian roads. Unlike conventional mapping apps, RoadSense actively classifies road surface quality (paved, unpaved, damaged) from Sentinel-2 satellite imagery and factors that classification into route cost functions."),
  ...spacer(1),
  body("The system is deployed on HuggingFace Spaces and supports four vehicle types: car, motorcycle, truck, and pedestrian. It integrates live traffic data from the Mappls API and dynamically adapts routing for sparse rural areas through bbox expansion."),

  heading2("1.1 Key Capabilities"),
  bullet("Dual-route mode: computes both the fastest and safest routes in a single API call"),
  bullet("ML surface classification: Random Forest on 47-dimensional GLCM+RGB+Sobel features"),
  bullet("Live traffic integration: Mappls Traffic Flow API with congestion multipliers"),
  bullet("Gap-aware routing: smart bbox expansion for rural/sparse OSM areas"),
  bullet("Label invariant enforcement: guarantees 'Fastest' is always objectively faster"),
  bullet("Full fallback matrix: 22 documented failure modes, none causing crashes"),

  heading2("1.2 Tech Stack Summary"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3200, 6160],
    rows: [
      headerRow(["Layer", "Technology"], [3200, 6160]),
      dataRow(["Backend Framework", "FastAPI (Python)"], [3200, 6160], false),
      dataRow(["Surface Classifier", "scikit-learn Random Forest (47-dim features)"], [3200, 6160], true),
      dataRow(["Road Extraction", "PyTorch U-Net"], [3200, 6160], false),
      dataRow(["Road Network", "OSMnx + NetworkX (Dijkstra)"], [3200, 6160], true),
      dataRow(["Satellite Imagery", "Google Earth Engine (Sentinel-2, 10m/pixel)"], [3200, 6160], false),
      dataRow(["Geocoding / Traffic", "Mappls API (OAuth2)"], [3200, 6160], true),
      dataRow(["State Boundaries", "Nominatim (OpenStreetMap)"], [3200, 6160], false),
      dataRow(["Frontend", "Vanilla HTML/CSS/JS + Leaflet.js"], [3200, 6160], true),
      dataRow(["Tile Layers", "CartoDB Dark Matter, CartoDB Voyager, Esri Satellite"], [3200, 6160], false),
      dataRow(["Deployment", "HuggingFace Spaces (Docker)"], [3200, 6160], true),
    ]
  }),
  ...spacer(1),

  divider(),

  // Section 2 - High Level Pipeline
  heading1("2. High-Level Pipeline"),
  body("The pipeline begins when a user enters origin and destination locations. The complete flow from input to routed response is summarised below:"),
  ...spacer(1),
  numbered("User enters From/To → Frontend resolves locations via /autocomplete"),
  numbered("Pin confirmation step — user can drag pins to adjust exact coordinates"),
  numbered("POST /navigate with lat/lng + vehicle type"),
  numbered("Backend: geocode and validate both coordinates"),
  numbered("Steps 3+4 run with a bbox dependency: OSM graph download → Sentinel-2 RGB fetch"),
  numbered("Step 5: OSM raster mask (512×512 pixel)"),
  numbered("Step 6: ML inference on pre-fetched RGB satellite data"),
  numbered("Step 7: Paint ML surface labels onto graph edges (majority vote per edge)"),
  numbered("Step 8: Set routing weights — time-based (fastest) and damage-weighted (safest)"),
  numbered("Step 9: Snap origin/destination to graph + bbox expansion for sparse areas"),
  numbered("Step 10: Dijkstra for fastest path and safest path simultaneously"),
  numbered("Step 11: Fetch Mappls live traffic data in parallel"),
  numbered("Step 12: Build full route info with colored segments and warnings"),
  numbered("Step 12b: Label invariant enforcement — swap routes if labels would be misleading"),
  numbered("Return JSON response to frontend"),
  numbered("Frontend: single-route or dual-route mode rendering + gap detection"),

  divider(),

  // Section 3 - Backend
  heading1("3. Backend Architecture"),
  body("The backend is implemented entirely in Python using FastAPI. The core routing logic lives in navigate.py (~1,370 lines) and is exposed through app.py."),

  heading2("3.1 Constants & Configuration"),
  body("The following constants control system-wide routing behaviour:"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 2760, 3800],
    rows: [
      headerRow(["Constant", "Purpose", "Values"], [2800, 2760, 3800]),
      dataRow(["_SPEED_DEFAULTS_KMH", "Base speed by OSM highway type", "motorway/trunk: 80, primary/secondary: 50, tertiary/residential: 30, track: 15, path/footway: 5"], [2800, 2760, 3800], false),
      dataRow(["_VEHICLE_SPEED_CAP_KMH", "Max speed per vehicle type", "car: 80, motorcycle: 60, truck: 60, pedestrian: 5"], [2800, 2760, 3800], true),
      dataRow(["_DAMAGE_MULTIPLIER", "Cost multiplier for safest routing", "paved: 1.0×, unpaved: 1.4×, damaged: 2.5×"], [2800, 2760, 3800], false),
      dataRow(["_IMPASSABLE_COST", "Penalty for blocked roads", "1,000,000 (soft penalty)"], [2800, 2760, 3800], true),
      dataRow(["_TRAFFIC_MAP", "Congestion → time multiplier", "free_flow: 1.0×, light: 1.2×, moderate: 1.5×, heavy: 2.0×, severe: 2.5×"], [2800, 2760, 3800], false),
      dataRow(["_GAP_THRESHOLD_M", "Snap gap to trigger bbox expansion", "300 metres"], [2800, 2760, 3800], true),
    ]
  }),

  heading2("3.2 The Routing Pipeline — navigate()"),
  body("The main navigate() function in navigate.py orchestrates the full 12-step pipeline. Each step is described in detail below."),

  heading3("Step 1 — Cache Check"),
  body("A cache key is formed from origin coordinates, destination coordinates, and vehicle type. If the key exists in the LRU cache, the cached result is returned immediately without any further computation."),
  noteBox("Cache hit eliminates the 18–25 second cold-request latency entirely.", "success"),

  heading3("Step 2 — Geocode Origin & Destination"),
  body("Both origin and destination strings are resolved to (lat, lng) coordinate pairs using the following priority order:"),
  bullet("Fast path: if the string matches the lat,lng format, it is parsed directly with no API call"),
  bullet("Mappls Place Search: queries atlas.mappls.com/api/places/search/json"),
  bullet("Mappls Geocode: queries atlas.mappls.com/api/places/geocode as fallback"),
  noteBox("Fallback: If Mappls API fails or the key is missing, a ValueError is raised → HTTP 400 response.", "warning"),

  heading3("Steps 3+4 — OSM Graph Download and Sentinel-2 Fetch"),
  body("Both operations run inside a ThreadPoolExecutor, but are not truly parallel — they have a bbox dependency. The graph must complete first to provide the bounding box for the satellite imagery request:"),
  ...spacer(1),
  code("with ThreadPoolExecutor(max_workers=2) as pool:"),
  code("    G, bbox  = pool.submit(build_road_graph, origin, dest, vehicle).result()   # ~5–8 sec"),
  code("    rgb, ml  = pool.submit(fetch_satellite, bbox).result()                      # ~15–20 sec"),
  ...spacer(1),
  bodyMixed([
    { text: "build_road_graph(): ", bold: true },
    { text: "Downloads the OSM road network via ox.graph_from_bbox() with a 500m buffer. Result is LRU-cached for repeated calls to the same area." }
  ]),
  bodyMixed([
    { text: "fetch_satellite(): ", bold: true },
    { text: "Fetches a Sentinel-2 RGB composite from Google Earth Engine — last 90 days, ≤20% cloud cover. Returns a 512×512×3 uint8 array." }
  ]),
  noteBox("Graph fallback: if network_type='drive' fails, retries with network_type='all'. Satellite fallback: if GEE is unavailable, returns a blank image and sets ml_active=False.", "warning"),

  heading3("Step 5 — OSM Raster Mask"),
  body("An OSMConnector instance fetches OSM road data for the bounding box and rasterizes it to a 512×512 binary mask. This mask is used for ML alignment — it tells the classifier which pixels are known to contain roads."),
  noteBox("Fallback: If OSMConnector import fails (_OSM_MASK_OK=False), a blank mask is used. ML still runs but with reduced accuracy.", "warning"),

  heading3("Step 6 — ML Surface Analysis"),
  body("The pre-fetched RGB satellite array and the OSM road mask are passed to run_ml_analysis(), which does not fetch any additional data. Inside this function:"),
  numbered("RoadTypeClassifier (Random Forest on 47-dim GLCM+RGB+Sobel features) predicts surface type per road pixel"),
  numbered("RoadWidthEstimator estimates road width in metres from the binary mask using morphological analysis"),
  numbered("Returns two 512×512 maps: surface_map (paved / unpaved / damaged) and width_m_map (float32)"),
  noteBox("Fallback: If ML models fail, returns empty maps and the pipeline continues using OSM-only surface labels.", "warning"),

  heading3("Step 7 — Paint ML Labels onto Graph Edges"),
  body("For each edge in the road graph, the system samples multiple points along the edge geometry, maps each point to a pixel in the ML surface map, and assigns the most common (majority-vote) surface label. Two overrides apply:"),
  bullet("If ML predicts 'damaged' but OSM highway type is 'trunk' or 'motorway' → OSM tag is trusted (see Section 5.1 for rationale)"),
  bullet("edge['passable'] is set to True or False based on vehicle type and the estimated road width"),

  heading3("Step 8 — Set Routing Weights"),
  body("Two independent weight sets are computed for every edge:"),
  ...spacer(1),
  bodyMixed([{ text: "Fastest weight (time-based): ", bold: true }, { text: "weight = edge_length_metres / (effective_speed_kmh / 3.6)  →  result in seconds" }]),
  bodyMixed([{ text: "Safest weight (damage-weighted): ", bold: true }, { text: "weight = edge_length_metres × damage_multiplier  →  paved: 1.0×, unpaved: 1.4×, damaged: 2.5×" }]),
  body("Impassable edges receive +1,000,000 on both weights — they are strongly discouraged but never hard-blocked, ensuring the router can always find a path."),

  heading3("Step 9 — Snap Origin/Destination to Graph"),
  body("ox.nearest_edges() locates the closest road segment (edge geometry) to each coordinate, then selects the nearer endpoint node. This approach reduces snap error significantly compared to nearest_nodes() on long straight roads (see Section 5.3 for rationale)."),

  heading3("Step 9b — Smart Bbox Expansion for Sparse Areas"),
  body("If either snap gap exceeds 300 metres, the system automatically attempts to expand the bounding box by ~0.005° (~550m) past the coordinate and re-downloads the OSM graph with network_type='all' (which captures tracks and footpaths not in the 'drive' network). The expanded graph is only adopted if the new snap gap is at least 40% smaller than the original gap."),
  noteBox("Fallback: If expansion fails or does not improve the gap, the original graph is retained silently. This step never crashes the pipeline.", "success"),

  heading3("Step 10 — Dijkstra Routing"),
  body("Two independent shortest-path queries run on the same graph using NetworkX:"),
  code("fastest_path = nx.shortest_path(G, orig, dest, weight='weight_fastest')"),
  code("safest_path  = nx.shortest_path(G, orig, dest, weight='weight_safest')"),
  noteBox("Fallback: If one path fails (NetworkXNoPath), it is mirrored from the other. If both fail, a ValueError is raised → HTTP 404 'No route found'.", "warning"),

  heading3("Step 11 — Traffic Data (Parallel)"),
  body("Traffic multipliers for both routes are fetched simultaneously using a ThreadPoolExecutor. For each route, up to 20 sample points are sent to the Mappls Traffic Flow API. The response is mapped to a time multiplier and applied per-edge to the estimated travel time."),
  noteBox("Fallback: If the Mappls key is missing, the API fails, or the vehicle is 'pedestrian', returns [1.0] — no traffic adjustment applied.", "warning"),

  heading3("Step 12 — Build Route Info"),
  body("build_route_info() walks the final node path and assembles the complete route descriptor:"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2600, 1400, 5360],
    rows: [
      headerRow(["Field", "Type", "Description"], [2600, 1400, 5360]),
      dataRow(["distance_km", "float", "Total route distance in kilometres"], [2600, 1400, 5360], false),
      dataRow(["estimated_minutes", "float", "Travel time with traffic multipliers applied"], [2600, 1400, 5360], true),
      dataRow(["polyline", "[[lat,lng]...]", "Route coordinates for map rendering"], [2600, 1400, 5360], false),
      dataRow(["segments.good_km", "float", "Kilometres of paved road"], [2600, 1400, 5360], true),
      dataRow(["segments.unpaved_km", "float", "Kilometres of unpaved road"], [2600, 1400, 5360], false),
      dataRow(["segments.damaged_km", "float", "Kilometres of damaged road"], [2600, 1400, 5360], true),
      dataRow(["warnings", "string[]", "Damage warnings grouped by road name"], [2600, 1400, 5360], false),
      dataRow(["traffic_status", "string", "Dominant congestion level across the route"], [2600, 1400, 5360], true),
      dataRow(["traffic_delay_minutes", "float", "Extra time added by live traffic"], [2600, 1400, 5360], false),
      dataRow(["surface_source", "string", "'ml', 'osm', or 'mixed' — origin of surface data"], [2600, 1400, 5360], true),
      dataRow(["colored_segments", "[{surface, coords}]", "Per-segment surface type with coordinates for route inspection"], [2600, 1400, 5360], false),
    ]
  }),

  heading3("Step 12b — Label Invariant Enforcement"),
  body("After both routes are built, two rules are applied to ensure the route labels are never misleading:"),
  ...spacer(1),
  bodyMixed([{ text: "Rule 1: ", bold: true }, { text: "If the 'Safest' route has fewer travel minutes than the 'Fastest' route, the two routes are swapped." }]),
  bodyMixed([{ text: "Rule 2: ", bold: true }, { text: "If the 'Fastest' route has more damaged road than the 'Safest' route but the time advantage is ≤5 minutes, the routes are swapped — the marginal time saving is not worth the extra road damage exposure." }]),
  noteBox("This guarantees that the 'Fastest Route' label is never paradoxically slower than the 'Safest Route'. See Section 5.4 for full rationale.", "info"),

  divider(),

  heading2("3.3 API Endpoints (app.py)"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1200, 2800, 5360],
    rows: [
      headerRow(["Method", "Path", "Purpose"], [1200, 2800, 5360]),
      dataRow(["GET", "/", "Redirects to /navigator"], [1200, 2800, 5360], false),
      dataRow(["GET", "/navigator", "Serves navigator.html (main SPA)"], [1200, 2800, 5360], true),
      dataRow(["GET", "/viewer", "Serves viewer.html (satellite viewer page)"], [1200, 2800, 5360], false),
      dataRow(["POST", "/navigate", "Full route calculation — calls navigate()"], [1200, 2800, 5360], true),
      dataRow(["GET", "/autocomplete", "Place search for input autocomplete"], [1200, 2800, 5360], false),
      dataRow(["POST", "/live", "Live satellite analysis (Sentinel-2)"], [1200, 2800, 5360], true),
    ]
  }),
  ...spacer(1),
  body("The /navigate endpoint wraps the pipeline with full error handling: ValueError maps to HTTP 400, any other Exception maps to HTTP 500, and the full traceback is always logged to console."),

  divider(),

  // Section 4 - ML Models
  heading1("4. ML Models"),

  heading2("4.1 Road Surface Classifier (road_type_classifier.py)"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2200, 7160],
    rows: [
      headerRow(["Property", "Detail"], [2200, 7160]),
      dataRow(["Architecture", "Supervised Random Forest on 47-dimensional feature vectors"], [2200, 7160], false),
      dataRow(["Features", "GLCM texture (5 properties × distances × angles) + RGB channel stats + Sobel edge gradients"], [2200, 7160], true),
      dataRow(["Input", "512×512×3 RGB satellite image + 512×512 binary road mask"], [2200, 7160], false),
      dataRow(["Output", "Per-road-pixel surface classification: 'paved' / 'unpaved' / 'damaged'"], [2200, 7160], true),
      dataRow(["Training", "train_surface_rf.py → produces surface_rf.pkl (scikit-learn RF + StandardScaler)"], [2200, 7160], false),
      dataRow(["Dataset", "389 labelled road patches (Unpaved: 161, Paved: 121, Damaged: 107)"], [2200, 7160], true),
      dataRow(["Metrics", "5-Fold CV Macro-F1: 60.45% ± 5.57%, Hold-Out Macro-F1: 53.49%, Hold-Out Accuracy: 55.0%, Weighted F1: 55.0%"], [2200, 7160], false),
      dataRow(["Fallback", "If surface_rf.pkl is missing → falls back to unsupervised KMeans + heuristic labelling"], [2200, 7160], true),
    ]
  }),

  heading2("4.2 Road Width Estimator (road_width.py)"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2200, 7160],
    rows: [
      headerRow(["Property", "Detail"], [2200, 7160]),
      dataRow(["Architecture", "Morphological analysis on binary road mask"], [2200, 7160], false),
      dataRow(["Input", "512×512 road mask"], [2200, 7160], true),
      dataRow(["Output", "512×512 width-in-metres map (float32)"], [2200, 7160], false),
      dataRow(["Purpose", "Determines if roads are wide enough for trucks (min 3.5m) or cars (min 2.5m)"], [2200, 7160], true),
    ]
  }),

  heading2("4.3 U-Net Road Extraction (models.py)"),
  body("Base U-Net model definitions used for Stage 1 segmentation tasks: road extraction, land cover segmentation, and building detection. These are pixel-level segmentation models — architecturally separate from the Random Forest surface classifier. Implemented in PyTorch."),
  
  heading3("Dataset Information"),
  body("The model was trained on the DeepGlobe Road Extraction Dataset:"),
  bullet("Total Images: 6,226"),
  bullet("Training Images: 4,981"),
  bullet("Validation Images: 1,245"),
  
  heading3("Evaluation Metrics"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4680, 4680],
    rows: [
      headerRow(["Metric", "Value"], [4680, 4680]),
      dataRow(["Mean IoU", "62.04%"], [4680, 4680], false),
      dataRow(["Mean Dice/F1", "75.38%"], [4680, 4680], true),
      dataRow(["Precision", "76.67%"], [4680, 4680], false),
      dataRow(["Recall", "76.67%"], [4680, 4680], true),
    ]
  }),
  ...spacer(1),
  body("The U-Net model demonstrates strong validation performance, achieving a balance of precision and recall that is critical for road extraction. The resulting segmentation masks are robust enough to accurately delineate road boundaries in complex rural and urban environments."),

  heading2("4.4 Dataset Description"),
  body("The system relies on two primary datasets for training its machine learning models:"),
  
  heading3("Road Extraction Dataset"),
  bullet("Source: DeepGlobe Road Extraction Dataset"),
  bullet("Total Images: 6,226"),
  bullet("Train Images: 4,981"),
  bullet("Validation Images: 1,245"),
  bullet("Task: Binary road segmentation"),
  
  heading3("Road Surface Classification Dataset"),
  bullet("Total labelled patches: 389"),
  bullet("Paved: 121"),
  bullet("Unpaved: 161"),
  bullet("Damaged: 107"),
  bullet("Features: 47-dimensional GLCM + RGB + Sobel feature vectors"),
  body("These datasets were carefully selected to provide a robust foundation for satellite-based road intelligence. The DeepGlobe dataset offers extensive high-resolution coverage essential for accurate road boundary detection, while the custom surface classification dataset enables the system to differentiate between subtle surface textures (paved, unpaved, damaged) that are critical for safety-aware routing calculations."),

  heading2("4.5 Model Evaluation"),
  
  heading3("A. U-Net Road Extraction Results"),
  bullet("Mean IoU: 62.04%"),
  bullet("Mean Dice/F1: 75.38%"),
  bullet("Precision: 76.67%"),
  bullet("Recall: 76.67%"),
  
  heading3("B. Random Forest Surface Classification Results"),
  bullet("Cross Validation Macro-F1: 60.45% ± 5.57%"),
  bullet("Hold-Out Macro-F1: 53.49%"),
  bullet("Hold-Out Accuracy: 55.0%"),
  bullet("Weighted F1: 55.0%"),
  body("These metrics indicate a solid baseline for satellite-based road analysis. The U-Net's high Dice score (75.38%) demonstrates strong spatial agreement between predicted and actual road geometries, ensuring precise routing graphs. While the surface classifier's accuracy (55.0%) reflects the inherent difficulty of distinguishing fine-grained surface damage from 10m/pixel Sentinel-2 imagery, the balanced precision and recall ensure the model provides a valuable probabilistic safety signal that consistently improves routing outcomes over unweighted paths."),

  heading2("4.6 Confusion Matrix Analysis"),
  body("A detailed breakdown of the Random Forest surface classifier performance reveals category-specific strengths and weaknesses:"),
  
  heading3("Paved"),
  bullet("Precision: 0.50 | Recall: 0.67 | F1: 0.57"),
  heading3("Unpaved"),
  bullet("Precision: 0.68 | Recall: 0.59 | F1: 0.63"),
  heading3("Damaged"),
  bullet("Precision: 0.44 | Recall: 0.36 | F1: 0.40"),
  
  heading3("Analysis"),
  bullet("Unpaved roads are classified most accurately."),
  bullet("Damaged roads are the most challenging category."),
  bullet("Sentinel-2 resolution (10m/pixel) contributes to classification difficulty."),
  bullet("Limited labelled data also contributes to lower damaged-road performance."),
  noteBox("Figure Placeholder: Random Forest Confusion Matrix", "info"),

  divider(),

  // Section 5 - Frontend
  heading1("5. Frontend Architecture"),
  body("The frontend is a single-page application (~2,040 lines) implemented in vanilla HTML, CSS, and JavaScript with Leaflet.js for interactive mapping. No frontend framework is used."),

  heading2("5.1 Layout Structure"),
  body("The app is a full-viewport flex row with two primary zones:"),
  ...spacer(1),
  bodyMixed([{ text: "Left panel (340px fixed): ", bold: true }, { text: "State/UT selector, From/To inputs with autocomplete, vehicle toggle (Car / Bike / Truck / Walk), Find Routes button, ML status badge, 4-step progress animation, error card, and route result cards." }]),
  bodyMixed([{ text: "Right map area (flex: 1): ", bold: true }, { text: "Leaflet map, floating peak hour badge (top-left), and tile layer switcher — Dark Matter / Street / Satellite (top-right)." }]),

  heading2("5.2 Key Frontend Systems"),

  heading3("Autocomplete Engine"),
  bullet("Triggers on ≥2 characters with a 300ms debounce"),
  bullet("Fetches from GET /autocomplete?q=...&state=...&lat=...&lng=..."),
  bullet("Recent searches stored in localStorage (last 5 entries)"),
  bullet("Keyboard navigation: ↑ ↓ Enter Esc"),
  bullet("AbortController cancels stale requests to prevent race conditions"),

  heading3("State Boundary System"),
  bullet("Instant zoom to rectangular STATE_BOUNDS on dropdown change"),
  bullet("Async GeoJSON polygon fetch from Nominatim API"),
  bullet("Draws blue dashed polygon + dark world overlay for clarity"),
  bullet("Re-fits map to actual polygon bounds after fetch completes"),
  bullet("Fallback: if Nominatim fails, keeps the rectangular boundary"),

  heading3("Pin Confirmation Flow"),
  body("After location resolution, draggable A (blue) and B (red) pins are placed on the map with a dashed preview line. As the user drags pins, the input values, tooltips, and sidebar text update in real-time. Clicking Confirm triggers callNavigateAPI() with the final coordinates."),

  heading3("Route Rendering — Single vs Dual Mode"),
  body("Routes are considered identical if distance differs by ≤0.05 km, time by ≤1 minute, and polyline point count by ≤2. In that case, single-route mode is activated:"),
  bullet("Safest card is hidden; fastest card becomes '🗺️ Best Available Route' with purple gradient"),
  bullet("Info banner: 'Only one viable route found — no safer alternative available via OSM road network'"),
  bullet("Purple polyline with auto-inspected colored segments on load"),
  ...spacer(1),
  body("In dual-route mode, both the blue (fastest) and green (safest) polylines are shown. Clicking a route card redraws it as color-coded surface segments while dimming the other route to 25% opacity."),

  heading3("Peak Hour ETA System"),
  body("A time-of-day multiplier is applied client-side to show peak-hour estimates:"),
  bullet("07:00 and 16:00 → 1.15× (pre-rush)"),
  bullet("08:00–10:30 → 1.35× (morning rush)"),
  bullet("17:00–19:59 → 1.40× (evening rush)"),
  bullet("All other hours → 1.0× (off-peak)"),

  heading3("Last-Mile Gap Detection"),
  body("If the server-reported snap gap (origin_snap_gap_m / dest_snap_gap_m) exceeds 150 metres, a grey dashed connector line is drawn between the user's pin and the actual route start/end, and an amber warning appears on the route card. The client falls back to a haversine calculation if server values are missing."),

  divider(),

  // Section 6 - Design Rationale
  heading1("6. Design Rationale"),

  heading2("6.1 Why OSM Tags Override ML on Motorways"),
  body("The ML surface classifier is trained on DeepGlobe imagery at 0.5m/pixel. At inference time it runs on Sentinel-2 imagery at 10m/pixel — 20× coarser. At this resolution, major highways can appear spectrally similar to unpaved roads due to mixing with surrounding terrain. OSM data for motorways and trunk roads is highly reliable — these roads are never marked as damaged in the mapped network."),
  noteBox("Rule: If highway ∈ {motorway, trunk} and ML predicts 'damaged' → override with the OSM surface tag.", "info"),

  heading2("6.2 Why Sequential Graph + Satellite Fetch"),
  body("Both build_road_graph() (OSMnx download, ~5–8 sec) and fetch_satellite() (GEE API call, ~15–20 sec) are pure network I/O. The graph must complete first to provide the bounding box for the satellite request. The ThreadPoolExecutor is retained for consistency and to leave room for a future optimisation: pre-computing a rough bounding box to fire both requests simultaneously."),

  heading2("6.3 Why OSMnx nearest_edges() Over nearest_nodes()"),
  body("On residential roads, graph nodes (intersections) can be 100–200m apart. Using nearest_nodes() snaps to the closest intersection, which may be far from the actual coordinate on a long straight segment. nearest_edges() finds the closest point on any road segment's geometry, then picks the nearer endpoint node. This substantially reduces snap error in practice."),
  noteBox("Fallback: nearest_nodes() is used if nearest_edges() raises an exception (e.g. on an empty graph).", "info"),

  heading2("6.4 Why the Label Invariant Swap"),
  body("The safest route's damage_multiplier weight sometimes accidentally discovers faster paths — the detour around damaged roads may use higher-class highways with faster speed limits. Without the invariant check, the 'Safest' route could paradoxically show a shorter travel time than the 'Fastest' route, which would confuse users and break trust in the UI labels."),
  body("The two-rule enforcement guarantees: (1) 'Fastest' always has fewer or equal minutes, and (2) if 'Fastest' has more damage but saves ≤5 minutes, labels are swapped — the marginal time saving is not worth the extra road damage exposure."),

  divider(),

  // Section 7 - API Response
  heading1("7. API Response Schema"),
  body("The /navigate endpoint returns a JSON object with the following top-level structure:"),
  ...spacer(1),
  code("{"),
  code('  "fastest": {'),
  code('    "distance_km": 85.5,'),
  code('    "estimated_minutes": 97,'),
  code('    "polyline": [[22.8, 86.2], ...],'),
  code('    "segments": { "good_km": 72.2, "unpaved_km": 6.0, "damaged_km": 7.2 },'),
  code('    "warnings": ["Damaged surface on unnamed road (~7.2 km total)"],'),
  code('    "road_count": 42,'),
  code('    "junction_count": 18,'),
  code('    "traffic_status": "free_flow",'),
  code('    "traffic_delay_minutes": 0,'),
  code('    "surface_source": "ml",'),
  code('    "colored_segments": [{ "surface": "paved", "coords": [...] }, ...]'),
  code("  },"),
  code('  "safest": { ... },'),
  code('  "origin_coords": [22.8046, 86.2029],'),
  code('  "destination_coords": [23.3315, 85.3250],'),
  code('  "ml_active": true,'),
  code('  "origin_snap_gap_m": 45,'),
  code('  "dest_snap_gap_m": 180,'),
  code('  "graph_stats": { "nodes": 12450, "edges": 15230 }'),
  code("}"),

  divider(),

  // Section 8 - Fallback Matrix
  heading1("8. Complete Fallback Matrix"),
  body("RoadSense has 22 documented failure modes, none of which cause an unhandled crash. The full matrix is provided below for operational reference:"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [480, 2080, 2200, 4600],
    rows: [
      headerRow(["#", "Failure Point", "What Breaks", "Fallback Behaviour"], [480, 2080, 2200, 4600]),
      dataRow(["1", "Mappls API key missing/expired", "Autocomplete, geocoding, traffic", "Autocomplete returns empty; geocode raises error; traffic returns 1.0×"], [480, 2080, 2200, 4600], false),
      dataRow(["2", "Mappls geocode fails", "Location resolution", "ValueError → HTTP 400 → 'Could not geocode'"], [480, 2080, 2200, 4600], true),
      dataRow(["3", "Sentinel-2 imagery unavailable", "ML surface classification", "fetch_satellite() returns blank image + ml_active=False"], [480, 2080, 2200, 4600], false),
      dataRow(["4", "ML model files missing", "Surface + width prediction", "run_ml_analysis() returns empty maps → OSM-only fallback"], [480, 2080, 2200, 4600], true),
      dataRow(["5", "RF bundle (surface_rf.pkl) missing", "Supervised classifier", "Falls back to unsupervised KMeans + heuristic labelling"], [480, 2080, 2200, 4600], false),
      dataRow(["6", "OSM has no surface tag", "Edge surface label", "Defaults to 'unpaved'"], [480, 2080, 2200, 4600], true),
      dataRow(["7", "ML says 'damaged' on motorway", "Incorrect ML prediction", "Trust OSM: major highways override ML labels"], [480, 2080, 2200, 4600], false),
      dataRow(["8", "network_type='drive' fails", "Graph download", "Falls back to network_type='all'"], [480, 2080, 2200, 4600], true),
      dataRow(["9", "Snap gap > 300m (sparse area)", "Route far from destination", "Bbox expansion with network_type='all'"], [480, 2080, 2200, 4600], false),
      dataRow(["10", "Bbox expansion fails", "Sparse area recovery", "Silently keeps original graph — never crashes"], [480, 2080, 2200, 4600], true),
      dataRow(["11", "Expanded graph is disconnected", "No path after merge", "Routing uses original graph nodes"], [480, 2080, 2200, 4600], false),
      dataRow(["12", "Nominatim boundary fetch fails", "State polygon", "Falls back to rectangular boundary from STATE_BOUNDS"], [480, 2080, 2200, 4600], true),
      dataRow(["13", "One Dijkstra route fails", "Missing fastest or safest", "Mirrors the successful route"], [480, 2080, 2200, 4600], false),
      dataRow(["14", "Both Dijkstra routes fail", "No path exists", "ValueError → HTTP 404 → 'No route found'"], [480, 2080, 2200, 4600], true),
      dataRow(["15", "Traffic API fails", "Live congestion data", "Returns [1.0] → no traffic adjustment applied"], [480, 2080, 2200, 4600], false),
      dataRow(["16", "colored_segments missing", "Route inspection view", "Draws single solid-color polyline instead"], [480, 2080, 2200, 4600], true),
      dataRow(["17", "Route endpoint far from pin", "Visual disconnect", "Dashed connector line + amber gap warning"], [480, 2080, 2200, 4600], false),
      dataRow(["18", "Safest route is actually faster", "Misleading route labels", "Label invariant swap ensures fastest always has fewer minutes"], [480, 2080, 2200, 4600], true),
      dataRow(["19", "Fastest has more damage (≤5min diff)", "Misleading recommendation", "Swaps labels → prefers safer option when time cost is marginal"], [480, 2080, 2200, 4600], false),
      dataRow(["20", "Both routes identical", "Redundant UI", "Single-route mode: one card, auto-inspected segments, info banner"], [480, 2080, 2200, 4600], true),
      dataRow(["21", "Clipboard API fails", "Copy button", "Silent .catch(() => {}) — no crash"], [480, 2080, 2200, 4600], false),
      dataRow(["22", "Geolocation denied", "Proximity bias in autocomplete", "_userLat/_userLng stay null → no bias applied"], [480, 2080, 2200, 4600], true),
    ]
  }),

  divider(),

  // Section 9 - Known Limitations
  heading1("9. Known Limitations"),

  heading2("9.1 Resolution Mismatch"),
  body("The ML classifier is trained on DeepGlobe imagery at 0.5m/pixel but runs inference on Sentinel-2 at 10m/pixel — a 20× resolution gap. Surface predictions are directional rather than ground-truth quality, particularly for narrow or residential roads where spectral signatures are mixed with adjacent land cover."),

  heading2("9.2 OSM Coverage in Rural Areas"),
  body("In sparsely mapped regions, route endpoints may snap 100–1,500m from the actual destination due to missing road data. The smart bbox expansion in Step 9b partially mitigates this, but it cannot add roads that do not exist in OpenStreetMap."),

  heading2("9.3 Response Time"),
  body("Cold requests take 18–25 seconds end-to-end, dominated by the Sentinel-2 GEE fetch (~15–20 sec). Repeated requests to the same city area benefit from LRU graph caching and typically complete in 8–12 seconds."),

  heading2("9.4 Single-City Scale"),
  body("A 50km haversine guard prevents cross-state routing. The system is optimised for intra-city and short inter-city routes. Long-distance routing would require tile-based graph management and incremental satellite fetching."),

  divider(),

  // Section 10 - File Map
  heading1("10. File Map"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 6560],
    rows: [
      headerRow(["File", "Purpose"], [2800, 6560]),
      dataRow(["app.py", "FastAPI server — all API endpoints"], [2800, 6560], false),
      dataRow(["navigate.py", "Core routing pipeline: geocoding → OSM graph → ML projection → Dijkstra → traffic → response (~1,370 lines)"], [2800, 6560], true),
      dataRow(["navigator.html", "Frontend SPA — full UI (~2,040 lines)"], [2800, 6560], false),
      dataRow(["viewer.html", "Satellite viewer page"], [2800, 6560], true),
      dataRow(["models.py", "U-Net model architecture (Stage 1: road/landcover/building)"], [2800, 6560], false),
      dataRow(["road_type_classifier.py", "Surface classifier (RF preferred, KMeans fallback)"], [2800, 6560], true),
      dataRow(["train_surface_rf.py", "RF training script → produces surface_rf.pkl"], [2800, 6560], false),
      dataRow(["road_width.py", "Road width estimator (morphological)"], [2800, 6560], true),
      dataRow(["road_graph.py", "Vehicle cost profiles and edge weighting"], [2800, 6560], false),
      dataRow(["pipeline.py", "ML inference orchestrator (satellite → prediction)"], [2800, 6560], true),
      dataRow(["inference.py", "Model loading and prediction utilities"], [2800, 6560], false),
      dataRow(["osm_connector.py", "OSM data → 512×512 road mask"], [2800, 6560], true),
      dataRow(["sentinel_connector.py", "Sentinel-2 imagery fetcher (Google Earth Engine)"], [2800, 6560], false),
      dataRow(["road_router.py", "Routing utilities"], [2800, 6560], true),
      dataRow(["surface_urban_patch.py", "Urban surface correction module"], [2800, 6560], false),
      dataRow(["patch_features.csv", "Training feature data"], [2800, 6560], true),
      dataRow(["requirements.txt", "Python dependencies"], [2800, 6560], false),
      dataRow(["Dockerfile", "HuggingFace Spaces deployment configuration"], [2800, 6560], true),
      dataRow(["train_*.py", "Training scripts (road, landcover, building, inpainting)"], [2800, 6560], false),
    ]
  }),

  divider(),

  heading1("11. Experimental Results"),
  body("The RoadSense routing engine was evaluated in a real-world urban environment to validate its core routing capabilities, specifically comparing the fastest and safest route generation under live traffic and varying surface conditions. The following demonstration highlights the system's performance on a route through central Delhi."),
  
  heading2("Case Study: Delhi Urban Navigation"),
  bulletMixed([{ text: "Origin: ", bold: true }, { text: "Supreme Court of India" }]),
  bulletMixed([{ text: "Destination: ", bold: true }, { text: "India Gate" }]),
  
  heading3("Route Generation & Comparison"),
  body("The system successfully computed two distinct paths: a time-optimized fastest route and a damage-minimizing safest route. The fastest route prioritizes higher-speed segments and optimal traffic flow, while the safest route explicitly avoids segments classified as 'damaged' or 'unpaved' by the Random Forest model."),
  noteBox("Figure Placeholder: Fastest vs Safest Route Comparison", "info"),
  
  heading3("Surface-Aware Routing & Damage Warnings"),
  body("During the graph projection phase, the ML pipeline successfully overlaid Sentinel-2 derived surface labels onto the OSM network. The route inspection module revealed that the fastest route traversed a known degraded segment, triggering a proactive road damage warning on the frontend. The safest route correctly bypassed this segment, yielding a slightly longer travel time but significantly higher surface quality."),
  noteBox("Figure Placeholder: Surface-Aware Route Inspection", "info"),
  
  heading3("Traffic Integration & Last-Mile Detection"),
  body("Live Mappls traffic data was successfully integrated, adjusting the baseline ETA to reflect current congestion levels. Furthermore, the haversine-based last-mile gap detection accurately identified when the OSM road graph terminated prior to the exact destination coordinates, rendering a dashed connector line to bridge the visual gap."),

  divider(),

  heading1("12. System Architecture Diagram"),
  body("The diagram below illustrates the complete end-to-end data flow of the RoadSense system, from user input to final visualization:"),
  ...spacer(1),
  code("User Input"),
  code("↓"),
  code("Autocomplete"),
  code("↓"),
  code("Pin Confirmation"),
  code("↓"),
  code("Navigate API"),
  code("↓"),
  code("OSM Graph Construction + Sentinel-2 Fetch"),
  code("↓"),
  code("OSM Raster Mask"),
  code("↓"),
  code("U-Net Road Extraction"),
  code("↓"),
  code("Random Forest Surface Classification"),
  code("↓"),
  code("Edge Surface Projection"),
  code("↓"),
  code("Route Weighting"),
  code("↓"),
  code("Dijkstra Routing"),
  code("↓"),
  code("Traffic Integration"),
  code("↓"),
  code("Fastest + Safest Routes"),
  code("↓"),
  code("Frontend Visualization"),
  ...spacer(1),
  body("This architecture ensures robust separation of concerns, executing heavy network I/O in parallel while maintaining a deterministic pipeline for graph weighting and shortest-path computation."),

  divider(),

  heading1("13. References"),
  bullet("DeepGlobe Road Extraction Challenge"),
  bullet("OpenStreetMap"),
  bullet("OSMnx"),
  bullet("NetworkX"),
  bullet("Sentinel-2"),
  bullet("Google Earth Engine"),
  bullet("Scikit-Learn"),
  bullet("PyTorch"),
  bullet("Mappls APIs"),

];

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22, color: DARK } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "333366" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
            children: [
              new TextRun({ text: "🛰️  RoadSense — Full System Architecture", font: "Arial", size: 18, color: ACCENT, bold: true }),
              new TextRun({ text: "   |   Technical Reference Document", font: "Arial", size: 18, color: GREY_TEXT }),
            ]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 18, color: GREY_TEXT }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: GREY_TEXT }),
              new TextRun({ text: "  |  huggingface.co/spaces/Ayushks07/updated_road_extraction", font: "Arial", size: 18, color: GREY_TEXT }),
            ]
          })
        ]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("RoadSense_Architecture.docx", buf);
  console.log("Done!");
});