---
title: DeepGlobe Satellite Road Intelligence API
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🛰️ DeepGlobe Satellite Road Intelligence API

FastAPI server exposing the full satellite image analysis pipeline.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`       | Root / liveness check |
| `GET`  | `/health` | Model load status |
| `GET`  | `/docs`   | Interactive Swagger UI |
| `POST` | `/analyze` | Road extraction + Tier 1 (width + surface) |
| `POST` | `/route`   | + Tier 2 vehicle-aware graph routing |
| `POST` | `/full`    | All 4 stages + Tier 1 + Tier 2 |

## Usage

```bash
curl -X POST "https://<space-url>/analyze?include_images=true" \
     -F "file=@satellite.jpg" | python -m json.tool
```
