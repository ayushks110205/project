# =============================================================================
# sentinel_connector.py  —  Sentinel-2 Live Data Connector (Google Earth Engine)
# =============================================================================
#
# Fetches real-time Sentinel-2 (S2_SR_HARMONIZED) imagery from Google Earth
# Engine for a given lat/lon bounding box and date range, and returns a
# 512×512 RGB numpy array ready for the DeepGlobe pipeline.
#
# One-time setup (run once in terminal):
#   pip install earthengine-api requests
#   earthengine authenticate          # opens browser → Google login
#
# Usage:
#   from sentinel_connector import SentinelConnector
#   conn = SentinelConnector()
#   rgb, meta = conn.fetch(
#       bbox       = [77.5, 12.9, 77.7, 13.1],   # [lon_min, lat_min, lon_max, lat_max]
#       start_date = '2024-01-01',
#       end_date   = '2024-03-31',
#   )
#   # rgb  → (512, 512, 3) uint8 numpy array — ready for pipeline
#   # meta → dict with image_count, cloud_pct, bbox, dates
# =============================================================================

from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Optional GEE import ────────────────────────────────────────────────────────
_GEE_OK = False
try:
    import ee
    import requests as _requests
    _GEE_OK = True
except ImportError:
    pass


# =============================================================================
# Constants
# =============================================================================

# Sentinel-2 SR reflectance range for natural-colour display
# Values above this are saturated (clouds, snow) — clamp to max
_S2_MIN = 0
_S2_MAX = 3000

# Cloud cover threshold (%) — images above this are excluded from composite
_CLOUD_THRESH = 20

# Relaxed threshold used as fallback when no clear images found
_CLOUD_THRESH_RELAXED = 60

# Output pixel size fed to the pipeline
_PIPELINE_SIZE = 512

# Collection name
_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'


# =============================================================================
# Main Connector
# =============================================================================

class SentinelConnector:
    """
    Fetches Sentinel-2 imagery from Google Earth Engine.

    Args:
        project_id : GEE Cloud project ID.  If None, uses the project set
                     during ``earthengine authenticate``.
        output_size: Side length (px) of the returned square RGB image.
                     Defaults to 512 to match the DeepGlobe pipeline.
    """

    def __init__(self,
                 project_id:  Optional[str] = None,
                 output_size: int           = _PIPELINE_SIZE) -> None:
        if not _GEE_OK:
            raise RuntimeError(
                "earthengine-api or requests not installed.\n"
                "Run:  pip install earthengine-api requests"
            )

        self._output_size = output_size
        self._project_id  = project_id or os.getenv('GEE_PROJECT', None)

        # Initialise GEE
        try:
            if self._project_id:
                ee.Initialize(project=self._project_id)
            else:
                ee.Initialize()
            print(f"[OK] Google Earth Engine initialised"
                  + (f" (project={self._project_id})" if self._project_id else ""))
        except Exception as exc:
            raise RuntimeError(
                f"GEE initialisation failed: {exc}\n"
                "Run:  earthengine authenticate"
            ) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        bbox:       List[float],
        start_date: str,
        end_date:   str,
        cloud_pct:  float = _CLOUD_THRESH,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fetch a Sentinel-2 median composite for a bounding box and date range.

        Args:
            bbox       : [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
            start_date : ISO date string, e.g. '2024-01-01'.
            end_date   : ISO date string, e.g. '2024-03-31'.
            cloud_pct  : Maximum allowed cloud cover percentage (default 20).

        Returns:
            (rgb, metadata) where:
                rgb      — (512, 512, 3) uint8 RGB numpy array.
                metadata — dict with keys: image_count, cloud_pct_threshold,
                           actual_cloud_pct, bbox, start_date, end_date,
                           relaxed_filter.

        Raises:
            ValueError  : No imagery found for the given bbox / dates.
            RuntimeError: GEE download failed.
        """
        self._validate_bbox(bbox)
        self._validate_dates(start_date, end_date)

        region = ee.Geometry.Rectangle(bbox)

        # ── Build collection ──────────────────────────────────────────────────
        col, relaxed = self._build_collection(region, start_date, end_date, cloud_pct)
        count        = col.size().getInfo()

        if count == 0:
            raise ValueError(
                f"No Sentinel-2 images found for bbox={bbox}, "
                f"dates={start_date}→{end_date}, cloud<{cloud_pct}%.\n"
                "Try a wider date range or higher cloud threshold."
            )

        print(f"[INFO] {count} Sentinel-2 scenes found"
              + (" (relaxed cloud filter)" if relaxed else "")
              + f" — compositing...")

        # ── Median composite ──────────────────────────────────────────────────
        composite = col.median()

        # Estimate mean cloud cover from the metadata (best-effort)
        try:
            mean_cloud = col.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        except Exception:
            mean_cloud = None

        # ── Download thumbnail ────────────────────────────────────────────────
        rgb = self._download_rgb(composite, region)

        metadata = {
            'image_count':         count,
            'cloud_pct_threshold': cloud_pct,
            'actual_cloud_pct':    round(mean_cloud, 2) if mean_cloud is not None else None,
            'bbox':                bbox,
            'start_date':          start_date,
            'end_date':            end_date,
            'relaxed_filter':      relaxed,
            'collection':          _COLLECTION,
            'output_size_px':      self._output_size,
        }

        return rgb, metadata

    @staticmethod
    def is_available() -> bool:
        """Return True if earthengine-api and requests are installed."""
        return _GEE_OK

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_collection(
        self,
        region:     'ee.Geometry',
        start_date: str,
        end_date:   str,
        cloud_pct:  float,
    ) -> Tuple['ee.ImageCollection', bool]:
        """
        Build a filtered Sentinel-2 collection.
        Falls back to a relaxed cloud filter if the strict filter yields 0 images.
        Returns (collection, was_relaxed).
        """
        base = (
            ee.ImageCollection(_COLLECTION)
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select(['B4', 'B3', 'B2'])   # Red, Green, Blue
        )

        strict = base.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
        if strict.size().getInfo() > 0:
            return strict, False

        # Relax cloud filter
        print(f"[WARN] No scenes with cloud < {cloud_pct}% — "
              f"relaxing to {_CLOUD_THRESH_RELAXED}%")
        relaxed = base.filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', _CLOUD_THRESH_RELAXED))
        return relaxed, True

    def _download_rgb(self,
                      composite: 'ee.Image',
                      region:    'ee.Geometry') -> np.ndarray:
        """
        Download composite as PNG via getThumbURL, decode to RGB numpy array,
        and upsample to self._output_size × self._output_size.
        """
        url = composite.getThumbURL({
            'min':        _S2_MIN,
            'max':        _S2_MAX,
            'bands':      ['B4', 'B3', 'B2'],
            'region':     region,
            'dimensions': f'{self._output_size}x{self._output_size}',
            'format':     'png',
        })

        print(f"[INFO] Downloading Sentinel-2 thumbnail...")
        resp = _requests.get(url, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(
                f"GEE thumbnail download failed (HTTP {resp.status_code}).\n"
                f"URL: {url}"
            )

        # Decode PNG bytes → BGR → RGB
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode GEE response as an image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Ensure exact output size (GEE may return slightly different dims)
        if rgb.shape[0] != self._output_size or rgb.shape[1] != self._output_size:
            rgb = cv2.resize(rgb, (self._output_size, self._output_size),
                             interpolation=cv2.INTER_CUBIC)

        print(f"[OK] Downloaded: shape={rgb.shape}  dtype={rgb.dtype}")
        return rgb

    @staticmethod
    def _validate_bbox(bbox: List[float]) -> None:
        if len(bbox) != 4:
            raise ValueError("bbox must be [lon_min, lat_min, lon_max, lat_max]")
        lon_min, lat_min, lon_max, lat_max = bbox
        if lon_min >= lon_max:
            raise ValueError(f"lon_min ({lon_min}) must be < lon_max ({lon_max})")
        if lat_min >= lat_max:
            raise ValueError(f"lat_min ({lat_min}) must be < lat_max ({lat_max})")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError("Longitudes must be in [-180, 180]")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitudes must be in [-90, 90]")

    @staticmethod
    def _validate_dates(start: str, end: str) -> None:
        from datetime import date
        try:
            s = date.fromisoformat(start)
            e = date.fromisoformat(end)
        except ValueError as exc:
            raise ValueError(f"Dates must be YYYY-MM-DD format: {exc}") from exc
        if s >= e:
            raise ValueError(f"start_date ({start}) must be before end_date ({end})")


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Test Sentinel-2 fetch')
    parser.add_argument('--lon-min', type=float, default=77.55,
                        help='Min longitude (default: Bangalore)')
    parser.add_argument('--lat-min', type=float, default=12.92)
    parser.add_argument('--lon-max', type=float, default=77.65)
    parser.add_argument('--lat-max', type=float, default=13.02)
    parser.add_argument('--start',   default='2024-01-01')
    parser.add_argument('--end',     default='2024-03-31')
    parser.add_argument('--project', default=None,
                        help='GEE project ID (optional)')
    parser.add_argument('--out',     default='sentinel_test.png')
    args = parser.parse_args()

    if not _GEE_OK:
        print("ERROR: pip install earthengine-api requests")
        sys.exit(1)

    conn = SentinelConnector(project_id=args.project)
    rgb, meta = conn.fetch(
        bbox       = [args.lon_min, args.lat_min, args.lon_max, args.lat_max],
        start_date = args.start,
        end_date   = args.end,
    )

    cv2.imwrite(args.out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {args.out}")
    print(f"Metadata: {meta}")
