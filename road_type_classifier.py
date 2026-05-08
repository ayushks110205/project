# =============================================================================
# road_type_classifier.py  –  Tier 1 Module 2: Road Surface Type Classifier
# =============================================================================
#
# Classifies road surface type (paved / unpaved / construction+damaged) using
# unsupervised KMeans clustering on texture, colour, and edge-density features
# extracted from 16×16 patches centred on sampled skeleton points.
#
# Since no ground-truth surface labels exist for DeepGlobe, cluster centres are
# labelled post-hoc by heuristics:
#   • Lowest contrast  → paved   (smooth, reflective asphalt)
#   • Highest edge density → damaged/construction (broken surface, debris)
#   • Remaining cluster → unpaved/dirt
#
# Usage:
#   from road_type_classifier import RoadTypeClassifier
#   clf = RoadTypeClassifier()
#   clf.fit([image1_np, ...], [mask1_np, ...])          # optional batch fit
#   result = clf.predict(image_np, road_mask_np, width_result)
#
# All arrays are pure numpy — no file I/O, no torch.
# =============================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.morphology import medial_axis

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """
    Configuration for the road surface type classifier.

    Attributes:
        patch_size     : Side length of square patch extracted per skeleton point.
        n_samples      : Target number of skeleton points to sample per image.
        n_clusters     : KMeans cluster count (fixed at 3: paved/unpaved/damaged).
        glcm_distances : GLCM inter-pixel distances.
        glcm_angles    : GLCM angles in radians (0, 45, 90, 135 degrees).
        random_state   : Reproducibility seed for KMeans and sampling.
    """
    patch_size:     int  = 16
    n_samples:      int  = 200
    n_clusters:     int  = 3
    glcm_distances: List[int]   = None
    glcm_angles:    List[float] = None
    random_state:   int  = 42

    def __post_init__(self) -> None:
        if self.glcm_distances is None:
            self.glcm_distances = [1, 2]
        if self.glcm_angles is None:
            self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SURFACE_TYPES: List[str] = ['paved', 'unpaved', 'damaged']

# Overlay colours (RGB) for visualisation
_SURFACE_COLORS: Dict[str, np.ndarray] = {
    'paved':   np.array([  0, 200,   0], dtype=np.uint8),   # green
    'unpaved': np.array([255, 140,   0], dtype=np.uint8),   # orange
    'damaged': np.array([220,  20,  60], dtype=np.uint8),   # crimson
}

# GLCM property names to extract
_GLCM_PROPS: Tuple[str, ...] = (
    'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patch_features(patch_rgb: np.ndarray,
                             cfg:       ClassifierConfig) -> np.ndarray:
    """
    Compute a feature vector for a single 16×16 RGB patch.

    Features (concatenated):
        • GLCM properties per (distance, angle) pair:
          contrast, dissimilarity, homogeneity, energy, correlation
          → len = 5 props × 2 distances × 4 angles = 40 values
        • RGB channel statistics: mean and std per channel → 6 values
        • Edge density: Sobel edge count / patch area → 1 value

        Total feature length = 47

    Args:
        patch_rgb : (P, P, 3) uint8 RGB patch.
        cfg       : :class:`ClassifierConfig`.

    Returns:
        feature_vector : (47,) float64 1-D array.
    """
    # ── Greyscale for GLCM ────────────────────────────────────────────────────
    grey = (0.299 * patch_rgb[:, :, 0].astype(np.float64) +
            0.587 * patch_rgb[:, :, 1].astype(np.float64) +
            0.114 * patch_rgb[:, :, 2].astype(np.float64))
    grey_u8 = grey.astype(np.uint8)

    # ── GLCM texture features ─────────────────────────────────────────────────
    glcm = graycomatrix(
        grey_u8,
        distances=cfg.glcm_distances,
        angles=cfg.glcm_angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    glcm_feats: List[float] = []
    for prop in _GLCM_PROPS:
        vals = graycoprops(glcm, prop)   # (n_distances, n_angles)
        glcm_feats.extend(vals.ravel().tolist())

    # ── RGB channel statistics ────────────────────────────────────────────────
    rgb_norm = patch_rgb.astype(np.float64) / 255.0
    colour_feats: List[float] = []
    for ch in range(3):
        colour_feats.append(float(np.mean(rgb_norm[:, :, ch])))
        colour_feats.append(float(np.std(rgb_norm[:, :, ch])))

    # ── Edge density (Sobel) ──────────────────────────────────────────────────
    edge_map   = sobel(grey / 255.0)
    edge_thresh = 0.05
    edge_density = float((edge_map > edge_thresh).sum()) / (grey_u8.size)

    return np.array(glcm_feats + colour_feats + [edge_density], dtype=np.float64)


def _sample_skeleton_points(skeleton: np.ndarray,
                             n:        int,
                             rng:      np.random.Generator) -> np.ndarray:
    """
    Randomly sample up to *n* pixel coordinates from the skeleton.

    Args:
        skeleton : (H, W) bool array.
        n        : number of samples requested.
        rng      : numpy random generator.

    Returns:
        points : (K, 2) int array of (row, col) pairs, K ≤ n.
    """
    rows, cols = np.where(skeleton)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int32)
    k = min(n, len(rows))
    idx = rng.choice(len(rows), size=k, replace=False)
    return np.stack([rows[idx], cols[idx]], axis=1)


def _extract_patches(image_rgb:  np.ndarray,
                     points:     np.ndarray,
                     patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centred patches from *image_rgb* at the given skeleton *points*.
    Patches that would fall (even partially) outside the image boundary are
    discarded.

    Args:
        image_rgb  : (H, W, 3) uint8.
        points     : (K, 2) int array of (row, col).
        patch_size : side length P (should be even).

    Returns:
        patches      : (M, P, P, 3) uint8, M ≤ K.
        valid_points : (M, 2) int, the subset of *points* whose patch fits.
    """
    H, W   = image_rgb.shape[:2]
    half   = patch_size // 2
    valid_patches: List[np.ndarray] = []
    valid_pts:     List[np.ndarray] = []

    for pt in points:
        r, c = int(pt[0]), int(pt[1])
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            continue
        valid_patches.append(image_rgb[r0:r1, c0:c1])
        valid_pts.append(pt)

    if not valid_patches:
        return (np.empty((0, patch_size, patch_size, 3), dtype=np.uint8),
                np.empty((0, 2), dtype=np.int32))

    return np.stack(valid_patches), np.stack(valid_pts)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster → surface-type heuristic labelling
# ─────────────────────────────────────────────────────────────────────────────

def _label_clusters(cluster_centers: np.ndarray,
                    feature_index:   Dict[str, int]) -> Dict[int, str]:
    """
    Assign surface-type labels to KMeans cluster centres using heuristics.

    Heuristic rules (applied in priority order):
        1. Highest edge_density  → 'damaged'
        2. Lowest contrast       → 'paved'
        3. Remaining cluster     → 'unpaved'

    Args:
        cluster_centers : (n_clusters, n_features) float64.
        feature_index   : mapping from feature name to column index.

    Returns:
        label_map : dict mapping cluster index → surface type string.
    """
    n = len(cluster_centers)
    contrast_idx     = feature_index['contrast']
    edge_density_idx = feature_index['edge_density']

    contrast_scores     = cluster_centers[:, contrast_idx]
    edge_density_scores = cluster_centers[:, edge_density_idx]

    label_map: Dict[int, str] = {}

    # Priority 1: highest edge density → damaged
    damaged_idx = int(np.argmax(edge_density_scores))
    label_map[damaged_idx] = 'damaged'

    # Priority 2: lowest contrast among remaining → paved
    remaining = [i for i in range(n) if i not in label_map]
    paved_idx = remaining[int(np.argmin(contrast_scores[remaining]))]
    label_map[paved_idx] = 'paved'

    # Priority 3: leftover → unpaved
    for i in range(n):
        if i not in label_map:
            label_map[i] = 'unpaved'

    return label_map


# ─────────────────────────────────────────────────────────────────────────────
# Main Classifier
# ─────────────────────────────────────────────────────────────────────────────

class RoadTypeClassifier:
    """
    Unsupervised KMeans classifier for road surface type.

    Workflow:
        1. ``fit()``     — optional; trains on a batch of (image, mask) pairs.
                           If skipped, ``predict()`` triggers a single-image fit.
        2. ``predict()`` — extracts features for one image and returns per-pixel
                           labels and a colour overlay.

    Args:
        config : :class:`ClassifierConfig`.  Defaults to standard settings.
    """

    def __init__(self, config: ClassifierConfig = None) -> None:
        self.cfg    = config or ClassifierConfig()
        self._rng   = np.random.default_rng(self.cfg.random_state)
        self._kmeans: Optional[KMeans]         = None
        self._scaler: Optional[StandardScaler] = None
        self._label_map: Optional[Dict[int, str]] = None
        self._feature_index: Dict[str, int]    = self._build_feature_index()

    # ── Feature index ─────────────────────────────────────────────────────────

    def _build_feature_index(self) -> Dict[str, int]:
        """
        Build a name→column mapping for the feature vector produced by
        :func:`_extract_patch_features`.

        Returns:
            dict mapping feature group names to their last column index.
        """
        idx = {}
        # GLCM: 5 props × n_distances × n_angles
        glcm_len = (len(_GLCM_PROPS) *
                    len(self.cfg.glcm_distances) *
                    len(self.cfg.glcm_angles))
        # contrast is the first GLCM property, angle-averaged index 0
        idx['contrast'] = 0
        idx['edge_density'] = glcm_len + 6   # after 6 colour stats
        return idx

    # ── Skeleton helper ────────────────────────────────────────────────────────

    @staticmethod
    def _get_skeleton(road_mask: np.ndarray) -> np.ndarray:
        """
        Compute the medial axis skeleton of *road_mask*.

        Args:
            road_mask : (H, W) uint8, values 0/255.

        Returns:
            skeleton : (H, W) bool.
        """
        binary   = (road_mask > 0)
        skeleton = medial_axis(binary)
        return skeleton

    # ── Feature matrix builder ────────────────────────────────────────────────

    def _build_feature_matrix(self,
                               image_rgb: np.ndarray,
                               skeleton:  np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample skeleton points and compute a feature matrix.

        Args:
            image_rgb : (H, W, 3) uint8.
            skeleton  : (H, W) bool.

        Returns:
            features : (M, F) float64 feature matrix.
            points   : (M, 2) int skeleton coordinate pairs.
        """
        points_raw = _sample_skeleton_points(
            skeleton, self.cfg.n_samples, self._rng)

        patches, points = _extract_patches(
            image_rgb, points_raw, self.cfg.patch_size)

        if patches.shape[0] == 0:
            return np.empty((0, 47), dtype=np.float64), points

        feats = np.stack(
            [_extract_patch_features(p, self.cfg) for p in patches])
        return feats, points

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self,
            images:     List[np.ndarray],
            road_masks: List[np.ndarray]) -> 'RoadTypeClassifier':
        """
        Fit the KMeans model on a batch of satellite images and road masks.

        Collects texture feature vectors from all images, fits a shared
        StandardScaler, then fits KMeans(k=3).  After fitting, cluster centres
        are labelled by the heuristic described in :func:`_label_clusters`.

        Args:
            images     : list of (H, W, 3) uint8 RGB arrays.
            road_masks : list of (H, W) uint8 binary masks (0/255).
                         Must be the same length as *images*.

        Returns:
            self (for chaining).

        Raises:
            ValueError : If *images* and *road_masks* have different lengths or
                         no valid feature vectors could be extracted.
        """
        if len(images) != len(road_masks):
            raise ValueError(
                f"images ({len(images)}) and road_masks ({len(road_masks)}) "
                "must have the same length.")

        all_feats: List[np.ndarray] = []
        for img, mask in zip(images, road_masks):
            skeleton = self._get_skeleton(mask)
            feats, _ = self._build_feature_matrix(img, skeleton)
            if feats.shape[0] > 0:
                all_feats.append(feats)

        if not all_feats:
            raise ValueError("No valid patches extracted — check masks/images.")

        X = np.vstack(all_feats)

        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._kmeans = KMeans(
            n_clusters=self.cfg.n_clusters,
            random_state=self.cfg.random_state,
            n_init=10,
        )
        self._kmeans.fit(X_scaled)

        # Label clusters using heuristics on un-scaled centres for interpretability
        centers_original = self._scaler.inverse_transform(
            self._kmeans.cluster_centers_)
        self._label_map = _label_clusters(centers_original, self._feature_index)

        return self

    def predict(self,
                image_rgb: np.ndarray,
                road_mask: np.ndarray,
                width_result=None) -> dict:
        """
        Classify road surface type for a single image.

        If the classifier has not been fitted via :meth:`fit`, it is
        automatically fitted on this single image first.

        Args:
            image_rgb    : (H, W, 3) uint8 RGB satellite image.
            road_mask    : (H, W) uint8 binary road mask (0/255).
            width_result : optional :class:`road_width.RoadWidthResult`;
                           its skeleton is reused if provided to avoid
                           recomputing the medial axis.

        Returns:
            dict with keys:
                ``surface_map``    — (H, W) str object array, label per skeleton px.
                ``confidence_map`` — (H, W) float32, KMeans cluster distance
                                     normalised to [0, 1] (1 = most confident).
                ``summary``        — dict with dominant type, type counts, etc.
                ``overlay_rgb``    — (H, W, 3) uint8 colour overlay.
                ``skeleton``       — (H, W) bool skeleton used.
                ``is_empty``       — True if no road pixels found.
        """
        H, W = road_mask.shape[:2]

        # Reuse skeleton from width_result if available
        if width_result is not None and hasattr(width_result, 'skeleton'):
            skeleton = width_result.skeleton
        else:
            skeleton = self._get_skeleton(road_mask)

        if not skeleton.any():
            return self._empty_predict_result(H, W, skeleton)

        feats, points = self._build_feature_matrix(image_rgb, skeleton)

        if feats.shape[0] == 0:
            return self._empty_predict_result(H, W, skeleton)

        # Auto-fit on this single image if not fitted
        if self._kmeans is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.fit([image_rgb], [road_mask])

        X_scaled = self._scaler.transform(feats)
        cluster_ids = self._kmeans.predict(X_scaled)

        # Distances to all cluster centres → confidence
        dists = self._kmeans.transform(X_scaled)   # (M, k)
        min_dists = dists[np.arange(len(dists)), cluster_ids]
        # Invert & normalise: low distance → high confidence
        max_dist = min_dists.max() if min_dists.max() > 0 else 1.0
        confidence = (1.0 - min_dists / max_dist).astype(np.float32)

        # Build output maps
        surface_map    = np.full((H, W), '', dtype=object)
        confidence_map = np.zeros((H, W), dtype=np.float32)
        overlay_rgb    = np.zeros((H, W, 3), dtype=np.uint8)

        for (r, c), cid, conf in zip(points, cluster_ids, confidence):
            label = self._label_map[int(cid)]
            surface_map[r, c]    = label
            confidence_map[r, c] = conf
            overlay_rgb[r, c]    = _SURFACE_COLORS[label]

        # ── KDTree propagation: label ALL skeleton pixels ─────────────────────
        # The sampling loop above only labelled `points` (≤ n_samples).
        # Build a KDTree on those sampled coords and propagate their label
        # to every remaining skeleton pixel via nearest-neighbour lookup.
        all_skel_rows, all_skel_cols = np.where(skeleton)
        all_skel_pts = np.stack(
            [all_skel_rows, all_skel_cols], axis=1).astype(np.float32)

        tree = cKDTree(points.astype(np.float32))
        _, nn_idx = tree.query(all_skel_pts, k=1)

        for i, (r, c) in enumerate(zip(all_skel_rows, all_skel_cols)):
            if surface_map[r, c] == '':          # not already labelled
                src_r, src_c = points[nn_idx[i]]
                label = surface_map[src_r, src_c]
                if label != '':
                    surface_map[r, c]    = label
                    overlay_rgb[r, c]    = _SURFACE_COLORS[label]
                    confidence_map[r, c] = confidence_map[src_r, src_c]

        # Summary statistics
        labels_on_skel = surface_map[skeleton]
        labels_on_skel = labels_on_skel[labels_on_skel != '']
        type_counts: Dict[str, int] = {t: 0 for t in SURFACE_TYPES}
        for lbl in labels_on_skel:
            if lbl in type_counts:
                type_counts[lbl] += 1

        dominant = max(type_counts, key=type_counts.get) if labels_on_skel.size > 0 else 'paved'

        summary = {
            'dominant_type':  dominant,
            'type_counts':    type_counts,
            'mean_confidence': float(confidence.mean()),
            'n_sampled_pts':  int(len(points)),
        }

        return {
            'surface_map':    surface_map,
            'confidence_map': confidence_map,
            'summary':        summary,
            'overlay_rgb':    overlay_rgb,
            'skeleton':       skeleton,
            'is_empty':       False,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _empty_predict_result(H: int,
                               W: int,
                               skeleton: np.ndarray) -> dict:
        """Return a degenerate predict result when no road pixels are present."""
        return {
            'surface_map':    np.full((H, W), '', dtype=object),
            'confidence_map': np.zeros((H, W), dtype=np.float32),
            'summary': {
                'dominant_type':   'paved',
                'type_counts':     {t: 0 for t in SURFACE_TYPES},
                'mean_confidence': 0.0,
                'n_sampled_pts':   0,
            },
            'overlay_rgb': np.zeros((H, W, 3), dtype=np.uint8),
            'skeleton':    skeleton,
            'is_empty':    True,
        }
