# =============================================================================
# train_surface_rf.py  –  Supervised Random Forest Surface Classifier Trainer
# =============================================================================
#
# Reads the labelled patch CSV produced by label_patches.py, trains a
# Random Forest classifier, evaluates it with 5-fold cross-validation,
# and saves the result to surface_rf.pkl.
#
# Usage
# -----
#   python train_surface_rf.py
#   python train_surface_rf.py --data patch_features.csv --output surface_rf.pkl
#   python train_surface_rf.py --data patch_features.csv --test-size 0.2
#
# Output
# ------
#   surface_rf.pkl  —  joblib bundle with keys 'scaler' and 'rf'
#                      Loaded automatically by RoadTypeClassifier when present.
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score)

try:
    import joblib
except ImportError:
    print("❌  joblib not found. Install with:  pip install joblib")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    _VIZ = True
except ImportError:
    _VIZ = False

LABELS = ['paved', 'unpaved', 'damaged']


# =============================================================================
# Training
# =============================================================================

def train_rf(
    data_csv:    str   = 'patch_features.csv',
    output_pkl:  str   = 'surface_rf.pkl',
    test_size:   float = 0.20,
    n_estimators: int  = 300,
    random_state: int  = 42,
    verbose:     bool  = True,
) -> dict:
    """
    Train the Random Forest surface classifier.

    Args:
        data_csv     : CSV from label_patches.py  (features + 'label' column)
        output_pkl   : where to save the trained bundle
        test_size    : fraction held out for final evaluation
        n_estimators : number of trees in the forest
        random_state : seed for reproducibility
        verbose      : print progress

    Returns:
        dict with keys: 'rf', 'scaler', 'cv_f1', 'test_f1', 'report'
    """

    # ── 1. Load data ───────────────────────────────────────────────────────────
    if not os.path.isfile(data_csv):
        print(f"❌  Data file not found: '{data_csv}'")
        print("    Run label_patches.py first to generate labels.")
        sys.exit(1)

    df = pd.read_csv(data_csv)
    print(f"📂  Loaded '{data_csv}'  →  {len(df)} rows")

    if 'label' not in df.columns:
        print("❌  'label' column missing from CSV.")
        sys.exit(1)

    # Drop any rows with unknown labels
    df = df[df['label'].isin(LABELS)].reset_index(drop=True)
    print(f"    Valid rows after filtering: {len(df)}")

    if len(df) < 30:
        print("⚠️   Very few samples — results will be unreliable.")
        print("    Label more patches with label_patches.py")

    # Distribution
    counts = df['label'].value_counts()
    print("\n📊  Label distribution:")
    for lbl in LABELS:
        n   = counts.get(lbl, 0)
        pct = n / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"    {lbl:8s}  {n:4d}  ({pct:5.1f}%)  {bar}")

    # ── 2. Features and labels ─────────────────────────────────────────────────
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(np.float64)
    y = df['label'].values

    # ── 3. Normalise features ──────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 4. Cross-validation (stratified 5-fold) ────────────────────────────────
    if verbose:
        print(f"\n🔁  5-fold stratified cross-validation  "
              f"(n_estimators={n_estimators}) …")

    rf_cv = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = None,
        min_samples_leaf = 2,
        class_weight  = 'balanced',
        random_state  = random_state,
        n_jobs        = -1,
    )
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(rf_cv, X_scaled, y, cv=cv,
                             scoring='f1_macro', n_jobs=-1)

    cv_f1 = float(scores.mean())
    cv_std = float(scores.std())
    print(f"    CV macro-F1: {cv_f1:.4f} ± {cv_std:.4f}")
    print(f"    Per-fold:    {['%.3f' % s for s in scores]}")

    if cv_f1 < 0.55:
        print("\n⚠️   Low CV F1 — consider labelling more patches, "
              "especially 'damaged' examples which are rare.")

    # ── 5. Train-test split for held-out evaluation ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = test_size,
        stratify     = y,
        random_state = random_state,
    )

    rf = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = None,
        min_samples_leaf = 2,
        class_weight  = 'balanced',
        random_state  = random_state,
        n_jobs        = -1,
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    report  = classification_report(y_test, y_pred, labels=LABELS,
                                    zero_division=0)

    print(f"\n📋  Hold-out evaluation  ({int(len(y_test))} samples):")
    print(report)

    # ── 6. Feature importance (top 10) ────────────────────────────────────────
    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    print("🌳  Top-10 feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        col = feat_cols[idx] if idx < len(feat_cols) else f'f{idx}'
        print(f"    {rank:2d}.  {col:12s}  {importances[idx]:.4f}")

    # ── 7. Retrain on ALL data for deployment ─────────────────────────────────
    print("\n🔄  Retraining on full dataset for deployment …")
    rf_final = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = None,
        min_samples_leaf = 2,
        class_weight  = 'balanced',
        random_state  = random_state,
        n_jobs        = -1,
    )
    rf_final.fit(X_scaled, y)

    # ── 8. Save bundle ────────────────────────────────────────────────────────
    bundle = {
        'scaler':       scaler,
        'rf':           rf_final,
        'feature_cols': feat_cols,
        'labels':       LABELS,
        'cv_f1':        cv_f1,
        'test_f1':      test_f1,
        'n_train':      len(df),
    }
    joblib.dump(bundle, output_pkl)
    size_kb = os.path.getsize(output_pkl) // 1024
    print(f"💾  Saved → '{output_pkl}'  ({size_kb} KB)")

    # ── 9. Confusion matrix plot (optional) ───────────────────────────────────
    if _VIZ:
        cm      = confusion_matrix(y_test, y_pred, labels=LABELS)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Surface RF  |  test F1={test_f1:.3f}  CV={cv_f1:.3f}')
        plt.tight_layout()
        fig_path = output_pkl.replace('.pkl', '_confusion.png')
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"📊  Confusion matrix → '{fig_path}'")

    print(f"\n✅  Training complete!")
    print(f"    CV macro-F1  : {cv_f1:.4f}")
    print(f"    Test macro-F1: {test_f1:.4f}")
    print(f"\nThe classifier will be loaded automatically by RoadTypeClassifier")
    print(f"when '{output_pkl}' is present in the working directory.")
    print(f"Set env var SURFACE_RF_PATH to use a different path.")

    return {
        'rf':      rf_final,
        'scaler':  scaler,
        'cv_f1':   cv_f1,
        'test_f1': test_f1,
        'report':  report,
    }


# =============================================================================
# Quick model test
# =============================================================================

def quick_test(pkl_path: str = 'surface_rf.pkl') -> None:
    """
    Smoke-test: load the saved bundle and predict on random feature vectors.
    """
    if not os.path.isfile(pkl_path):
        print(f"❌  '{pkl_path}' not found. Run training first.")
        return

    bundle  = joblib.load(pkl_path)
    scaler  = bundle['scaler']
    rf      = bundle['rf']
    labels  = bundle['labels']

    rng      = np.random.default_rng(0)
    X_dummy  = rng.uniform(-1, 1, size=(5, 47))
    X_scaled = scaler.transform(X_dummy)
    preds    = rf.predict(X_scaled)
    probs    = rf.predict_proba(X_scaled)

    print(f"✅  Smoke-test on '{pkl_path}':")
    print(f"    Classes : {rf.classes_}")
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        top_prob = prob.max()
        print(f"    sample {i+1}: {pred:8s}  confidence={top_prob:.3f}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train supervised Random Forest road surface classifier')
    parser.add_argument('--data',         default='patch_features.csv',
                        help='Input CSV from label_patches.py (default: patch_features.csv)')
    parser.add_argument('--output',       default='surface_rf.pkl',
                        help='Output pkl path (default: surface_rf.pkl)')
    parser.add_argument('--test-size',    type=float, default=0.20,
                        help='Fraction for hold-out evaluation (default: 0.20)')
    parser.add_argument('--n-estimators', type=int,   default=300,
                        help='Number of trees (default: 300)')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--test-only',    action='store_true',
                        help='Just run quick_test on an existing pkl')
    args = parser.parse_args()

    if args.test_only:
        quick_test(args.output)
    else:
        train_rf(
            data_csv     = args.data,
            output_pkl   = args.output,
            test_size    = args.test_size,
            n_estimators = args.n_estimators,
            random_state = args.seed,
        )
