"""
train.py
--------
Train and evaluate pregnancy risk models on the Dodoma dataset.
Saves the best model to models/risk_model.pkl
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")

# Add src to path
# preprocess already defined above in this notebook

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("/content/maternal_dataset_csv.csv")
MODEL_PATH = Path("/content/risk_model.pkl")


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    X = build_features(df)
    y = get_target(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    return X, y


def build_pipelines():
    """Return dict of model pipelines to compare."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    pipelines = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", C=1.0, random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )),
        ]),
    }

    # Try to add LightGBM if available
    try:
        import lightgbm as lgb
        pipelines["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                class_weight="balanced", random_state=42, verbose=-1
            )),
        ])
        print("  LightGBM available ✓")
    except ImportError:
        print("  LightGBM not installed, skipping.")

    return pipelines


def evaluate_pipeline(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name.upper().replace('_',' ')}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:  {auc:.4f}")
    print(report)
    print(f"  Confusion matrix:\n{cm}")

    return auc, pipeline


def get_feature_importance(pipeline, feature_names):
    """Extract feature importances from the last step of a pipeline."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        try:
            auc, fitted = evaluate_pipeline(
                name, pipeline, X_train, X_test, y_train, y_test
            )
            results[name] = (auc, fitted)
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")

    # Pick best model by AUC
    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_pipeline = results[best_name]
    print(f"\n🏆 Best model: {best_name} (AUC = {best_auc:.4f})")

    # Feature importance
    fi = get_feature_importance(best_pipeline, X.columns.tolist())
    if fi is not None:
        print("\nTop 15 important features:")
        print(fi.head(15).to_string())

    # Save model + metadata
    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "auc": best_auc,
        "feature_names": X.columns.tolist(),
        "feature_importances": fi,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train()
