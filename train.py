"""Train and evaluate fraud detection pipelines on the Kaggle credit card fraud dataset.

Usage:
    FRAUD_CSV=/path/to/creditcard.csv python train.py

Everything that learns from data (scaler, SMOTE, model) lives inside one
imblearn Pipeline, so it is fit on the training split only.
"""
import json
import os
import platform
import sys
from pathlib import Path

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "fraud_pipeline.joblib"


def load_data():
    path = os.environ.get("FRAUD_CSV")
    if not path:
        sys.exit("Set the FRAUD_CSV environment variable to the path of creditcard.csv")
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def split(X, y):
    """Stratified 60/20/20 train/validation/test split."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_model(name, balancing, pos_weight):
    """Fixed settings, no tuning. With balancing='class_weight' the model weights
    the fraud class instead of seeing SMOTE-resampled data."""
    weighted = balancing == "class_weight"
    if name == "logreg":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced" if weighted else None, random_state=SEED
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample" if weighted else None, random_state=SEED,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="aucpr",
            scale_pos_weight=pos_weight if weighted else 1.0, n_jobs=-1, random_state=SEED,
        )
    raise ValueError(name)


def build_pipeline(model, balancing):
    steps = [("scaler", StandardScaler())]
    if balancing == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return Pipeline(steps)


MODELS = ["logreg", "random_forest", "xgboost"]
BALANCING = ["smote", "class_weight"]


def tune_threshold(y_true, scores):
    """Threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # last P/R point has no threshold
    return float(thresholds[best])


def evaluate(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


COLORS = {"logreg": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}
LINESTYLES = {"smote": "-", "class_weight": "--"}


def plot_pr_curves(y_test, test_scores, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    for (name, balancing), scores in test_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, color=COLORS[name], linestyle=LINESTYLES[balancing],
                linewidth=2, drawstyle="steps-post",
                label=f"{name} + {balancing} (PR AUC {ap:.3f})")
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, color="#8a8a85", linewidth=1, linestyle=":",
               label=f"random baseline ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves on the held-out test split")
    ax.grid(True, color="#e5e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    results, test_scores, pipelines = [], {}, {}
    for name in MODELS:
        for balancing in BALANCING:
            pipe = build_pipeline(make_model(name, balancing, pos_weight), balancing)
            pipe.fit(X_train, y_train)
            val_scores = pipe.predict_proba(X_val)[:, 1]
            scores = pipe.predict_proba(X_test)[:, 1]
            threshold = tune_threshold(y_val, val_scores)
            results.append({
                "model": name,
                "balancing": balancing,
                "threshold": round(threshold, 6),
                "validation": evaluate(y_val, val_scores, threshold),
                "test": evaluate(y_test, scores, threshold),
            })
            test_scores[(name, balancing)] = scores
            pipelines[(name, balancing)] = pipe
            print(f"{name:14s} {balancing:13s} val PR AUC {results[-1]['validation']['pr_auc']:.4f}"
                  f"  test PR AUC {results[-1]['test']['pr_auc']:.4f}")

    # Selection uses validation PR AUC only; the test split is reported, never used to choose.
    selected = max(results, key=lambda r: r["validation"]["pr_auc"])

    metrics = {
        "dataset": {
            "rows": int(len(y)),
            "fraud_cases": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 6),
        },
        "split": {
            "method": "stratified random 60/20/20",
            "seed": SEED,
            "train": {"rows": int(len(y_train)), "fraud": int(y_train.sum())},
            "validation": {"rows": int(len(y_val)), "fraud": int(y_val.sum())},
            "test": {"rows": int(len(y_test)), "fraud": int(y_test.sum())},
        },
        "threshold_rule": "maximise F1 on the validation split",
        "selection_rule": "highest validation PR AUC",
        "selected": {"model": selected["model"], "balancing": selected["balancing"]},
        "results": results,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    DOCS_DIR.mkdir(exist_ok=True)
    plot_pr_curves(y_test, test_scores, DOCS_DIR / "pr_curve.png")

    # The saved bundle carries the whole fitted pipeline (scaler included) and its threshold.
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": pipelines[(selected["model"], selected["balancing"])],
        "threshold": selected["threshold"],
        "features": FEATURES,
        "model": selected["model"],
        "balancing": selected["balancing"],
    }, PIPELINE_PATH)
    print(f"selected: {selected['model']} + {selected['balancing']} -> {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
