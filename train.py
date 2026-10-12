"""Train and evaluate fraud detection pipelines on the Kaggle credit card fraud dataset.

Usage:
    FRAUD_CSV=/path/to/creditcard.csv python train.py

Everything that learns from data (scaler, SMOTE, model) lives inside one
imblearn Pipeline, so it is fit on the training split only.
"""
import os
import sys

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


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


def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    for name in MODELS:
        for balancing in BALANCING:
            pipe = build_pipeline(make_model(name, balancing, pos_weight), balancing)
            pipe.fit(X_train, y_train)
            print(f"fitted {name} + {balancing}")


if __name__ == "__main__":
    main()
