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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])


def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipe.fit(X_train, y_train)


if __name__ == "__main__":
    main()
