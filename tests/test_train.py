import numpy as np
import pandas as pd
import pytest

from train import BALANCING, FEATURES, MODELS, build_pipeline, make_model, tune_threshold


def synthetic_data(n=300, n_fraud=30, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = pd.Series([1] * n_fraud + [0] * (n - n_fraud))
    X.loc[y == 1, "V14"] -= 3  # give the fraud class a learnable signal
    return X, y


@pytest.mark.parametrize("balancing", BALANCING)
@pytest.mark.parametrize("name", MODELS)
def test_pipeline_fits_and_scores(name, balancing):
    X, y = synthetic_data()
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(make_model(name, balancing, pos_weight), balancing)
    pipe.fit(X, y)
    scores = pipe.predict_proba(X)[:, 1]
    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert ("smote" in pipe.named_steps) == (balancing == "smote")


def test_tune_threshold_separates_classes():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    threshold = tune_threshold(y, scores)
    assert 0.4 < threshold <= 0.8
    assert np.array_equal((scores >= threshold).astype(int), y)
