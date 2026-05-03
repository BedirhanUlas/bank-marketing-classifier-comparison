import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import build_pipelines, evaluate_model, preprocess


@pytest.fixture
def sample_df():
    """Minimal bank marketing DataFrame for testing."""
    return pd.DataFrame({
        "age": [30, 45, 25, 60, 35],
        "job": ["admin.", "blue-collar", "student", "retired", "admin."],
        "marital": ["married", "single", "single", "married", "divorced"],
        "education": ["university.degree", "basic.4y", "high.school", "basic.6y", "university.degree"],
        "default": ["no", "no", "no", "no", "no"],
        "housing": ["yes", "no", "no", "yes", "yes"],
        "loan": ["no", "no", "no", "yes", "no"],
        "contact": ["cellular", "telephone", "cellular", "cellular", "telephone"],
        "month": ["may", "jun", "jul", "aug", "sep"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
        "duration": [200, 150, 300, 100, 250],
        "campaign": [1, 2, 1, 3, 1],
        "pdays": [999, 999, 5, 999, 10],
        "previous": [0, 0, 1, 0, 2],
        "poutcome": ["nonexistent", "nonexistent", "success", "nonexistent", "success"],
        "emp.var.rate": [-1.8, 1.1, -0.1, 1.4, -3.4],
        "cons.price.idx": [92.893, 93.994, 93.200, 94.001, 92.379],
        "cons.conf.idx": [-46.2, -36.4, -42.0, -35.0, -50.0],
        "euribor3m": [1.354, 4.857, 4.120, 4.962, 0.635],
        "nr.employed": [5099.1, 5228.1, 5195.8, 5228.1, 5017.5],
        "y": ["yes", "no", "yes", "no", "yes"],
    })


def test_preprocess_shape(sample_df):
    X, y, feature_names = preprocess(sample_df)
    assert X.shape[0] == 5
    assert len(y) == 5
    assert len(feature_names) == X.shape[1]


def test_preprocess_target_binary(sample_df):
    _, y, _ = preprocess(sample_df)
    assert set(y).issubset({0, 1})


def test_preprocess_no_nan(sample_df):
    X, y, _ = preprocess(sample_df)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_build_pipelines_returns_four_models():
    pipelines = build_pipelines()
    assert len(pipelines) == 4
    assert all(isinstance(p, Pipeline) for p in pipelines.values())


def test_evaluate_model_returns_expected_keys(sample_df):
    X, y, _ = preprocess(sample_df)
    pipe = Pipeline([("clf", LogisticRegression(max_iter=1000))])
    metrics = evaluate_model("LR", pipe, X, y, X, y)
    assert "Test Accuracy" in metrics
    assert "F1" in metrics
    assert "Train Time (s)" in metrics


def test_test_accuracy_between_zero_and_one(sample_df):
    X, y, _ = preprocess(sample_df)
    pipe = Pipeline([("clf", LogisticRegression(max_iter=1000))])
    metrics = evaluate_model("LR", pipe, X, y, X, y)
    assert 0.0 <= metrics["Test Accuracy"] <= 1.0
