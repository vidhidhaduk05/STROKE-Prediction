"""
Unit tests for the stroke treatment-benefit prediction pipeline.

Tests each module with small synthetic data to verify correctness
without requiring the full IST dataset download.
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from utils import sigmoid, compute_metrics, smd, benefit_score, bootstrap_metrics
from counterfactual import build_counterfactual_matrices, make_recommendation
from iptw import compute_stabilized_weights, estimate_propensity
from data_prep import engineer_features, add_interaction_terms, THRESHOLDS, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_ist_row():
    """A single row mimicking IST column structure."""
    return pd.DataFrame([{
        "AGE": 75, "SEX": "M", "RATRIAL": "N", "RSBP": 150,
        "RCONSC": "F", "RDELAY": 4, "STYPE": "PACS",
        "RDEF1": "Y", "RDEF2": "N", "RDEF3": "N", "RDEF4": "N",
        "RDEF5": "N", "RDEF6": "N", "RDEF7": "N", "RDEF8": "N",
        "RASP3": "N", "RVISINF": "N", "RXHEP": "N",
        "RXASP": "Y", "OCCODE": "3", "COUNTRY": "UK",
    }])


@pytest.fixture
def synthetic_feature_matrix():
    """Small feature matrix for modeling tests."""
    np.random.seed(42)
    n = 200
    X = np.random.binomial(1, 0.3, size=(n, 5)).astype(float)
    treatment = np.random.binomial(1, 0.5, size=n)
    y = np.random.binomial(1, 0.3, size=n)
    # Add treatment column and interactions
    X_full = np.column_stack([X, treatment, treatment[:, None] * X])
    cols = [f"f{i}" for i in range(5)] + ["treatment"] + [f"treat_x_f{i}" for i in range(5)]
    return X_full, y, treatment, cols


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------
class TestSigmoid:
    def test_sigmoid_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert sigmoid(100) == pytest.approx(1.0)

    def test_sigmoid_large_negative(self):
        assert sigmoid(-100) == pytest.approx(0.0)

    def test_sigmoid_array(self):
        x = np.array([-1, 0, 1])
        result = sigmoid(x)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.5)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y, prob)
        assert m["AUC"] == pytest.approx(1.0)
        assert m["Accuracy"] == pytest.approx(1.0)
        assert m["Sensitivity"] == pytest.approx(1.0)
        assert m["Specificity"] == pytest.approx(1.0)

    def test_random_prediction(self):
        np.random.seed(42)
        y = np.array([0, 1, 0, 1, 0, 1])
        prob = np.array([0.4, 0.6, 0.5, 0.5, 0.3, 0.7])
        m = compute_metrics(y, prob)
        assert 0.0 <= m["AUC"] <= 1.0
        assert 0.0 <= m["Brier"] <= 1.0


class TestSMD:
    def test_identical_groups(self):
        g1 = np.array([1, 2, 3, 4, 5], dtype=float)
        g2 = np.array([1, 2, 3, 4, 5], dtype=float)
        assert smd(g1, g2) == pytest.approx(0.0)

    def test_different_groups(self):
        g1 = np.array([10, 11, 12, 13, 14], dtype=float)
        g2 = np.array([0, 1, 2, 3, 4], dtype=float)
        assert smd(g1, g2) > 0


class TestBenefitScore:
    def test_score_range(self):
        cate = np.array([-0.05, 0.0, 0.05])
        scores = benefit_score(cate)
        assert scores.min() >= 1.0
        assert scores.max() <= 20.0

    def test_constant_cate(self):
        cate = np.array([0.01, 0.01, 0.01])
        scores = benefit_score(cate)
        assert np.all(scores == scores[0])


# ---------------------------------------------------------------------------
# Data prep tests
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_binarization(self, synthetic_ist_row):
        feat = engineer_features(synthetic_ist_row)
        assert feat["age_gt80"].iloc[0] == 0  # 75 < 80
        assert feat["male"].iloc[0] == 1
        assert feat["afib"].iloc[0] == 0
        assert feat["impaired_conscious"].iloc[0] == 0  # F = alert
        assert feat["tacs"].iloc[0] == 0  # PACS
        assert feat["deficit_ge3"].iloc[0] == 0  # only 1 deficit
        assert feat["treatment"].iloc[0] == 1  # RXASP = Y
        assert feat["favorable"].iloc[0] == 1  # OCCODE = 3

    def test_age_threshold(self):
        df = pd.DataFrame([{
            "AGE": 81, "SEX": "F", "RATRIAL": "N", "RSBP": 120,
            "RCONSC": "F", "RDELAY": 2, "STYPE": "LACS",
            "RDEF1": "N", "RDEF2": "N", "RDEF3": "N", "RDEF4": "N",
            "RDEF5": "N", "RDEF6": "N", "RDEF7": "N", "RDEF8": "N",
            "RASP3": "N", "RVISINF": "N", "RXHEP": "N",
            "RXASP": "N", "OCCODE": "1", "COUNTRY": "UK",
        }])
        feat = engineer_features(df)
        assert feat["age_gt80"].iloc[0] == 1
        assert feat["favorable"].iloc[0] == 0  # OCCODE = 1

    def test_interaction_terms(self, synthetic_ist_row):
        feat = engineer_features(synthetic_ist_row)
        feat = add_interaction_terms(feat)
        for fname in FEATURE_NAMES:
            col = f"treat_x_{fname}"
            assert col in feat.columns
            assert feat[col].iloc[0] == feat["treatment"].iloc[0] * feat[fname].iloc[0]


# ---------------------------------------------------------------------------
# IPTW tests
# ---------------------------------------------------------------------------
class TestIPTW:
    def test_stabilized_weights(self):
        treatment = np.array([1, 1, 0, 0, 1, 0])
        ps = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        weights = compute_stabilized_weights(treatment, ps, truncate=False)
        # When PS = 0.5 and treatment rate = 0.5, weights should be 1.0
        assert np.allclose(weights, 1.0)

    def test_propensity_range(self):
        np.random.seed(42)
        X = np.random.binomial(1, 0.3, size=(100, 5))
        t = np.random.binomial(1, 0.5, size=100)
        model, ps = estimate_propensity(X, t)
        assert np.all(ps > 0) and np.all(ps < 1)


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------
class TestCounterfactual:
    def test_counterfactual_matrices(self):
        """Verify that setting treatment=1/0 correctly updates interaction terms."""
        X = np.array([
            [1, 0, 1, 0, 0],  # main effects
            [0, 1, 0, 1, 0],
        ], dtype=float)
        # Add treatment and interaction columns
        treatment = np.array([1, 0])
        X_full = np.column_stack([X, treatment, treatment[:, None] * X])
        cols = ["f0", "f1", "f2", "f3", "f4", "treatment",
                "treat_x_f0", "treat_x_f1", "treat_x_f2", "treat_x_f3", "treat_x_f4"]
        t_idx = cols.index("treatment")

        X_t, X_c = build_counterfactual_matrices(X_full, cols, t_idx)

        # Treated: treatment=1, interactions = main effects
        assert X_t[0, t_idx] == 1
        assert X_t[1, t_idx] == 1
        assert X_t[0, t_idx + 1] == X_t[0, 0]  # treat_x_f0 = f0
        assert X_t[1, t_idx + 2] == X_t[1, 1]  # treat_x_f1 = f1

        # Control: treatment=0, interactions = 0
        assert X_c[0, t_idx] == 0
        assert X_c[1, t_idx] == 0
        assert X_c[0, t_idx + 1] == 0  # treat_x_f0 = 0
        assert X_c[1, t_idx + 2] == 0  # treat_x_f1 = 0

    def test_recommendation_thresholds(self):
        cate = np.array([0.06, 0.03, -0.03, -0.06])
        recs = make_recommendation(cate, threshold=0.05)
        assert recs[0] == "Treat (aspirin)"
        assert recs[1] == "No clear benefit"
        assert recs[2] == "No clear benefit"
        assert recs[3] == "No treatment (control)"


# ---------------------------------------------------------------------------
# Integration test (small scale)
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_end_to_end_small(self):
        """Small end-to-end test with synthetic data."""
        from sklearn.linear_model import LogisticRegression
        np.random.seed(42)
        n = 300
        X_main = np.random.binomial(1, 0.3, size=(n, 5)).astype(float)
        treatment = np.random.binomial(1, 0.5, size=n)
        y = ((X_main[:, 0] + treatment * 0.3 + np.random.normal(0, 0.5, n)) > 0.5).astype(int)

        X_full = np.column_stack([X_main, treatment, treatment[:, None] * X_main])
        cols = [f"f{i}" for i in range(5)] + ["treatment"] + [f"treat_x_f{i}" for i in range(5)]

        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        model.fit(X_full, y)
        y_prob = model.predict_proba(X_full)[:, 1]
        m = compute_metrics(y, y_prob)
        assert m["AUC"] > 0.5  # Better than chance

        # Counterfactual
        t_idx = cols.index("treatment")
        X_t, X_c = build_counterfactual_matrices(X_full, cols, t_idx)
        prob_t = model.predict_proba(X_t)[:, 1]
        prob_c = model.predict_proba(X_c)[:, 1]
        cate = prob_t - prob_c
        assert cate.shape == (n,)
