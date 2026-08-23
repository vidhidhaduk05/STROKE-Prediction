"""
Step 2: Inverse Probability of Treatment Weighting (IPTW).

Estimates propensity scores via logistic regression on all core covariates,
computes stabilized weights truncated at the 1st/99th percentiles, and
verifies covariate balance (standardized mean differences < 0.1).

Note: IST was a randomized trial, so treatment groups should already be
well-balanced. IPTW is implemented here for methodological completeness,
addressing potential confounding by indication.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import smd
from data_prep import FEATURE_NAMES


def estimate_propensity(X_covariates, treatment, random_state=42):
    """
    Fit a propensity score model (logistic regression of treatment on covariates).
    Returns (model, propensity_scores).
    """
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=1000, random_state=random_state,
    )
    model.fit(X_covariates, treatment)
    ps = model.predict_proba(X_covariates)[:, 1]
    return model, ps


def compute_stabilized_weights(treatment, propensity_scores, truncate=True):
    """
    Compute stabilized IPTW weights.
    Stabilized weight = P(T=1) / PS  for treated,  P(T=0) / (1-PS) for control.
    Optionally truncate at 1st and 99th percentiles.
    """
    p_treated = treatment.mean()
    weights = np.where(
        treatment == 1,
        p_treated / np.clip(propensity_scores, 1e-6, 1 - 1e-6),
        (1 - p_treated) / np.clip(1 - propensity_scores, 1e-6, 1 - 1e-6),
    )

    if truncate:
        lo, hi = np.percentile(weights, [1, 99])
        weights = np.clip(weights, lo, hi)

    return weights


def check_balance(df, treatment, weights, feature_names=None):
    """
    Check covariate balance before and after weighting.
    Returns a DataFrame with SMDs for each covariate.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    t = np.asarray(treatment)
    w = np.asarray(weights)

    results = []
    for feat in feature_names:
        vals = df[feat].values.astype(float)

        # Unweighted SMD
        smd_unweighted = smd(vals[t == 1], vals[t == 0])

        # Weighted SMD (using weighted means and variances)
        w1, w0 = w[t == 1], w[t == 0]
        v1, v0 = vals[t == 1], vals[t == 0]
        m1_w = np.sum(w1 * v1) / np.sum(w1)
        m0_w = np.sum(w0 * v0) / np.sum(w0)
        var1_w = np.sum(w1 * (v1 - m1_w) ** 2) / np.sum(w1)
        var0_w = np.sum(w0 * (v0 - m0_w) ** 2) / np.sum(w0)
        pooled_sd_w = np.sqrt((var1_w + var0_w) / 2)
        smd_weighted = abs(m1_w - m0_w) / pooled_sd_w if pooled_sd_w > 0 else 0.0

        results.append({
            "covariate": feat,
            "smd_unweighted": round(smd_unweighted, 4),
            "smd_weighted": round(smd_weighted, 4),
            "balanced": abs(smd_weighted) < 0.1,
        })

    return pd.DataFrame(results)


def run_iptw(df, feature_info, random_state=42):
    """
    Full IPTW pipeline.
    Returns (weights, propensity_model, propensity_scores, balance_df).
    """
    fnames = feature_info["feature_names"]
    X_cov = df[fnames].values.astype(float)
    treatment = df["treatment"].values.astype(int)

    # Estimate propensity scores
    ps_model, ps = estimate_propensity(X_cov, treatment, random_state)

    # Compute stabilized weights
    weights = compute_stabilized_weights(treatment, ps, truncate=True)

    # Check balance
    balance = check_balance(df, treatment, weights, fnames)

    print("=== IPTW Balance Check ===")
    print(f"Propensity score range: [{ps.min():.4f}, {ps.max():.4f}]")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"Weight mean: {weights.mean():.3f}, std: {weights.std():.3f}")
    print(f"\nMax |SMD| unweighted: {balance['smd_unweighted'].abs().max():.4f}")
    print(f"Max |SMD| weighted:   {balance['smd_weighted'].abs().max():.4f}")
    n_balanced = balance["balanced"].sum()
    print(f"Covariates balanced (|SMD| < 0.1): {n_balanced}/{len(balance)}")
    if n_balanced < len(balance):
        unbalanced = balance[~balance["balanced"]]
        print(f"Still unbalanced: {list(unbalanced['covariate'])}")

    return weights, ps_model, ps, balance


if __name__ == "__main__":
    from data_prep import prepare_data, get_feature_matrix
    dev, ext, imp, info = prepare_data()
    weights, ps_model, ps, balance = run_iptw(dev, info)
    print(f"\n{balance.to_string(index=False)}")
