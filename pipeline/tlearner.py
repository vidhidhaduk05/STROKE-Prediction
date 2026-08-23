"""
Step 7: T-learner HTE benchmark.

Trains separate Elastic Net models on the treated (aspirin) and control
(no aspirin) subgroups (T-learner meta-learner), then compares CATE
estimates against the interaction model. Mirrors paper Fig 3:
Pearson correlation, Bland-Altman plot, clinical recommendation agreement.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

from counterfactual import counterfactual_predict


def train_tlearner(X, y, treatment, weights, params, random_state=42,
                   main_feature_indices=None):
    """
    Train separate models for treated and control groups (T-learner).
    Uses ONLY main effect features (no treatment column, no interactions)
    so both models share the same feature space.
    Returns (model_treated, model_control).
    """
    if main_feature_indices is None:
        # Default: first n_features columns (main effects only)
        # Caller must pass the correct indices
        raise ValueError("main_feature_indices must be provided")

    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_main = X[:, main_feature_indices]

    # Treated model
    model_t = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=params["C"], l1_ratio=params["l1_ratio"],
        max_iter=5000, class_weight="balanced",
        random_state=random_state,
    )
    model_t.fit(X_main[treated_mask], y[treated_mask],
                sample_weight=weights[treated_mask])

    # Control model
    model_c = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=params["C"], l1_ratio=params["l1_ratio"],
        max_iter=5000, class_weight="balanced",
        random_state=random_state,
    )
    model_c.fit(X_main[control_mask], y[control_mask],
                sample_weight=weights[control_mask])

    return model_t, model_c


def tlearner_cate(model_t, model_c, X, main_feature_indices):
    """
    Compute CATE using the T-learner approach.
    Both models predict on main-effect features only.
    CATE_T = P_t(favorable) - P_c(favorable).
    """
    X_main = X[:, main_feature_indices]
    prob_t = model_t.predict_proba(X_main)[:, 1]
    prob_c = model_c.predict_proba(X_main)[:, 1]
    cate = prob_t - prob_c
    return prob_t, prob_c, cate


def bland_altman(cate_interaction, cate_tlearner):
    """
    Bland-Altman analysis: mean difference and 95% limits of agreement.
    """
    diff = cate_interaction - cate_tlearner
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_low = mean_diff - 1.96 * std_diff
    loa_high = mean_diff + 1.96 * std_diff
    return {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "diffs": diff.tolist(),
    }


def recommendation_agreement(cate_int, cate_tl, threshold=0.05):
    """
    Check agreement in clinical recommendations between the two methods.
    """
    def rec(cate):
        return np.where(cate > threshold, "T",
               np.where(cate < -threshold, "C", "N"))

    rec_int = rec(cate_int)
    rec_tl = rec(cate_tl)
    agreement = np.mean(rec_int == rec_tl)
    return float(agreement)


def run_tlearner_benchmark(model, X, y, treatment, weights, params,
                           feature_names, treatment_col_idx,
                           cf_results, threshold=0.05, random_state=42):
    """
    Full T-learner benchmark pipeline.
    Compares interaction model CATE vs T-learner CATE.
    """
    # Identify main effect feature indices (exclude treatment and interactions)
    main_indices = [
        i for i, name in enumerate(feature_names)
        if not name.startswith("treat_x_") and name != "treatment"
    ]

    # Train T-learner on main effects only
    model_t, model_c = train_tlearner(
        X, y, treatment, weights, params, random_state,
        main_feature_indices=main_indices,
    )

    # T-learner CATE
    prob_t_tl, prob_c_tl, cate_tl = tlearner_cate(
        model_t, model_c, X, main_indices
    )

    # Interaction model CATE (from counterfactual results)
    cate_int = cf_results["cate"]

    # Pearson correlation
    r, p_val = pearsonr(cate_int, cate_tl)

    # Bland-Altman
    ba = bland_altman(cate_int, cate_tl)

    # Recommendation agreement
    agree = recommendation_agreement(cate_int, cate_tl, threshold)

    print("=== T-Learner HTE Benchmark ===")
    print(f"\nInteraction model CATE: mean={cate_int.mean():.4f}, std={cate_int.std():.4f}")
    print(f"T-Learner CATE:         mean={cate_tl.mean():.4f}, std={cate_tl.std():.4f}")
    print(f"\nPearson correlation: r = {r:.4f} (p = {p_val:.2e})")
    print(f"\nBland-Altman:")
    print(f"  Mean difference: {ba['mean_diff']:.4f}")
    print(f"  95% LoA: [{ba['loa_low']:.4f}, {ba['loa_high']:.4f}]")
    print(f"\nRecommendation agreement: {agree*100:.1f}%")

    return {
        "model_treated": model_t,
        "model_control": model_c,
        "cate_tlearner": cate_tl,
        "cate_interaction": cate_int,
        "pearson_r": float(r),
        "pearson_p": float(p_val),
        "bland_altman": ba,
        "recommendation_agreement": agree,
        "threshold": threshold,
    }


if __name__ == "__main__":
    from data_prep import prepare_data, get_feature_matrix
    from iptw import run_iptw
    from modeling import run_modeling
    from counterfactual import run_counterfactual

    dev, ext, imp, info = prepare_data()
    weights, ps_model, ps, balance = run_iptw(dev, info)
    X, y, t, cols = get_feature_matrix(dev, info)
    model_results = run_modeling(X, y, weights, cols)

    treatment_idx = cols.index("treatment")
    cf_results = run_counterfactual(
        model_results["model"], X, y, t, cols, treatment_idx
    )

    tl_results = run_tlearner_benchmark(
        model_results["model"], X, y, t, weights, model_results["params"],
        cols, treatment_idx, cf_results
    )
