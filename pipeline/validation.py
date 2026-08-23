"""
Steps 4-5: Internal and external validation.

Internal: repeated stratified k-fold CV (5 folds x 100 reps = 500 folds)
with 1,000 bootstrap iterations for 95% CIs. Mirrors paper Table 2.

External: apply frozen model to the Italian cohort without retraining,
with bootstrap CIs and calibration assessment. Mirrors paper Fig 4.
"""
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve

from utils import compute_metrics, bootstrap_metrics, sigmoid

warnings.filterwarnings("ignore")


def repeated_cv(X, y, weights, params, n_splits=5, n_repeats=100, random_state=42):
    """
    Repeated stratified k-fold cross-validation.
    Returns per-fold metrics and aggregated summary.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    fold_metrics = []
    for train_idx, test_idx in rskf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = weights[train_idx]

        model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=params["C"], l1_ratio=params["l1_ratio"],
            max_iter=5000, class_weight="balanced",
            random_state=random_state,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, y_prob)
        fold_metrics.append(m)

    # Aggregate
    summary = {}
    for key in fold_metrics[0]:
        vals = np.array([fm[key] for fm in fold_metrics])
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }

    print(f"=== Internal Validation (Repeated {n_splits}-fold CV, {n_splits*n_repeats} folds) ===")
    for key, s in summary.items():
        print(f"  {key:12s}: {s['mean']:.4f} +/- {s['std']:.4f}  (95% CI: {s['ci_low']:.4f}-{s['ci_high']:.4f})")

    return summary, fold_metrics


def bootstrap_validation(model, X, y, n_iter=1000, random_state=42):
    """
    Bootstrap validation on the full development cohort.
    Returns metric summary with 95% CIs.
    """
    y_prob = model.predict_proba(X)[:, 1]
    summary = bootstrap_metrics(y, y_prob, n_iter=n_iter, rng=np.random.default_rng(random_state))

    print(f"\n=== Bootstrap Validation ({n_iter} iterations) ===")
    for key, s in summary.items():
        print(f"  {key:12s}: {s['mean']:.4f} +/- {s['std']:.4f}  (95% CI: {s['ci_low']:.4f}-{s['ci_high']:.4f})")

    return summary


def external_validation(model, X_ext, y_ext, n_iter=1000, random_state=42):
    """
    External validation on the held-out Italian cohort.
    Returns metrics, bootstrap CIs, and calibration data.
    """
    y_prob = model.predict_proba(X_ext)[:, 1]
    point_metrics = compute_metrics(y_ext, y_prob)
    boot_summary = bootstrap_metrics(y_ext, y_prob, n_iter=n_iter,
                                      rng=np.random.default_rng(random_state))

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_ext, y_prob, n_bins=10, strategy="quantile")

    print(f"\n=== External Validation (Italian cohort, n={len(y_ext)}) ===")
    print("Point estimates:")
    for key, val in point_metrics.items():
        print(f"  {key:12s}: {val:.4f}")
    print("\nBootstrap 95% CIs:")
    for key, s in boot_summary.items():
        print(f"  {key:12s}: {s['mean']:.4f}  (95% CI: {s['ci_low']:.4f}-{s['ci_high']:.4f})")

    return {
        "point_metrics": point_metrics,
        "bootstrap": boot_summary,
        "calibration": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()},
        "y_prob": y_prob,
    }


def run_validation(model, params, X, y, weights, X_ext, y_ext,
                   n_splits=5, n_repeats=100, n_boot=1000, random_state=42):
    """
    Full validation pipeline: repeated CV + bootstrap + external.
    Returns dict with all results.
    """
    # Internal: repeated CV
    cv_summary, fold_metrics = repeated_cv(
        X, y, weights, params, n_splits=n_splits, n_repeats=n_repeats,
        random_state=random_state,
    )

    # Internal: bootstrap on full model
    boot_summary = bootstrap_validation(model, X, y, n_iter=n_boot, random_state=random_state)

    # External validation
    ext_results = external_validation(model, X_ext, y_ext, n_iter=n_boot, random_state=random_state)

    return {
        "cv_summary": cv_summary,
        "fold_metrics": fold_metrics,
        "bootstrap_summary": boot_summary,
        "external": ext_results,
    }


if __name__ == "__main__":
    from data_prep import prepare_data, get_feature_matrix
    from iptw import run_iptw
    from modeling import run_modeling

    dev, ext, imp, info = prepare_data()
    weights, ps_model, ps, balance = run_iptw(dev, info)
    X, y, t, cols = get_feature_matrix(dev, info)
    model_results = run_modeling(X, y, weights, cols)

    X_ext, y_ext, t_ext, _ = get_feature_matrix(ext, info)

    # Use fewer repeats for quick test
    val_results = run_validation(
        model_results["model"], model_results["params"],
        X, y, weights, X_ext, y_ext,
        n_splits=5, n_repeats=10, n_boot=200,  # Quick test
    )
