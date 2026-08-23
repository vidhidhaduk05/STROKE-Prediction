"""
Step 3: Model training and selection.

Trains the primary Elastic Net logistic regression with IPTW weights and
tuned hyperparameters (C, l1_ratio) via 5-fold CV. Compares against
Decision Tree, Support Vector Classifier, and XGBoost (mirroring the
paper's comparator table). Includes forward and backward feature selection.
"""
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)


def tune_elastic_net(X, y, weights, cv_folds=5, random_state=42):
    """
    Tune Elastic Net hyperparameters (C, l1_ratio) via stratified k-fold CV.
    Returns (best_model, best_params, cv_results).
    """
    best_auc = 0
    best_params = {}
    best_model = None
    cv_results = []

    C_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for C in C_grid:
        for l1 in l1_grid:
            model = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=C, l1_ratio=l1, max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            )
            aucs = cross_val_score(
                model, X, y, cv=skf, scoring="roc_auc",
                params={"sample_weight": weights},
            )
            mean_auc = aucs.mean()
            cv_results.append({"C": C, "l1_ratio": l1, "mean_auc": mean_auc, "std_auc": aucs.std()})
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {"C": C, "l1_ratio": l1}
                best_model = LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    C=C, l1_ratio=l1, max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                )
                best_model.fit(X, y, sample_weight=weights)

    print(f"=== Elastic Net Tuning ===")
    print(f"Best params: C={best_params['C']}, l1_ratio={best_params['l1_ratio']}")
    print(f"Best CV AUC: {best_auc:.4f}")
    return best_model, best_params, cv_results


def train_comparators(X, y, weights, cv_folds=5, random_state=42):
    """
    Train comparator models (Decision Tree, SVC, XGBoost) with CV.
    Returns dict of {name: {model, cv_auc}}.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=random_state
    )
    dt_aucs = cross_val_score(dt, X, y, cv=skf, scoring="roc_auc",
                               params={"sample_weight": weights})
    dt.fit(X, y, sample_weight=weights)
    results["DecisionTree"] = {"model": dt, "cv_auc": dt_aucs.mean(), "cv_std": dt_aucs.std()}

    # Support Vector Classifier
    svc = SVC(kernel="rbf", C=1.0, probability=True,
              class_weight="balanced", random_state=random_state)
    svc_aucs = cross_val_score(svc, X, y, cv=skf, scoring="roc_auc",
                                params={"sample_weight": weights})
    svc.fit(X, y, sample_weight=weights)
    results["SVC"] = {"model": svc, "cv_auc": svc_aucs.mean(), "cv_std": svc_aucs.std()}

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
        xgb_aucs = cross_val_score(xgb, X, y, cv=skf, scoring="roc_auc",
                                    params={"sample_weight": weights})
        xgb.fit(X, y, sample_weight=weights)
        results["XGBoost"] = {"model": xgb, "cv_auc": xgb_aucs.mean(), "cv_std": xgb_aucs.std()}
    except ImportError:
        print("XGBoost not available, skipping.")

    print(f"\n=== Comparator Models (5-fold CV AUC) ===")
    for name, res in results.items():
        print(f"  {name}: AUC = {res['cv_auc']:.4f} +/- {res['cv_std']:.4f}")

    return results


def forward_selection(X, y, weights, feature_names, cv_folds=5, random_state=42):
    """
    Forward feature selection: iteratively add the feature that most
    improves CV AUC. Returns (selected_features, selection_history).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    remaining = list(range(X.shape[1]))
    selected = []
    history = []
    best_auc = 0

    while remaining:
        best_idx = None
        best_feat_auc = best_auc
        for idx in remaining:
            trial = selected + [idx]
            model = LogisticRegression(
                penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5,
                max_iter=5000, class_weight="balanced", random_state=random_state,
            )
            aucs = cross_val_score(model, X[:, trial], y, cv=skf, scoring="roc_auc",
                                    params={"sample_weight": weights})
            if aucs.mean() > best_feat_auc:
                best_feat_auc = aucs.mean()
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_auc = best_feat_auc
        history.append({"step": len(selected), "feature": feature_names[best_idx], "cv_auc": best_auc})

    print(f"\n=== Forward Selection ===")
    print(f"Selected {len(selected)} features, CV AUC = {best_auc:.4f}")
    for h in history:
        print(f"  Step {h['step']}: +{h['feature']} -> AUC = {h['cv_auc']:.4f}")

    return [feature_names[i] for i in selected], selected, history


def get_coefficients(model, feature_names):
    """Extract coefficients as a dict {feature: coef}."""
    coefs = model.coef_[0]
    return dict(zip(feature_names, coefs))


def run_modeling(X, y, weights, feature_names, random_state=42):
    """
    Full modeling pipeline: tune Elastic Net, train comparators,
    run forward selection. Returns (best_model, best_params, comparators,
    forward_result, coefficients).
    """
    # Tune Elastic Net
    enet_model, enet_params, cv_results = tune_elastic_net(
        X, y, weights, cv_folds=5, random_state=random_state
    )

    # Train comparators
    comparators = train_comparators(X, y, weights, cv_folds=5, random_state=random_state)

    # Forward selection
    fwd_features, fwd_indices, fwd_history = forward_selection(
        X, y, weights, feature_names, cv_folds=5, random_state=random_state
    )

    # Extract coefficients
    coefficients = get_coefficients(enet_model, feature_names)

    print(f"\n=== Final Elastic Net Coefficients ===")
    for feat, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:30s}: {coef:+.4f}  (OR = {np.exp(coef):.3f})")

    return {
        "model": enet_model,
        "params": enet_params,
        "cv_results": cv_results,
        "comparators": comparators,
        "forward_features": fwd_features,
        "forward_indices": fwd_indices,
        "forward_history": fwd_history,
        "coefficients": coefficients,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    from data_prep import prepare_data, get_feature_matrix
    from iptw import run_iptw

    dev, ext, imp, info = prepare_data()
    weights, ps_model, ps, balance = run_iptw(dev, info)
    X, y, t, cols = get_feature_matrix(dev, info)
    results = run_modeling(X, y, weights, cols)
