"""
Step 6: Counterfactual treatment estimation.

The core of the counterfactual estimation approach. For each patient, predicts
the probability of favorable outcome under BOTH treatment scenarios
(aspirin=1 and aspirin=0) using the same model with interaction terms.
Computes CATE, Treatment Benefit Score (1-20), and treatment recommendations
with a +/-5% differential benefit threshold.
"""
import numpy as np
from utils import sigmoid, benefit_score


def build_counterfactual_matrices(X, feature_names, treatment_col_idx):
    """
    Given the feature matrix X, create two versions:
    - X_treated: treatment column set to 1, interaction terms recomputed
    - X_control: treatment column set to 0, interaction terms zeroed out

    Returns (X_treated, X_control).
    """
    X_treated = X.copy()
    X_control = X.copy()

    # Set treatment column
    X_treated[:, treatment_col_idx] = 1
    X_control[:, treatment_col_idx] = 0

    # Update interaction terms: treat_x_feat = treatment * feat
    interaction_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("treat_x_")
    ]
    main_feature_map = {}
    for i, name in enumerate(feature_names):
        if name.startswith("treat_x_"):
            base_name = name.replace("treat_x_", "")
            for j, fname in enumerate(feature_names):
                if fname == base_name:
                    main_feature_map[i] = j
                    break

    for int_idx, main_idx in main_feature_map.items():
        X_treated[:, int_idx] = X_treated[:, main_idx]  # 1 * feat
        X_control[:, int_idx] = 0  # 0 * feat

    return X_treated, X_control


def counterfactual_predict(model, X, feature_names, treatment_col_idx):
    """
    Predict P(favorable) under both treatment scenarios for each patient.
    Returns (prob_treated, prob_control, cate).
    """
    X_treated, X_control = build_counterfactual_matrices(
        X, feature_names, treatment_col_idx
    )

    prob_treated = model.predict_proba(X_treated)[:, 1]
    prob_control = model.predict_proba(X_control)[:, 1]
    cate = prob_treated - prob_control

    return prob_treated, prob_control, cate


def make_recommendation(cate, threshold=0.05):
    """
    Treatment recommendation based on differential benefit.
    - If CATE > +threshold: recommend treatment (aspirin)
    - If CATE < -threshold: recommend control (no aspirin)
    - Otherwise: no clear benefit (either treatment acceptable)
    """
    recommendations = []
    for c in cate:
        if c > threshold:
            recommendations.append("Treat (aspirin)")
        elif c < -threshold:
            recommendations.append("No treatment (control)")
        else:
            recommendations.append("No clear benefit")
    return np.array(recommendations)


def run_counterfactual(model, X, y, treatment, feature_names,
                       treatment_col_idx, threshold=0.05):
    """
    Full counterfactual estimation pipeline.
    Returns dict with probabilities, CATE, Treatment Benefit Scores, recommendations.
    """
    prob_t, prob_c, cate = counterfactual_predict(
        model, X, feature_names, treatment_col_idx
    )

    scores = benefit_score(cate)
    recs = make_recommendation(cate, threshold=threshold)

    # Summary statistics
    print("=== Counterfactual Treatment Estimation ===")
    print(f"Patients: {len(cate)}")
    print(f"\nP(favorable | aspirin):    mean={prob_t.mean():.4f}, std={prob_t.std():.4f}")
    print(f"P(favorable | no aspirin): mean={prob_c.mean():.4f}, std={prob_c.std():.4f}")
    print(f"\nCATE (differential benefit):")
    print(f"  mean={cate.mean():.4f}, std={cate.std():.4f}")
    print(f"  range=[{cate.min():.4f}, {cate.max():.4f}]")
    print(f"  median={np.median(cate):.4f}")

    # Recommendation distribution
    from collections import Counter
    rec_counts = Counter(recs)
    print(f"\nRecommendations (threshold = +/-{threshold*100:.0f}%):")
    for rec, cnt in rec_counts.most_common():
        print(f"  {rec}: {cnt} ({cnt/len(recs)*100:.1f}%)")

    # Treatment Benefit Score distribution
    print(f"\nTreatment Benefit Score (1-20):")
    print(f"  mean={scores.mean():.1f}, std={scores.std():.1f}")
    print(f"  range=[{scores.min():.1f}, {scores.max():.1f}]")

    # Heterogeneity check: is there meaningful variation in CATE?
    iqr = np.percentile(cate, 75) - np.percentile(cate, 25)
    print(f"\nTreatment effect heterogeneity:")
    print(f"  CATE IQR = {iqr:.4f}")
    print(f"  % with |CATE| > {threshold*100:.0f}%: {np.mean(np.abs(cate) > threshold)*100:.1f}%")

    return {
        "prob_treated": prob_t,
        "prob_control": prob_c,
        "cate": cate,
        "benefit_scores": scores,
        "recommendations": recs,
        "threshold": threshold,
    }


if __name__ == "__main__":
    from data_prep import prepare_data, get_feature_matrix
    from iptw import run_iptw
    from modeling import run_modeling

    dev, ext, imp, info = prepare_data()
    weights, ps_model, ps, balance = run_iptw(dev, info)
    X, y, t, cols = get_feature_matrix(dev, info)
    model_results = run_modeling(X, y, weights, cols)

    treatment_idx = cols.index("treatment")
    cf_results = run_counterfactual(
        model_results["model"], X, y, t, cols, treatment_idx
    )
