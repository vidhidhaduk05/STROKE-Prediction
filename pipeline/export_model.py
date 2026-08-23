"""
Step 8a: Export the trained Elastic Net model to JSON for client-side
web app deployment.

The logistic regression model is just: sigmoid(intercept + sum(coef_i * x_i)).
We export coefficients, intercept, feature names, binarization thresholds,
and Treatment Benefit Score scaling parameters so the static web app can compute
predictions entirely in JavaScript.
"""
import json
import numpy as np


def export_model_json(model, feature_names, feature_info, cf_results,
                      model_params, validation_summary, output_path):
    """
    Export the trained model and all metadata needed for the web app.
    """
    coefficients = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])

    # Treatment Benefit Score scaling: use 1st/99th percentile of CATE from training
    cate = np.asarray(cf_results["cate"])
    cate_p1 = float(np.percentile(cate, 1))
    cate_p99 = float(np.percentile(cate, 99))

    # Feature metadata for the web form
    feature_metadata = []
    for name in feature_info["feature_names"]:
        meta = {"name": name, "label": _feature_label(name)}
        feature_metadata.append(meta)

    model_json = {
        "model_type": "Elastic Net Logistic Regression",
        "feature_names": feature_names,
        "coefficients": coefficients,
        "intercept": intercept,
        "main_features": feature_info["feature_names"],
        "interaction_features": feature_info["interaction_features"],
        "thresholds": feature_info["thresholds"],
        "feature_metadata": feature_metadata,
        "model_params": model_params,
        "benefit_score": {
            "cate_p1": cate_p1,
            "cate_p99": cate_p99,
            "min_score": 1,
            "max_score": 20,
        },
        "recommendation_threshold": cf_results["threshold"],
        "validation": {
            "internal_auc": validation_summary.get("cv_summary", {}).get("AUC", {}).get("mean", None),
            "internal_auc_ci": [
                validation_summary.get("cv_summary", {}).get("AUC", {}).get("ci_low", None),
                validation_summary.get("cv_summary", {}).get("AUC", {}).get("ci_high", None),
            ],
            "external_auc": validation_summary.get("external", {}).get("point_metrics", {}).get("AUC", None),
        },
        "dataset": "International Stroke Trial (IST), n=19,435",
        "methodology": "Counterfactual treatment estimation (Elastic Net + IPTW)",
        "disclaimer": "Educational tool only. Not for clinical use.",
    }

    with open(output_path, "w") as f:
        json.dump(model_json, f, indent=2)

    print(f"Model exported to {output_path}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Non-zero coefficients: {sum(1 for c in coefficients if abs(c) > 1e-10)}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Treatment Benefit Score range: CATE [{cate_p1:.4f}, {cate_p99:.4f}] -> [1, 20]")

    return model_json


def _feature_label(name):
    """Human-readable label for each feature."""
    labels = {
        "age_gt80": "Age > 80 years",
        "male": "Male sex",
        "afib": "Atrial fibrillation",
        "sbp_gt180": "Systolic BP > 180 mmHg",
        "impaired_conscious": "Impaired consciousness (drowsy/unconscious)",
        "delay_gt6h": "Delay from stroke onset > 6 hours",
        "tacs": "Total anterior circulation syndrome",
        "deficit_ge3": ">= 3 neurological deficits (NIHSS proxy)",
        "prior_aspirin": "Prior aspirin use (within 3 days)",
        "infarct_visible": "Infarct visible on CT",
        "heparin_allocated": "Heparin allocated (co-treatment)",
    }
    return labels.get(name, name)


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

    export_model_json(
        model_results["model"], cols, info, cf_results,
        model_results["params"], {},
        "/workspace/stroke-tool/docs/model.json"
    )
