"""
Full pipeline runner: executes all steps with production parameters
and generates all figures for the report.

Parameters:
  - Repeated CV: 5 folds x 100 repeats = 500 folds
  - Bootstrap: 1000 iterations
  - All figures saved to /mnt/results/figures/
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Font and style settings
rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rcParams['svg.fonttype'] = 'none'
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Ensure pipeline is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_prep import prepare_data, get_feature_matrix
from iptw import run_iptw
from modeling import run_modeling
from validation import run_validation, repeated_cv, bootstrap_validation, external_validation
from counterfactual import run_counterfactual
from tlearner import run_tlearner_benchmark
from export_model import export_model_json
from utils import compute_metrics

FIGURE_DIR = "/mnt/results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save figure as both PNG and SVG."""
    png_path = os.path.join(FIGURE_DIR, f"{name}.png")
    svg_path = os.path.join(FIGURE_DIR, f"{name}.svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"  Saved: {name}.png, {name}.svg")
    return png_path


def main():
    print("=" * 70)
    print("FULL PIPELINE RUN — Production Parameters")
    print("=" * 70)

    # ---- Step 1: Data Prep ----
    print("\n[1/8] Data Preparation")
    dev, ext, imp, info = prepare_data()
    X, y, t, cols = get_feature_matrix(dev, info)
    X_ext, y_ext, t_ext, _ = get_feature_matrix(ext, info)

    # ---- Step 2: IPTW ----
    print("\n[2/8] IPTW")
    weights, ps_model, ps, balance = run_iptw(dev, info)

    # ---- Step 3: Modeling ----
    print("\n[3/8] Model Training & Selection")
    model_results = run_modeling(X, y, weights, cols)
    model = model_results["model"]
    params = model_results["params"]

    # ---- Step 4-5: Validation (FULL parameters) ----
    print("\n[4/8] Internal & External Validation (500 folds, 1000 bootstrap)")
    val_results = run_validation(
        model, params, X, y, weights, X_ext, y_ext,
        n_splits=5, n_repeats=100, n_boot=1000, random_state=42,
    )

    # ---- Step 6: Counterfactual ----
    print("\n[5/8] Counterfactual Treatment Estimation")
    treatment_idx = cols.index("treatment")
    cf_results = run_counterfactual(
        model, X, y, t, cols, treatment_idx, threshold=0.05
    )

    # ---- Step 7: T-learner ----
    print("\n[6/8] T-learner HTE Benchmark")
    tl_results = run_tlearner_benchmark(
        model, X, y, t, weights, params, cols, treatment_idx,
        cf_results, threshold=0.05, random_state=42,
    )

    # ---- Step 8: Export model ----
    print("\n[7/8] Model Export")
    validation_summary = {
        "cv_summary": val_results["cv_summary"],
        "external": val_results["external"],
    }
    model_json = export_model_json(
        model, cols, info, cf_results, params,
        validation_summary, "/workspace/stroke-tool/docs/model.json",
    )

    # ---- Figures ----
    print("\n[8/8] Generating Figures")
    generate_all_figures(
        model, X, y, X_ext, y_ext, val_results,
        cf_results, tl_results, cols, params, weights, model_results,
        balance,
    )

    # ---- Save summary results ----
    save_results_summary(
        model_results, val_results, cf_results, tl_results, balance, info
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Model JSON: /workspace/stroke-tool/docs/model.json")
    print("=" * 70)


def generate_all_figures(model, X, y, X_ext, y_ext, val_results,
                         cf_results, tl_results, cols, params, weights,
                         model_results, balance):
    """Generate all publication-quality figures."""

    # --- Figure 1: ROC curves (internal + external) ---
    print("  [Fig 1] ROC curves")
    y_prob_dev = model.predict_proba(X)[:, 1]
    y_prob_ext = model.predict_proba(X_ext)[:, 1]

    fpr_dev, tpr_dev, _ = roc_curve(y, y_prob_dev)
    auc_dev = auc(fpr_dev, tpr_dev)
    fpr_ext, tpr_ext, _ = roc_curve(y_ext, y_prob_ext)
    auc_ext = auc(fpr_ext, tpr_ext)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_dev, tpr_dev, color="#0279EE", lw=2,
            label=f"Internal (UK, n={len(y)}): AUC = {auc_dev:.3f}")
    ax.plot(fpr_ext, tpr_ext, color="#FF9400", lw=2,
            label=f"External (Italy, n={len(y_ext)}): AUC = {auc_ext:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    ax.set_xlabel("1 - Specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    save_fig(fig, "fig1_roc_curves")

    # --- Figure 2: Calibration plot (external) ---
    print("  [Fig 2] Calibration plot")
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_ext, y_prob_ext, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#75A025", lw=2, ms=6,
            label=f"External (Italy): Brier = {val_results['external']['point_metrics']['Brier']:.3f}")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed proportion", fontsize=12)
    ax.set_title("Calibration Plot (External Cohort)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    save_fig(fig, "fig2_calibration")

    # --- Figure 3: CATE distribution (mirrors paper Fig 2) ---
    print("  [Fig 3] CATE distribution")
    cate = cf_results["cate"]
    prob_t = cf_results["prob_treated"]
    prob_c = cf_results["prob_control"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Counterfactual probability density
    ax = axes[0]
    ax.hist(prob_c, bins=40, alpha=0.5, color="#0279EE", label="P(favorable | no aspirin)", density=True)
    ax.hist(prob_t, bins=40, alpha=0.5, color="#FF9400", label="P(favorable | aspirin)", density=True)
    ax.set_xlabel("Predicted probability of favorable outcome", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Counterfactual Outcome Distributions", fontsize=13)
    ax.legend(fontsize=9)

    # Right: CATE distribution
    ax = axes[1]
    ax.hist(cate, bins=40, color="#75A025", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="#000000", lw=1.5, ls="-")
    ax.axvline(x=0.05, color="#FF9400", lw=1.5, ls="--", label="+5% threshold")
    ax.axvline(x=-0.05, color="#FF9400", lw=1.5, ls="--", label="-5% threshold")
    ax.set_xlabel("CATE (differential benefit)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CATE Distribution", fontsize=13)
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "fig3_cate_distribution")

    # --- Figure 4: Bland-Altman plot (mirrors paper Fig 3) ---
    print("  [Fig 4] Bland-Altman plot")
    cate_int = tl_results["cate_interaction"]
    cate_tl = tl_results["cate_tlearner"]
    ba = tl_results["bland_altman"]

    avg = (cate_int + cate_tl) / 2
    diff = cate_int - cate_tl

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(avg, diff, alpha=0.3, s=12, color="#0279EE")
    ax.axhline(y=ba["mean_diff"], color="#FF9400", lw=2, label=f"Mean diff = {ba['mean_diff']:.4f}")
    ax.axhline(y=ba["loa_low"], color="#FF9400", lw=1, ls="--",
               label=f"95% LoA: [{ba['loa_low']:.4f}, {ba['loa_high']:.4f}]")
    ax.axhline(y=ba["loa_high"], color="#FF9400", lw=1, ls="--")
    ax.axhline(y=0, color="#999999", lw=0.8, ls="-")
    ax.set_xlabel("Mean CATE (interaction + T-learner) / 2", fontsize=11)
    ax.set_ylabel("CATE difference (interaction - T-learner)", fontsize=11)
    ax.set_title("Bland-Altman: Interaction Model vs T-learner", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    save_fig(fig, "fig4_bland_altman")

    # --- Figure 5: CATE scatter (interaction vs T-learner) ---
    print("  [Fig 5] CATE scatter")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cate_tl, cate_int, alpha=0.3, s=12, color="#0279EE")
    lims = [min(cate_int.min(), cate_tl.min()), max(cate_int.max(), cate_tl.max())]
    ax.plot(lims, lims, color="#FF9400", lw=2, ls="--", label="y = x")
    ax.set_xlabel("T-learner CATE", fontsize=11)
    ax.set_ylabel("Interaction model CATE", fontsize=11)
    ax.set_title(f"CATE Correlation (r = {tl_results['pearson_r']:.3f})", fontsize=13)
    ax.legend(fontsize=10)
    save_fig(fig, "fig5_cate_scatter")

    # --- Figure 6: Coefficient plot ---
    print("  [Fig 6] Coefficient plot")
    coefs = model_results["coefficients"]
    nonzero = {k: v for k, v in coefs.items() if abs(v) > 1e-10}
    sorted_coefs = sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c[0] for c in sorted_coefs]
    vals = [c[1] for c in sorted_coefs]
    colors = ["#0279EE" if v > 0 else "#FF9400" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Coefficient (log-odds)", fontsize=11)
    ax.set_title("Elastic Net Coefficients (non-zero)", fontsize=13)
    ax.axvline(x=0, color="#000000", lw=0.8)
    ax.invert_yaxis()
    save_fig(fig, "fig6_coefficients")

    # --- Figure 7: Treatment Benefit Score distribution ---
    print("  [Fig 7] Treatment Benefit Score distribution")
    scores = cf_results["benefit_scores"]
    recs = cf_results["recommendations"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_map = {"Treat (aspirin)": "#75A025", "No treatment (control)": "#FF9400",
                  "No clear benefit": "#0279EE"}
    for rec_name, color in colors_map.items():
        mask = recs == rec_name
        if mask.sum() > 0:
            ax.hist(scores[mask], bins=range(1, 22), alpha=0.6, color=color,
                    label=f"{rec_name} (n={mask.sum()})", edgecolor="white")
    ax.set_xlabel("Treatment Benefit Score (1-20)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Treatment Benefit Score Distribution by Recommendation", fontsize=13)
    ax.legend(fontsize=9)
    save_fig(fig, "fig7_benefit_scores")

    # --- Figure 8: IPTW balance plot ---
    print("  [Fig 8] IPTW balance plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    covariates = balance["covariate"].tolist()
    x = np.arange(len(covariates))
    ax.bar(x - 0.15, balance["smd_unweighted"], 0.3, color="#FF9400", label="Unweighted")
    ax.bar(x + 0.15, balance["smd_weighted"], 0.3, color="#0279EE", label="IPTW weighted")
    ax.axhline(y=0.1, color="#999999", lw=1, ls="--", label="SMD = 0.1 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(covariates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Standardized Mean Difference", fontsize=11)
    ax.set_title("Covariate Balance Before/After IPTW", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig8_iptw_balance")


def save_results_summary(model_results, val_results, cf_results, tl_results, balance, info):
    """Save a JSON summary of all results."""
    summary = {
        "data": {
            "dev_country": info["dev_country"],
            "ext_country": info["ext_country"],
            "n_features": len(info["feature_names"]),
            "features": info["feature_names"],
        },
        "model": {
            "type": "Elastic Net Logistic Regression",
            "params": model_results["params"],
            "n_nonzero_coefs": sum(1 for v in model_results["coefficients"].values() if abs(v) > 1e-10),
            "coefficients": {k: round(v, 4) for k, v in model_results["coefficients"].items()},
        },
        "validation": {
            "internal_cv": {k: {sk: round(sv, 4) if isinstance(sv, float) else sv
                                for sk, sv in v.items()}
                            for k, v in val_results["cv_summary"].items()},
            "bootstrap": {k: {sk: round(sv, 4) if isinstance(sv, float) else sv
                              for sk, sv in v.items()}
                          for k, v in val_results["bootstrap_summary"].items()},
            "external_point": {k: round(v, 4) for k, v in val_results["external"]["point_metrics"].items()},
            "external_bootstrap": {k: {sk: round(sv, 4) if isinstance(sv, float) else sv
                                       for sk, sv in v.items()}
                                   for k, v in val_results["external"]["bootstrap"].items()},
        },
        "counterfactual": {
            "cate_mean": round(float(cf_results["cate"].mean()), 6),
            "cate_std": round(float(cf_results["cate"].std()), 6),
            "cate_min": round(float(cf_results["cate"].min()), 6),
            "cate_max": round(float(cf_results["cate"].max()), 6),
            "benefit_score_mean": round(float(cf_results["benefit_scores"].mean()), 2),
            "threshold": cf_results["threshold"],
        },
        "tlearner": {
            "pearson_r": round(tl_results["pearson_r"], 4),
            "bland_altman_mean_diff": round(tl_results["bland_altman"]["mean_diff"], 6),
            "bland_altman_loa": [
                round(tl_results["bland_altman"]["loa_low"], 6),
                round(tl_results["bland_altman"]["loa_high"], 6),
            ],
            "recommendation_agreement": round(tl_results["recommendation_agreement"], 4),
        },
        "iptw": {
            "max_smd_unweighted": round(float(balance["smd_unweighted"].abs().max()), 4),
            "max_smd_weighted": round(float(balance["smd_weighted"].abs().max()), 4),
            "all_balanced": bool(balance["balanced"].all()),
        },
    }

    # Recommendation distribution
    from collections import Counter
    rec_counts = Counter(cf_results["recommendations"])
    summary["counterfactual"]["recommendations"] = {
        k: {"count": v, "pct": round(v / len(cf_results["recommendations"]) * 100, 1)}
        for k, v in rec_counts.most_common()
    }

    path = "/mnt/results/pipeline_results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults summary saved to: {path}")


if __name__ == "__main__":
    main()
