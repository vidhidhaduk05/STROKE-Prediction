# Stroke Treatment Benefit Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**Live site:** [https://vidhidhaduk05.github.io/STROKE-Prediction/](https://vidhidhaduk05.github.io/STROKE-Prediction/)

A client-side clinical decision-support tool that estimates individualized treatment benefit for acute ischemic stroke patients using counterfactual treatment estimation. The pipeline trains an Elastic Net logistic regression with inverse probability of treatment weighting (IPTW) and a T-learner architecture on the International Stroke Trial (IST, n = 19,435), then deploys the trained model as a static web application requiring no backend infrastructure.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for clinical decision-making.

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Dataset](#dataset)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

The tool addresses a core challenge in precision medicine: moving beyond average treatment effects to estimate which individual patients benefit from a given therapy. Using the IST — a landmark randomized trial of aspirin in acute stroke — the pipeline:

1. Engineers 11 binarized clinical features from raw trial variables
2. Estimates propensity scores and applies stabilized IPTW weights
3. Trains an Elastic Net logistic regression with treatment × covariate interactions
4. Validates internally (repeated 5-fold cross-validation, 500 folds, 1000-iteration bootstrap) and externally (Italian cohort)
5. Computes conditional average treatment effects (CATE) via counterfactual prediction
6. Benchmarks CATE estimates against a T-learner meta-learner
7. Exports the trained model to JSON for client-side inference in the browser

The deployed web application provides two pages: an interactive prediction tool where clinicians can input patient characteristics and receive individualized treatment benefit estimates, and a technical paper presenting the methodology and validation results.

---

## Methodology

### Counterfactual Framework

The model adopts a potential-outcomes framework. For each patient with features *X*, the conditional average treatment effect is estimated as:

> **CATE(X) = E[Y | X, T=1] − E[Y | X, T=0]**

The primary model estimates both conditional expectations within a single logistic regression that includes the treatment indicator and 11 treatment × feature interaction terms (23 columns total). CATE is computed by evaluating the model under each treatment arm and taking the difference in predicted probabilities.

### Pipeline Architecture

| Stage | Module | Description |
|-------|--------|-------------|
| Data preparation | `pipeline/data_prep.py` | IST loading, feature engineering, MICE imputation, interaction terms |
| Propensity weighting | `pipeline/iptw.py` | Propensity score estimation, stabilized IPTW weights, SMD balance check |
| Model training | `pipeline/modeling.py` | Elastic Net logistic regression, comparator models, forward feature selection |
| Validation | `pipeline/validation.py` | Repeated 5-fold internal CV (500 folds), 1000-iteration bootstrap, frozen external validation |
| Counterfactual estimation | `pipeline/counterfactual.py` | Dual-arm prediction, CATE computation, Treatment Benefit Score (1–20) |
| T-learner benchmark | `pipeline/tlearner.py` | Separate-arm models, Pearson correlation, Bland-Altman analysis |
| Model export | `pipeline/export_model.py` | Serialization to JSON for browser deployment |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Elastic Net Logistic Regression |
| Regularization (C) | 0.1 |
| L1 ratio | 0.9 |
| Solver | SAGA |
| Class weighting | Balanced |
| Max iterations | 5,000 |
| Non-zero coefficients | 14 / 23 |

### Treatment Comparison

| Element | Specification |
|---------|---------------|
| Treatment | Aspirin vs. no aspirin (`RXASP`, 50/50 randomized) |
| Outcome | Favorable functional outcome at 6 months (`OCCODE` 3–4) |
| Development cohort | UK centres (n = 6,252) |
| External validation cohort | Italian centres (n = 3,437) |

---

## Results

| Metric | Internal CV | 95% CI | External | 95% CI |
|--------|-------------|--------|----------|--------|
| AUC | 0.763 | 0.737 – 0.787 | 0.767 | 0.750 – 0.783 |
| Brier score | 0.203 | 0.194 – 0.213 | 0.197 | 0.192 – 0.203 |
| Sensitivity | 0.775 | 0.719 – 0.823 | 0.802 | 0.782 – 0.822 |
| Specificity | 0.623 | 0.589 – 0.660 | 0.605 | 0.583 – 0.626 |

**T-learner concordance:** Pearson r = 0.943, Bland-Altman limits of agreement ±0.037, recommendation agreement 77.7%.

**Treatment benefit distribution:** CATE range −0.054 to 0.045 (mean −0.011). Using a ±5% recommendation threshold, 96.9% of patients were classified as no clear benefit — consistent with the modest average effect of aspirin in acute stroke (OR ≈ 1.08).

---

## Project Structure

```
stroke-tool/
├── index.html                # Tool page — interactive prediction interface
├── paper.html                # Paper page — technical report with figures
├── style.css                 # Application styling
├── app.js                    # Client-side inference engine
├── model.json                # Exported model coefficients & metadata
├── figures/                  # Publication-quality figures (8 figures)
├── pipeline/
│   ├── data_prep.py          # Data loading & feature engineering
│   ├── iptw.py               # Propensity scoring & IPTW weights
│   ├── modeling.py           # Elastic Net training & model selection
│   ├── validation.py         # Internal & external validation
│   ├── counterfactual.py     # CATE estimation & benefit scoring
│   ├── tlearner.py           # T-learner HTE benchmark
│   ├── export_model.py       # Model export to JSON
│   ├── run_full_pipeline.py  # End-to-end pipeline runner
│   └── utils.py              # Shared metrics & helper functions
├── tests/
│   └── test_pipeline.py      # Unit tests (18 tests)
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Pages deployment workflow
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/vidhidhaduk05/STROKE-Prediction.git
cd STROKE-Prediction
pip install -r requirements.txt
```

**Dependencies:** Python 3.10+, scikit-learn, pandas, numpy, matplotlib, seaborn, statsmodels, xgboost.

---

## Usage

### Run the full pipeline

```bash
python pipeline/run_full_pipeline.py
```

This executes all stages sequentially: data preparation, IPTW, model training, validation, counterfactual estimation, T-learner benchmark, and model export. Outputs are saved to `model.json` and `figures/`.

### Run individual stages

```bash
python pipeline/data_prep.py        # Step 1: Load IST & engineer features
python pipeline/iptw.py             # Step 2: Propensity scores & IPTW weights
python pipeline/modeling.py         # Step 3: Train Elastic Net
python pipeline/validation.py       # Step 4–5: Internal & external validation
python pipeline/counterfactual.py   # Step 6: CATE & benefit scoring
python pipeline/tlearner.py         # Step 7: T-learner benchmark
python pipeline/export_model.py     # Step 8: Export model to JSON
```

### Run the web application locally

```bash
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000) in a browser. The tool page loads with sample patient data pre-filled; adjust the inputs and click **Calculate Treatment Benefit** to view individualized predictions.

---

## Testing

```bash
pytest tests/ -v
```

The test suite covers data preparation, feature engineering, IPTW computation, model training, counterfactual prediction, benefit score calculation, and model export. All 18 tests must pass before deployment.

---

## Dataset

**International Stroke Trial (IST)** — a multicentre, randomized, factorial trial of aspirin and heparin in 19,435 patients with acute ischemic stroke (1991–1996, 36 countries).

- **Source:** [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/124)
- **Citation:** International Stroke Trial Collaborative Group. *The International Stroke Trial (IST): a randomised trial of aspirin, subcutaneous heparin, both, or neither among 19,435 patients with acute ischaemic stroke.* Lancet. 1997;349(9065):1569–1581.

---

## Deployment

The application is deployed as a static site via GitHub Pages, served directly from the repository root:

1. Push to the `main` branch
2. Enable Pages in **Repository Settings → Pages → Source: Deploy from a branch → main / (root)**
3. The site is live at `https://vidhidhaduk05.github.io/STROKE-Prediction/`

No backend server is required. All inference runs client-side in the browser using the exported model JSON.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The IST dataset is publicly available under the Edinburgh DataShare terms and is not included in this repository.
