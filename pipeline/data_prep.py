"""
Step 1: Data preparation and feature engineering for the IST-based
stroke treatment-benefit prediction pipeline.

Loads the International Stroke Trial (IST) dataset, maps variables to
a clinical feature schema, binarizes continuous variables at
clinical thresholds, creates treatment x covariate interaction terms,
and applies MICE imputation for remaining missingness.
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ---------------------------------------------------------------------------
# Configuration: variable mapping and binarization thresholds
# ---------------------------------------------------------------------------
IST_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv"
IST_ENCODING = "latin-1"

THRESHOLDS = {
    "age": 80,
    "delay": 6,
    "sbp": 180,
    "deficit_count": 3,
}

FEATURE_NAMES = [
    "age_gt80", "male", "afib", "sbp_gt180", "impaired_conscious",
    "delay_gt6h", "tacs", "deficit_ge3", "prior_aspirin",
    "infarct_visible", "heparin_allocated",
]

INTERACTION_FEATURES = [f"treat_x_{f}" for f in FEATURE_NAMES]


def load_ist(path=None):
    """Load the IST dataset from local path or remote URL."""
    if path is None:
        path = IST_URL
    return pd.read_csv(path, encoding=IST_ENCODING, low_memory=False)


def _safe_binary(series, positive_vals):
    """Convert a categorical series to binary (1 for positive_vals, 0 otherwise)."""
    return series.astype(str).str.strip().isin(positive_vals).astype(int)


def engineer_features(df):
    """Transform raw IST columns into the binarized feature schema."""
    out = pd.DataFrame(index=df.index)

    out["age_gt80"] = (df["AGE"] > THRESHOLDS["age"]).astype(int)
    out["male"] = _safe_binary(df["SEX"], ["M"])
    out["afib"] = _safe_binary(df["RATRIAL"], ["Y"])
    out["sbp_gt180"] = (pd.to_numeric(df["RSBP"], errors="coerce") > THRESHOLDS["sbp"]).astype(int)
    out["impaired_conscious"] = _safe_binary(df["RCONSC"], ["D", "U"])
    out["delay_gt6h"] = (pd.to_numeric(df["RDELAY"], errors="coerce") > THRESHOLDS["delay"]).astype(int)
    out["tacs"] = _safe_binary(df["STYPE"], ["TACS"])

    deficit_cols = [f"RDEF{i}" for i in range(1, 9)]
    deficit_count = sum(_safe_binary(df[c], ["Y"]) for c in deficit_cols)
    out["deficit_ge3"] = (deficit_count >= THRESHOLDS["deficit_count"]).astype(int)

    out["prior_aspirin"] = _safe_binary(df["RASP3"], ["Y"])
    out["infarct_visible"] = _safe_binary(df["RVISINF"], ["Y"])
    out["heparin_allocated"] = _safe_binary(df["RXHEP"], ["L", "M", "H"])

    out["treatment"] = _safe_binary(df["RXASP"], ["Y"])

    occode = df["OCCODE"].astype(str).str.strip()
    out["favorable"] = occode.isin(["3", "4"]).astype(int)
    out["outcome_known"] = occode.isin(["1", "2", "3", "4"]).astype(int)
    out["country"] = df["COUNTRY"].astype(str).str.strip()

    return out


def add_interaction_terms(df):
    """Create treatment x covariate interaction columns."""
    for feat in FEATURE_NAMES:
        df[f"treat_x_{feat}"] = df["treatment"] * df[feat]
    return df


def impute_missing(df, imputer=None, random_state=42):
    """
    MICE imputation. If imputer is None, fit a new one; otherwise transform.
    Returns (df, imputer).
    """
    cols = FEATURE_NAMES + ["treatment"]
    if imputer is None:
        imputer = IterativeImputer(
            max_iter=10, random_state=random_state,
            sample_posterior=False, min_value=0, max_value=1,
        )
        df[cols] = imputer.fit_transform(df[cols])
    else:
        df[cols] = imputer.transform(df[cols])

    for col in cols:
        if col != "treatment":
            df[col] = (df[col] >= 0.5).astype(int)
    df["treatment"] = df["treatment"].round().astype(int)
    return df, imputer


def prepare_data(path=None, dev_country="UK", ext_country="ITAL"):
    """
    Full data preparation pipeline.
    Returns (dev_df, ext_df, imputer, feature_info).
    """
    raw = load_ist(path)
    print(f"Loaded IST: {raw.shape[0]} patients, {raw.shape[1]} columns")

    feat = engineer_features(raw)
    print(f"After feature engineering: {feat.shape[0]} patients, {len(FEATURE_NAMES)} features")

    feat = feat[feat["outcome_known"] == 1].copy()
    print(f"After dropping unknown outcomes: {feat.shape[0]} patients")
    print(f"  Favorable: {feat['favorable'].sum()} ({feat['favorable'].mean()*100:.1f}%)")
    print(f"  Unfavorable: {(~feat['favorable'].astype(bool)).sum()} ({(1-feat['favorable'].mean())*100:.1f}%)")

    feat = add_interaction_terms(feat)

    dev_df = feat[feat["country"] == dev_country].copy()
    ext_df = feat[feat["country"] == ext_country].copy()
    print(f"\nDevelopment ({dev_country}): {dev_df.shape[0]} | Treated: {dev_df['treatment'].sum()} | Favorable: {dev_df['favorable'].sum()}")
    print(f"External ({ext_country}): {ext_df.shape[0]} | Treated: {ext_df['treatment'].sum()} | Favorable: {ext_df['favorable'].sum()}")

    # Check missingness
    missing_pct = dev_df[FEATURE_NAMES + ["treatment"]].isna().mean() * 100
    high_missing = missing_pct[missing_pct > 10]
    if len(high_missing) > 0:
        print(f"\nExcluding >10% missing vars: {list(high_missing.index)}")
        for col in high_missing.index:
            if col in FEATURE_NAMES:
                FEATURE_NAMES.remove(col)
    else:
        print(f"\nNo variables with >10% missingness.")

    # Impute: fit on dev, transform ext
    dev_df, imputer = impute_missing(dev_df)
    ext_df, _ = impute_missing(ext_df, imputer=imputer)

    # Rebuild interaction terms after imputation
    dev_df = add_interaction_terms(dev_df)
    ext_df = add_interaction_terms(ext_df)

    feature_info = {
        "feature_names": list(FEATURE_NAMES),
        "interaction_features": [f"treat_x_{f}" for f in FEATURE_NAMES],
        "thresholds": THRESHOLDS,
        "dev_country": dev_country,
        "ext_country": ext_country,
    }

    return dev_df, ext_df, imputer, feature_info


def get_feature_matrix(df, feature_info):
    """
    Build the full feature matrix: [main effects, treatment, interactions].
    Returns (X, y, treatment) as numpy arrays.
    """
    fnames = feature_info["feature_names"]
    inames = feature_info["interaction_features"]
    cols = fnames + ["treatment"] + inames
    X = df[cols].values.astype(float)
    y = df["favorable"].values.astype(int)
    t = df["treatment"].values.astype(int)
    return X, y, t, cols


if __name__ == "__main__":
    dev, ext, imp, info = prepare_data()
    print(f"\nFeatures: {info['feature_names']}")
    print(f"Dev: {dev.shape}, Ext: {ext.shape}")
    X, y, t, cols = get_feature_matrix(dev, info)
    print(f"Feature matrix: {X.shape}, columns: {cols}")
