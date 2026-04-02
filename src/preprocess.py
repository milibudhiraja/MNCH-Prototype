"""
preprocess.py
-------------
Feature engineering and preprocessing for the Dodoma pregnancy risk dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Core features to use for modelling ────────────────────────────────────────
BINARY_COLS = [
    "vaginal_bleeding", "nausea_and_vomiting", "problems_in_current_pregnancy",
    "preterm_labour_1", "prior_caesarean_delivery", "prio_forceps_delivery",
    "prio_vacuum_delivery", "miscarriages_or_abortions",
    "bleeding_in_previous_pregnancies", "bleeding_in_previous_deliveries",
    "bleeding_in_previous_puerperium",
    "presence_of_hypertension_in_previous_pregnancies",
    "problems_labour_and_puerperium", "heart_problems", "hypertension",
    "diabetes_mellitus", "gestational_diabetes", "liver_diseases", "hiv",
    "sexually_transmitted_diseases", "tuberculosis", "history_of_surgery",
    "kidney_diseases", "sickle_cell_or_haematological_disorders", "asthma",
    "epilepsy", "history_of_trauma_or_accidents", "tetanus_immunization",
    "smoking", "use_of_drugs", "use_of_alcohol", "intimate_partner_violence",
    "twin_pregnancy", "any_genetic_disease", "previous_contraceptive_history",
    "recent_contraception_before_pregnancy",
]

NUMERIC_COLS = [
    "age", "no_pregnancy", "duration_of_pregnancy_weeks_",
    "number_of_prior_pregnancies_with_live_births",
    "number_of_prior_pregnancies_with_stillbirths",
    "number_of_prior_deliveries", "number_of_living_children_and_birth_weight",
    "height_cm", "weight_kg",
]


def parse_bp(bp_str):
    """Extract systolic pressure from a string like '120/80'. Returns NaN on failure."""
    try:
        if pd.isna(bp_str):
            return np.nan
        parts = str(bp_str).strip().split("/")
        return float(parts[0])
    except Exception:
        return np.nan


def binary_encode(series):
    """Convert yes/no/true/false columns to 1/0."""
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
    return series.str.strip().str.lower().map(mapping).fillna(0).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer a clean feature matrix from the raw dataset.

    Returns a DataFrame with:
      - Baseline demographics & medical history
      - Aggregated vital signs across visits 1–3 (most complete visits)
      - Derived features (BMI, BP flags, haemoglobin trend)
    """
    feat = pd.DataFrame(index=df.index)

    # ── Demographics ──────────────────────────────────────────────────────────
    feat["age"] = pd.to_numeric(df["age"], errors="coerce").clip(10, 60)
    feat["no_pregnancy"] = pd.to_numeric(df["no_pregnancy"], errors="coerce")
    feat["duration_weeks"] = pd.to_numeric(
        df["duration_of_pregnancy_weeks_"], errors="coerce"
    )
    feat["prior_deliveries"] = pd.to_numeric(
        df["number_of_prior_deliveries"], errors="coerce"
    )
    feat["prior_live_births"] = pd.to_numeric(
        df["number_of_prior_pregnancies_with_live_births"], errors="coerce"
    )
    feat["prior_stillbirths"] = pd.to_numeric(
        df["number_of_prior_pregnancies_with_stillbirths"], errors="coerce"
    )
    feat["living_children"] = pd.to_numeric(
        df["number_of_living_children_and_birth_weight"], errors="coerce"
    )
    feat["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    feat["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")

    # BMI proxy (weight / (height/100)^2)
    h = feat["height_cm"] / 100
    feat["bmi"] = feat["weight_kg"] / (h ** 2)
    feat["bmi"] = feat["bmi"].clip(10, 60)  # remove physiological impossibilities

    # ── Binary medical history ────────────────────────────────────────────────
    for col in BINARY_COLS:
        if col in df.columns:
            feat[col] = binary_encode(df[col].astype(str))

    # ── Visit 1 vitals ────────────────────────────────────────────────────────
    feat["bp_systolic_v1"] = df["blood_pressure_v1"].apply(parse_bp)
    feat["hemoglobin_v1"] = pd.to_numeric(
        df["hemoglobin_check_result_v1"], errors="coerce"
    )
    feat["weight_v1"] = pd.to_numeric(df["weight_kg_v1"], errors="coerce")
    feat["gestational_week_v1"] = pd.to_numeric(
        df["pregnant_week_number_v1"], errors="coerce"
    )
    feat["bp_flag_v1"] = df["bp_is_greater_than_140_90_v1"].apply(
        lambda x: 1 if str(x).strip() == "1" else 0
    )

    # HIV visit 1 result
    feat["hiv_positive_v1"] = df["hiv_test_result_v1"].apply(
        lambda x: 1 if "positive" in str(x).lower() else 0
    )

    # Malaria visit 1
    feat["malaria_positive_v1"] = df["malaria_rapid_test_result_v1"].apply(
        lambda x: 1 if "positive" in str(x).lower() or "p0sitive" in str(x).lower() else 0
    )

    # Urine albumin visit 1
    feat["proteinuria_v1"] = df["urine_albumin_check_result_v1"].apply(
        lambda x: 1 if "positive" in str(x).lower() or "trace" in str(x).lower() else 0
    )

    # ── Visit 2 vitals ────────────────────────────────────────────────────────
    feat["bp_systolic_v2"] = df["blood_pressure_v2"].apply(parse_bp)
    feat["hemoglobin_v2"] = pd.to_numeric(
        df["hemoglobin_check_result_v2"], errors="coerce"
    )
    feat["weight_v2"] = pd.to_numeric(df["weight_kg_v2"], errors="coerce")
    feat["gestational_week_v2"] = pd.to_numeric(
        df["pregnant_week_number_v2"], errors="coerce"
    )

    # ── Visit 3 vitals ────────────────────────────────────────────────────────
    feat["bp_systolic_v3"] = df["blood_pressure_v3"].apply(parse_bp)
    feat["hemoglobin_v3"] = pd.to_numeric(
        df["hemoglobin_check_result_v3"], errors="coerce"
    )
    feat["weight_v3"] = pd.to_numeric(df["weight_kg_v3"], errors="coerce")
    feat["gestational_week_v3"] = pd.to_numeric(
        df["pregnant_week_number_v3"], errors="coerce"
    )

    # ── Aggregated / derived features ─────────────────────────────────────────
    # Max systolic across visits
    bp_cols = ["bp_systolic_v1", "bp_systolic_v2", "bp_systolic_v3"]
    feat["bp_systolic_max"] = feat[bp_cols].max(axis=1)

    # Min haemoglobin across visits (anaemia indicator)
    hb_cols = ["hemoglobin_v1", "hemoglobin_v2", "hemoglobin_v3"]
    feat["hemoglobin_min"] = feat[hb_cols].min(axis=1)
    feat["anaemia_flag"] = (feat["hemoglobin_min"] < 11.0).astype(int)

    # Number of BP flags across visits
    bp_flag_cols = [c for c in df.columns if c.startswith("bp_is_greater_than_140_90")]
    bp_flags = df[bp_flag_cols].apply(
        lambda col: col.apply(lambda x: 1 if str(x).strip() not in ["0", "not applicable", "nan"] else 0)
    )
    feat["bp_high_count"] = bp_flags.sum(axis=1)

    # Total antenatal visits
    feat["total_anc_visits"] = pd.to_numeric(
        df["total_antenatal_visits"], errors="coerce"
    )

    # Adolescent mother flag
    feat["adolescent"] = (feat["age"] < 18).astype(int)

    # Grand multipara (5+ prior deliveries)
    feat["grand_multipara"] = (feat["prior_deliveries"] >= 5).astype(int)

    # Weight gain proxy (v3 - v1)
    feat["weight_gain_v1_v3"] = feat["weight_v3"] - feat["weight_v1"]

    return feat


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return binary target: 1=high risk, 0=low risk."""
    return (df["Risk"].str.strip().str.lower() == "high").astype(int)
