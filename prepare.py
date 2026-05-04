from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import json

# STEP ONE: DATA PROCESSING

DATA_DIR = Path("mimic_data/baseline_tables")

# start off with 4 tables for baseline
admissions = pd.read_csv(DATA_DIR / "admissions.csv", low_memory=False)
patients = pd.read_csv(DATA_DIR / "patients.csv", low_memory=False)
diagnoses = pd.read_csv(DATA_DIR / "diagnoses_icd.csv", low_memory=False)
emar = pd.read_csv(DATA_DIR / "emar.csv", low_memory=False)

# start with admissions table
# merge patient data
# calculate age
features = admissions.merge(patients, on="subject_id", how="left")
features["admittime"] = pd.to_datetime(features["admittime"])
features["age_at_admission"] = (
    features["anchor_age"] + features["admittime"].dt.year - features["anchor_year"]
)
features = features.drop(columns=["anchor_age", "anchor_year"])

# interactions: one row per (admission, medication) pair
interactions = (
    emar.dropna(subset=["hadm_id"])
    .assign(medication=lambda df: df["medication"].fillna("").str.strip())
    .query("medication != ''")[["hadm_id", "medication"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# drop admissions with no medication records
features = features[features["hadm_id"].isin(interactions["hadm_id"])].reset_index(
    drop=True
)

# multi-hot encode diagnoses (scipy.sparse)
dx = diagnoses.dropna(subset=["hadm_id", "icd_code"]).copy()
dx["icd_code"] = dx["icd_code"].astype(str).str.strip()
dx = dx[dx["icd_code"] != ""]

# rows is which admission, cols is which ICD code (diagnosis)
hadm_index = {h: i for i, h in enumerate(features["hadm_id"])}
dx = dx[dx["hadm_id"].isin(hadm_index)]
icd_codes = dx["icd_code"].unique().tolist()
icd_index = {c: i for i, c in enumerate(icd_codes)}
rows = dx["hadm_id"].map(hadm_index).to_numpy()
cols = dx["icd_code"].map(icd_index).to_numpy()
icd_matrix = csr_matrix(
    (np.ones(len(dx), dtype=np.int8), (rows, cols)),
    shape=(len(features), len(icd_codes)),
)

print(f"admissions: {len(features):,}")
print(f"dense feature columns: {len(features.columns):,}")
print(f"icd columns (sparse): {icd_matrix.shape[1]:,}")
print(f"unique medications: {interactions['medication'].nunique():,}")
print(f"interaction pairs: {len(interactions):,}")

OUT_DIR = Path("mimic_data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT_DIR / "features.csv", index=False)
interactions.to_csv(OUT_DIR / "interactions.csv", index=False)
save_npz(OUT_DIR / "icd_matrix.npz", icd_matrix)
json.dump(icd_codes, open(OUT_DIR / "icd_codes.json", "w"))
print(f"saved to {OUT_DIR}")


# STEP TWO: DATA QUALITY

# check nulls in joins, orphans, date range, etc

print("\n\nDATA QUALITY:")

for name, df in [
    ("admissions", admissions),
    ("patients", patients),
    ("diagnoses", diagnoses),
    ("emar", emar),
]:
    for col in ["hadm_id", "subject_id"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            print(f"{name}.{col} nulls: {null_count:,} ({null_count / len(df):.1%})")

valid_hadm_ids = set(admissions["hadm_id"])
dx_orphaned = (~diagnoses["hadm_id"].isin(valid_hadm_ids)).sum()
emar_orphaned = (~emar["hadm_id"].dropna().isin(valid_hadm_ids)).sum()
print(f"diagnoses hadm_ids not in admissions: {dx_orphaned:,}")
print(f"emar hadm_ids not in admissions: {emar_orphaned:,}")

missing_meds = emar["medication"].isna().sum()
missing_dx = diagnoses["icd_code"].isna().sum()
print(f"emar missing medication name: {missing_meds:,}")
print(f"diagnoses missing icd_code: {missing_dx:,}")

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
print(
    f"admission date range: {admissions['admittime'].min().date()} to {admissions['admittime'].max().date()}"
)

admissions_with_meds = interactions["hadm_id"].nunique()
admissions_with_dx = diagnoses["hadm_id"].nunique()
print(
    f"admissions with at least one medication: {admissions_with_meds:,} ({admissions_with_meds / len(admissions):.1%})"
)
print(
    f"admissions with at least one diagnosis: {admissions_with_dx:,} ({admissions_with_dx / len(admissions):.1%})"
)
