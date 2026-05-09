# this file is to prepare the dataset
# take raw tables and turn them into a format for modeling
# we will end up with 2 tables
# one is (patient, admission) -> patient features, admission features, history
# the other one is (patient, admission, drug) -> label

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

DATA_DIR = Path("mimic_data/baseline_tables")
OUT_DIR = Path("mimic_data/processed")

RANDOM_STATE = 42

# negative sampling
# model needs both positive and negative examples to learn what is good/bad
# we can change this to per positive.
NEGATIVES_PER_ADMISSION = 10


# build diagnoses matrix
# use multi-hot encoding matrix
# returns:
# a sparse matrix (num_admissions, num_icd_codes)
# and a list of ICD codes, mapping column index to code
def build_current_dx_matrix(snapshot, diagnoses):
    # map each hadm_id to row number
    snap_hadms = snapshot["hadm_id"].tolist()
    hadm_to_row = {h: i for i, h in enumerate(snap_hadms)}

    dx = diagnoses.dropna(subset=["hadm_id", "icd_code"]).copy()
    dx["icd_code"] = dx["icd_code"].fillna("").astype(str).str.strip()
    dx = dx[dx["icd_code"] != ""]
    dx = dx[dx["hadm_id"].isin(hadm_to_row)]
    dx = dx[["hadm_id", "icd_code"]].drop_duplicates()

    dx_codes = sorted(dx["icd_code"].unique())
    # map icd code to column number
    code_to_col = {code: i for i, code in enumerate(dx_codes)}

    # now build sparse matrix (multi-hot)
    rows = dx["hadm_id"].map(hadm_to_row).values
    cols = dx["icd_code"].map(code_to_col).values
    data = np.ones(len(dx), dtype=np.int8)
    matrix = csr_matrix((data, (rows, cols)), shape=(snapshot.shape[0], len(dx_codes)))
    return matrix, dx_codes


# HISTORY (patient's past admissions)
# includes prior diagnoses, medications, number of admissions, days since last admission
# basically collapsing user history into a fixed size feature vector
def compute_history(snapshot, diagnoses, interactions, dx_codes, medications):
    dx_to_col = {c: i for i, c in enumerate(dx_codes)}
    med_to_col = {m: i for i, m in enumerate(medications)}

    n_rows = len(snapshot)
    num_prior_admissions = np.zeros(n_rows, dtype=int)
    days_since_last_admission = np.full(n_rows, np.nan)

    # per admission diagnosis/medication sets
    dx_clean = diagnoses.dropna(subset=["hadm_id", "icd_code"]).copy()
    dx_clean["icd_code"] = dx_clean["icd_code"].fillna("").astype(str).str.strip()
    dx_clean = dx_clean[dx_clean["icd_code"] != ""]

    current_dx_by_hadm = {}
    for hadm_id, grp in dx_clean.groupby("hadm_id"):
        current_dx_by_hadm[hadm_id] = set(grp["icd_code"])

    current_meds_by_hadm = {}
    for hadm_id, grp in interactions.groupby("hadm_id"):
        current_meds_by_hadm[hadm_id] = set(grp["medication"])

    prior_dx_rows, prior_dx_cols = [], []
    prior_med_rows, prior_med_cols = [], []

    # go through each patient's admissions in chronological order
    # and build their history
    for _, group in snapshot.groupby("subject_id", sort=False):
        seen_dx = set()
        seen_meds = set()
        last_admit_time = None
        prior_count = 0

        for row in group.itertuples():
            i = row.Index
            num_prior_admissions[i] = prior_count

            # days_since_last_admission
            if last_admit_time is not None:
                gap = (row.admittime - last_admit_time).days
                days_since_last_admission[i] = gap

            for code in seen_dx:
                if code in dx_to_col:
                    prior_dx_rows.append(i)
                    prior_dx_cols.append(dx_to_col[code])
            for med in seen_meds:
                if med in med_to_col:
                    prior_med_rows.append(i)
                    prior_med_cols.append(med_to_col[med])

            if row.hadm_id in current_dx_by_hadm:
                seen_dx.update(current_dx_by_hadm[row.hadm_id])
            if row.hadm_id in current_meds_by_hadm:
                seen_meds.update(current_meds_by_hadm[row.hadm_id])
            last_admit_time = row.admittime
            prior_count += 1

    dx_data = np.ones(len(prior_dx_rows), dtype=np.int8)
    prior_dx_matrix = csr_matrix(
        (dx_data, (prior_dx_rows, prior_dx_cols)),
        shape=(n_rows, len(dx_codes)),
    )

    med_data = np.ones(len(prior_med_rows), dtype=np.int8)
    prior_med_matrix = csr_matrix(
        (med_data, (prior_med_rows, prior_med_cols)),
        shape=(n_rows, len(medications)),
    )

    history = pd.DataFrame(
        {
            "num_prior_admissions": num_prior_admissions,
            "days_since_last_admission": days_since_last_admission,
        },
        index=snapshot.index,
    )
    return history, prior_dx_matrix, prior_med_matrix


# build the label table for admission drug pairs (final interaction table)
# build negative samples for training
def make_label_table(snapshot, interactions):
    rng = np.random.default_rng(RANDOM_STATE)
    all_drugs = sorted(interactions["medication"].unique())

    positive_by_hadm = {}
    for hadm_id, grp in interactions.groupby("hadm_id"):
        positive_by_hadm[hadm_id] = set(grp["medication"])

    rows = []
    for hadm_id in snapshot["hadm_id"]:
        positive_drugs = positive_by_hadm.get(hadm_id, set())

        for drug in sorted(positive_drugs):
            rows.append({"hadm_id": hadm_id, "candidate_drug": drug, "label": 1})

        # sample negatives from drugs not given in this admission
        non_positive_drugs = [d for d in all_drugs if d not in positive_drugs]

        n_neg = min(NEGATIVES_PER_ADMISSION, len(non_positive_drugs))
        negative_drugs = rng.choice(non_positive_drugs, size=n_neg, replace=False)

        for drug in sorted(negative_drugs):
            rows.append({"hadm_id": hadm_id, "candidate_drug": drug, "label": 0})

    return pd.DataFrame(rows)


# === put everything together ===

print("loading raw tables...")
admissions = pd.read_csv(DATA_DIR / "admissions.csv", low_memory=False)
patients = pd.read_csv(DATA_DIR / "patients.csv", low_memory=False)
diagnoses = pd.read_csv(DATA_DIR / "diagnoses_icd.csv", low_memory=False)
emar = pd.read_csv(DATA_DIR / "emar.csv", low_memory=False)
print("admissions:", len(admissions), "patients:", len(patients))
print("diagnoses:", len(diagnoses), "emar:", len(emar))


# creates (hadm_id, medication) foundation
# no matching hadm_id - means not tied to an admission
interactions = emar.dropna(subset=["hadm_id"]).copy()
interactions["hadm_id"] = interactions["hadm_id"].astype(int)

meds = interactions["medication"].fillna("")
meds = meds.astype(str).str.strip()
interactions["medication"] = meds
interactions = interactions[interactions["medication"] != ""]

# for each admission, what drugs were administered (no dups)
interactions = interactions[["hadm_id", "medication"]].drop_duplicates()
interactions = interactions.reset_index(drop=True)
print("interactions:", len(interactions))


# start by joining patients/admissions, building basic blocks
# one row per admission
# patient is an easy merge into admissions
snapshot = admissions.merge(patients, on="subject_id", how="left")
snapshot["admittime"] = pd.to_datetime(snapshot["admittime"])

# get age, we need to calculate this because of how MIMIC shuffles data
snapshot["age_at_admission"] = (
    snapshot["anchor_age"] + snapshot["admittime"].dt.year - snapshot["anchor_year"]
)
snapshot = snapshot.drop(columns=["anchor_age", "anchor_year"])

# remove admissions that don't have any medication records (useless to us)
valid_hadms = set(interactions["hadm_id"])
snapshot = snapshot[snapshot["hadm_id"].isin(valid_hadms)]

# sort so we can create user history later.
snapshot = snapshot.sort_values(["subject_id", "admittime", "hadm_id"])
snapshot = snapshot.reset_index(drop=True)
print("snapshot:", snapshot.shape)


print("building current dx matrix...")
current_dx_matrix, dx_codes = build_current_dx_matrix(snapshot, diagnoses)
medications = sorted(interactions["medication"].unique())
print("dx codes:", len(dx_codes), "medications:", len(medications))

print("computing history features...")
history_features, prior_dx_matrix, prior_med_matrix = compute_history(
    snapshot, diagnoses, interactions, dx_codes, medications
)

patient_admission_snapshot = pd.concat([snapshot, history_features], axis=1)

print("making label table...")
admission_drug_labels = make_label_table(snapshot, interactions)
print("label rows:", len(admission_drug_labels))


# save everything

OUT_DIR.mkdir(parents=True, exist_ok=True)
patient_admission_snapshot.to_csv(
    OUT_DIR / "patient_admission_snapshot.csv", index=False
)
admission_drug_labels.to_csv(OUT_DIR / "admission_drug_labels.csv", index=False)
save_npz(OUT_DIR / "current_dx_matrix.npz", current_dx_matrix)
save_npz(OUT_DIR / "prior_dx_matrix.npz", prior_dx_matrix)
save_npz(OUT_DIR / "prior_med_matrix.npz", prior_med_matrix)

metadata = {
    "dx_codes": dx_codes,
    "medications": medications,
    "negatives_per_admission": NEGATIVES_PER_ADMISSION,
    "random_state": RANDOM_STATE,
}
with open(OUT_DIR / "feature_metadata.json", "w") as fp:
    json.dump(metadata, fp, indent=2)

print("done")

# final files:
# - patient_admission_snapshot.csv
# - admission_drug_labels.csv
# - current_dx_matrix.npz
# - prior_dx_matrix.npz
# - prior_med_matrix.npz
# - feature_metadata.json
