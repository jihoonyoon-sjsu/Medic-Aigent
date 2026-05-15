# this file is to prepare the dataset
# take raw tables and turn them into a format for modeling
# we will end up with 2 tables
# one is (patient, admission) -> patient features, admission features, history
# the other one is (patient, admission, drug) -> label

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from config import DATA_DIR, PROCESSED_DIR, RANDOM_STATE, LAB_SOURCES, load_raw_tables

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
# includes prior diagnoses, medications, procedures, number of admissions, days since last admission
# basically collapsing user history into a fixed size feature vector
def compute_history(snapshot, diagnoses, interactions, procedures, dx_codes, medications, proc_codes):
    dx_to_col = {c: i for i, c in enumerate(dx_codes)}
    med_to_col = {m: i for i, m in enumerate(medications)}
    proc_to_col = {c: i for i, c in enumerate(proc_codes)}

    n_rows = len(snapshot)
    num_prior_admissions = np.zeros(n_rows, dtype=int)
    days_since_last_admission = np.full(n_rows, np.nan)

    # per admission diagnosis/medication/procedure sets
    dx_clean = diagnoses.dropna(subset=["hadm_id", "icd_code"]).copy()
    dx_clean["icd_code"] = dx_clean["icd_code"].fillna("").astype(str).str.strip()
    dx_clean = dx_clean[dx_clean["icd_code"] != ""]

    proc_clean = procedures.dropna(subset=["hadm_id", "icd_code"]).copy()
    proc_clean["icd_code"] = proc_clean["icd_code"].fillna("").astype(str).str.strip()
    proc_clean = proc_clean[proc_clean["icd_code"] != ""]

    current_dx_by_hadm = {}
    for hadm_id, grp in dx_clean.groupby("hadm_id"):
        current_dx_by_hadm[hadm_id] = set(grp["icd_code"])

    current_meds_by_hadm = {}
    for hadm_id, grp in interactions.groupby("hadm_id"):
        current_meds_by_hadm[hadm_id] = set(grp["medication"])

    current_proc_by_hadm = {}
    for hadm_id, grp in proc_clean.groupby("hadm_id"):
        current_proc_by_hadm[hadm_id] = set(grp["icd_code"])

    prior_dx_rows, prior_dx_cols = [], []
    prior_med_rows, prior_med_cols = [], []
    prior_proc_rows, prior_proc_cols = [], []

    # go through each patient's admissions in chronological order
    # and build their history
    for _, group in snapshot.groupby("subject_id", sort=False):
        seen_dx = set()
        seen_meds = set()
        seen_procs = set()
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
            for code in seen_procs:
                if code in proc_to_col:
                    prior_proc_rows.append(i)
                    prior_proc_cols.append(proc_to_col[code])

            if row.hadm_id in current_dx_by_hadm:
                seen_dx.update(current_dx_by_hadm[row.hadm_id])
            if row.hadm_id in current_meds_by_hadm:
                seen_meds.update(current_meds_by_hadm[row.hadm_id])
            if row.hadm_id in current_proc_by_hadm:
                seen_procs.update(current_proc_by_hadm[row.hadm_id])
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

    proc_data = np.ones(len(prior_proc_rows), dtype=np.int8)
    prior_proc_matrix = csr_matrix(
        (proc_data, (prior_proc_rows, prior_proc_cols)),
        shape=(n_rows, len(proc_codes)),
    )

    history = pd.DataFrame(
        {
            "num_prior_admissions": num_prior_admissions,
            "days_since_last_admission": days_since_last_admission,
        },
        index=snapshot.index,
    )
    return history, prior_dx_matrix, prior_med_matrix, prior_proc_matrix


# labs - just first value per admission for each test
def build_current_labs(snapshot, source):
    df = pd.read_csv(DATA_DIR / f"{source}.csv", low_memory=False)
    # drop labs not tied to admissions
    df = df.dropna(subset=["hadm_id"]).copy()
    df["hadm_id"] = df["hadm_id"].astype(int)
    df = df[df["hadm_id"].isin(set(snapshot["hadm_id"]))]
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values(["hadm_id", "charttime"])

    lab_cols = [c for c in df.columns if c not in ("hadm_id", "charttime")]

    first = df.groupby("hadm_id")[lab_cols].first()
    aligned = first.reindex(snapshot["hadm_id"])

    values = aligned.to_numpy(dtype=np.float32)
    flags = (~aligned.isna()).to_numpy(dtype=np.float32)
    return values, flags, lab_cols

# labs history
def compute_prior_labs(snapshot, current_values, current_flags):
    n_rows, n_cols = current_values.shape
    prior_values = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    prior_flags = np.zeros((n_rows, n_cols), dtype=np.float32)

    for _, group in snapshot.groupby("subject_id", sort=False):
        last_vals = np.full(n_cols, np.nan, dtype=np.float32)
        last_flags = np.zeros(n_cols, dtype=np.float32)
        for row in group.itertuples():
            i = row.Index
            prior_values[i] = last_vals
            prior_flags[i] = last_flags

            measured = current_flags[i] > 0
            last_vals[measured] = current_values[i][measured]
            last_flags[measured] = 1.0

    return prior_values, prior_flags


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

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

existing_metadata = {}
metadata_path = PROCESSED_DIR / "feature_metadata.json"
if metadata_path.exists():
    with open(metadata_path) as fp:
        existing_metadata = json.load(fp)

print("loading raw tables...")
admissions, patients, diagnoses, emar = load_raw_tables()
procedures = pd.read_csv(DATA_DIR / "procedures_icd.csv", low_memory=False)
print("admissions:", len(admissions), "patients:", len(patients))
print("diagnoses:", len(diagnoses), "emar:", len(emar), "procedures:", len(procedures))


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

print("building current proc matrix...")
# use prev dx build function, same process
current_proc_matrix, proc_codes = build_current_dx_matrix(snapshot, procedures)
print("proc codes:", len(proc_codes))

print("computing history features...")
history_features, prior_dx_matrix, prior_med_matrix, prior_proc_matrix = compute_history(
    snapshot, diagnoses, interactions, procedures, dx_codes, medications, proc_codes
)

patient_admission_snapshot = pd.concat([snapshot, history_features], axis=1)

# charlson, 1 to 1 rows with admissions
print("merging charlson...")
charlson = pd.read_csv(DATA_DIR / "charlson.csv", low_memory=False)
charlson = charlson.drop_duplicates("hadm_id")
patient_admission_snapshot = patient_admission_snapshot.merge(
    charlson, on="hadm_id", how="left"
)

# services: last current service per admission
print("merging services...")
services = pd.read_csv(DATA_DIR / "services.csv", low_memory=False)
services = services.sort_values("transfertime")
services = services.drop_duplicates("hadm_id", keep="last")
services = services[["hadm_id", "curr_service"]]
patient_admission_snapshot = patient_admission_snapshot.merge(
    services, on="hadm_id", how="left"
)

# labs
print("building lab matrices...")
lab_cols_by_source = {}
for source in LAB_SOURCES:
    cur_path = PROCESSED_DIR / f"current_{source}_labs.npz"
    pri_path = PROCESSED_DIR / f"prior_{source}_labs.npz"
    cached_cols = existing_metadata.get("lab_sources", {}).get(source)
    if cur_path.exists() and pri_path.exists() and cached_cols is not None:
        print(f"  {source} already exists. skipping.")
        lab_cols_by_source[source] = cached_cols
        continue
    print(f"  {source}...")
    current_v, current_f, lab_cols = build_current_labs(snapshot, source)
    prior_v, prior_f = compute_prior_labs(snapshot, current_v, current_f)
    np.savez_compressed(cur_path, values=current_v, flags=current_f)
    np.savez_compressed(pri_path, values=prior_v, flags=prior_f)
    lab_cols_by_source[source] = lab_cols

label_path = PROCESSED_DIR / "admission_drug_labels.csv"
if label_path.exists():
    print(f"{label_path.name} already exists. skipping.")
else:
    print("making label table...")
    admission_drug_labels = make_label_table(snapshot, interactions)
    print("label rows:", len(admission_drug_labels))
    admission_drug_labels.to_csv(label_path, index=False)


# save everything

path = PROCESSED_DIR / "patient_admission_snapshot.csv"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    patient_admission_snapshot.to_csv(path, index=False)

path = PROCESSED_DIR / "current_dx_matrix.npz"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    save_npz(path, current_dx_matrix)

path = PROCESSED_DIR / "prior_dx_matrix.npz"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    save_npz(path, prior_dx_matrix)

path = PROCESSED_DIR / "prior_med_matrix.npz"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    save_npz(path, prior_med_matrix)

path = PROCESSED_DIR / "current_proc_matrix.npz"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    save_npz(path, current_proc_matrix)

path = PROCESSED_DIR / "prior_proc_matrix.npz"
if path.exists():
    print(f"{path.name} already exists. skipping.")
else:
    save_npz(path, prior_proc_matrix)

charlson_cols = [c for c in charlson.columns if c != "hadm_id"]
metadata = {
    "dx_codes": dx_codes,
    "medications": medications,
    "proc_codes": proc_codes,
    "lab_sources": lab_cols_by_source,
    "charlson_cols": charlson_cols,
    "negatives_per_admission": NEGATIVES_PER_ADMISSION,
    "random_state": RANDOM_STATE,
}
if metadata_path.exists():
    print(f"{metadata_path.name} already exists. skipping.")
else:
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2)

print("done")

# final files:
# - patient_admission_snapshot.csv
# - admission_drug_labels.csv
# - current_dx_matrix.npz
# - prior_dx_matrix.npz
# - prior_med_matrix.npz
# - current_proc_matrix.npz
# - prior_proc_matrix.npz
# - current_{source}_labs.npz / prior_{source}_labs.npz for each lab source
# - feature_metadata.json
