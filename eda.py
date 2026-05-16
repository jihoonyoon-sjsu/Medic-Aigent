from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

from config import PROCESSED_DIR, LAB_SOURCES, load_raw_tables

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === DATA CHECKS SECTION ===

print("=" * 60)
print("DATA CHECKS")
print("=" * 60)

snapshot_path = PROCESSED_DIR / "patient_admission_snapshot.csv"
labels_path = PROCESSED_DIR / "admission_drug_labels.csv"
matrix_files = {
    "current_dx": PROCESSED_DIR / "current_dx_matrix.npz",
    "prior_dx": PROCESSED_DIR / "prior_dx_matrix.npz",
    "prior_med": PROCESSED_DIR / "prior_med_matrix.npz",
    "current_proc": PROCESSED_DIR / "current_proc_matrix.npz",
    "prior_proc": PROCESSED_DIR / "prior_proc_matrix.npz",
}

# run prepare.py before running

# load processed tables
snapshot = pd.read_csv(snapshot_path, low_memory=False)
n_snap = len(snapshot)
print("snapshot loaded: %d rows" % n_snap)

labels = None
if labels_path.exists():
    labels = pd.read_csv(labels_path, low_memory=False)
    print(f"labels loaded: {len(labels):,} rows")
else:
    print(f"missing {labels_path}")

# check encoded features
missing_matrices = [name for name, path in matrix_files.items() if not path.exists()]

if missing_matrices:
    print(f"missing matrices: {missing_matrices}")
else:
    # all matrix should have same rows
    row_counts_match = True
    for name, p in matrix_files.items():
        mat = load_npz(p)
        if mat.shape[0] != n_snap:
            row_counts_match = False
            print(f"  row count mismatch: {name} has {mat.shape[0]} rows, snapshot has {n_snap}")
    print(f"snapshot/matrix row counts match: {row_counts_match}")

# interactions table
if labels is not None:
    hadm_ids_match = labels["hadm_id"].isin(snapshot["hadm_id"]).all()
    duplicate_pairs = labels.duplicated(["hadm_id", "candidate_drug"]).sum()
    pair_label_counts = labels.groupby(["hadm_id", "candidate_drug"])["label"].nunique()
    conflict_count = (pair_label_counts > 1).sum()
    print(f"all label hadm_id values exist in snapshot: {hadm_ids_match}")
    print(f"duplicate (hadm_id, candidate_drug) rows: {duplicate_pairs:,}")
    print(f"admission-drug label conflicts: {conflict_count:,}")


# === RAW TABLES SECTION ===

print("=" * 60)
print("RAW TABLES")
print("=" * 60)

# these are raw tables, before processing
admissions, patients, diagnoses, emar = load_raw_tables()

# row counts
print(f"admissions: {len(admissions):,} rows")
print(f"patients: {len(patients):,} rows")
print(f"diagnoses: {len(diagnoses):,} rows")
print(f"emar: {len(emar):,} rows")

# unique ids
print(f"unique patients in admissions: {admissions['subject_id'].nunique():,}")
# how many admissions tied to emar medications
print(f"unique admissions in emar: {emar['hadm_id'].nunique():,}")

# admissions per patient (cold start)
admissions_per_patient = admissions.groupby("subject_id").size()
print("\nadmissions per patient:")
desc = admissions_per_patient.describe().round(2)
print(desc.to_string())
single_visit = (admissions_per_patient == 1).sum()
multi_visit = (admissions_per_patient > 1).sum()
print(f"single-visit patients: {single_visit:,}")
print(f"multi-visit patients: {multi_visit:,}")

# diagnoses per admission
dx_per_admission_raw = diagnoses.groupby("hadm_id").size()
print("\ndiagnoses per admission (raw):")
desc = dx_per_admission_raw.describe().round(2)
print(desc.to_string())

# medication per admission
# no medication per admission, because of dups (since raw tables)
emar_per_admission = emar.groupby("hadm_id").size()
print("\nemar rows per admission:")
desc = emar_per_admission.describe().round(2)
print(desc.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(admissions_per_patient, bins=50, edgecolor="white")
axes[0].set_title("admissions per patient")
axes[0].set_xlabel("admissions")
axes[0].set_ylabel("patients")
axes[0].set_yscale("log")

axes[1].hist(dx_per_admission_raw, bins=50, edgecolor="white")
axes[1].set_title("diagnoses per admission")
axes[1].set_xlabel("diagnoses")
axes[1].set_ylabel("admissions")

axes[2].hist(emar_per_admission, bins=50, edgecolor="white")
axes[2].set_title("emar rows per admission")
axes[2].set_xlabel("rows")
axes[2].set_ylabel("admissions")

plt.tight_layout()
plt.savefig(OUT_DIR / "eda_baseline.png", dpi=150)
plt.close()
print(f"\nsaved {OUT_DIR / 'eda_baseline.png'}")


# === Feature Table ===
# this section is after we merged, on the feature table
# we have things added like age, history features, etc.

print("=" * 60)
print("FEATURE TABLE")
print("=" * 60)

# size after filtering to admissions with medications
print(f"snapshot rows (admissions kept): {len(snapshot):,}")
print(f"unique patients in snapshot: {snapshot['subject_id'].nunique():,}")

# Age
print("\nage at admission:")
age_desc = snapshot["age_at_admission"].describe().round(1)
print(age_desc.to_string())

# patient history, cold start admissions
print("\nnum_prior_admissions:")
prior_desc = snapshot["num_prior_admissions"].describe().round(2)
print(prior_desc.to_string())
first_visit = (snapshot["num_prior_admissions"] == 0).sum()
pct = first_visit / len(snapshot)
print(f"first-visit (cold-start) admissions: {first_visit:,} ({pct:.1%})")

# days between this and the most recent last admission
gap = snapshot["days_since_last_admission"].dropna()
print("\ndays_since_last_admission (warm-start only):")
gap_desc = gap.describe().round(1)
print(gap_desc.to_string())

# charlson
cci = snapshot["charlson_comorbidity_index"]
print("\ncharlson_comorbidity_index:")
print(cci.describe().round(2).to_string())
print(f"admissions with charlson row: {cci.notna().sum():,} ({cci.notna().mean():.1%})")

# services (last service)
svc = snapshot["curr_service"]
print(f"\ncurr_service coverage: {svc.notna().sum():,} ({svc.notna().mean():.1%})")
print("top services:")
print(svc.value_counts().head(10).to_string())

# Three plots: age dist, prior-admission count (log y because heavy
# tail), and the readmission gap distribution.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(snapshot["age_at_admission"].dropna(), bins=40, edgecolor="white")
axes[0].set_title("age at admission")
axes[0].set_xlabel("age")
axes[0].set_ylabel("admissions")

axes[1].hist(snapshot["num_prior_admissions"], bins=40, edgecolor="white")
axes[1].set_title("num prior admissions")
axes[1].set_xlabel("prior admissions")
axes[1].set_ylabel("admissions")
axes[1].set_yscale("log")

axes[2].hist(
    snapshot["days_since_last_admission"].dropna(),
    bins=50,
    edgecolor="white",
)
axes[2].set_title("days since last admission")
axes[2].set_xlabel("days")
axes[2].set_ylabel("admissions")
axes[2].set_yscale("log")

plt.tight_layout()
plt.savefig(OUT_DIR / "eda_snapshot.png", dpi=150)
plt.close()
print(f"\nsaved {OUT_DIR / 'eda_snapshot.png'}")


# === Sparse Matrices ===
# this section is aggregated multi-hot features
# diagnoses, medication, labs, procedures
print("\n" + "=" * 60)
print("Sparse matrices")
print("=" * 60)

missing = [name for name, p in matrix_files.items() if not p.exists()]

if missing:
    print(f"missing matrices: {missing}. Run prepare.py first. Skipping section 3.")
else:
    for name, p in matrix_files.items():
        # print stats
        # is cur values, history values mostly populated or empty
        mat = load_npz(p)
        per_row = np.asarray(mat.sum(axis=1)).flatten()

        density = mat.nnz / (mat.shape[0] * mat.shape[1])

        print(f"\n{name}: shape={mat.shape}, nnz={mat.nnz:,}, density={density:.4%}")
        print("  per-admission counts: mean=%.1f, median=%.0f, max=%d" % (per_row.mean(), np.median(per_row), per_row.max()))
        zero_rows = (per_row == 0).sum()
        print(f"  rows with zero entries: {zero_rows:,}")


# === Interaction Table ===
print("\n" + "=" * 60)
print("Interaction Table")
print("=" * 60)

if labels is None:
    print(f"missing {labels_path}. Run prepare.py first. Skipping section 4.")
else:
    n_labels = len(labels)
    n_pos = (labels["label"] == 1).sum()
    n_neg = (labels["label"] == 0).sum()
    n_drugs = labels["candidate_drug"].nunique()
    print("label rows: %d" % n_labels)
    print(f"positives: {n_pos:,}")
    print(f"negatives: {n_neg:,}")
    print(f"unique candidate drugs: {n_drugs:,}")

    # drugs per admission
    positives = labels[labels["label"] == 1]
    pos_per_admission = positives.groupby("hadm_id").size()
    print("\npositives per admission:")
    pos_desc = pos_per_admission.describe().round(2)
    print(pos_desc.to_string())

    # most popular drugs
    drug_popularity = positives["candidate_drug"].value_counts()
    print("\ntop 20 drugs by admissions administered:")
    print(drug_popularity.head(20).to_string())

    # what share of administrations are from the popular drugs
    # higher means stronger popularity bias, which means it may be harder to beat.
    n_unique_drugs = len(drug_popularity)
    top_10pct = max(1, int(n_unique_drugs * 0.1))
    share = drug_popularity.head(top_10pct).sum() / drug_popularity.sum()
    print(f"\ntop 10% of drugs ({top_10pct:,}) cover {share:.1%} of administrations")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pos_per_admission, bins=50, edgecolor="white")
    axes[0].set_title("positives per admission")
    axes[0].set_xlabel("drugs administered")
    axes[0].set_ylabel("admissions")

    top20 = drug_popularity.head(20)
    axes[1].barh(top20.index[::-1], top20.values[::-1])
    axes[1].set_title("top 20 drugs")
    axes[1].set_xlabel("admissions")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_labels.png", dpi=150)
    plt.close()
    print(f"\nsaved {OUT_DIR / 'eda_labels.png'}")


# === Lab Matrices ===
print("\n" + "=" * 60)
print("Lab matrices")
print("=" * 60)

for source in LAB_SOURCES:
    cur_path = PROCESSED_DIR / f"current_{source}_labs.npz"
    pri_path = PROCESSED_DIR / f"prior_{source}_labs.npz"
    if not cur_path.exists() or not pri_path.exists():
        print(f"\n{source}: missing files, skipping")
        continue

    cur = np.load(cur_path)
    pri = np.load(pri_path)
    cur_vals = cur["values"]
    cur_flags = cur["flags"]
    pri_flags = pri["flags"]

    n_cols = cur_vals.shape[1]
    cur_coverage = (cur_flags.sum(axis=1) > 0).mean()
    pri_coverage = (pri_flags.sum(axis=1) > 0).mean()
    per_col_cur = cur_flags.mean(axis=0)

    print(f"\n{source}: shape={cur_vals.shape}, cols={n_cols}")
    print(f"  current: admissions with any measurement {cur_coverage:.1%}")
    print(f"  prior:   admissions with any measurement {pri_coverage:.1%}")
    print(f"  per-column measured rate (current): min={per_col_cur.min():.1%}, "
          f"mean={per_col_cur.mean():.1%}, max={per_col_cur.max():.1%}")
