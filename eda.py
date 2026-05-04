from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import json

PROCESSED_DIR = Path("mimic_data/processed")

features = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv", low_memory=False)
icd_matrix = load_npz(PROCESSED_DIR / "icd_matrix.npz")
icd_codes = json.load(open(PROCESSED_DIR / "icd_codes.json"))

print(
    f"loaded {len(features):,} admissions, {icd_matrix.shape[1]:,} icd columns, {len(interactions):,} interaction pairs"
)

# medications per admission (mean/median/max)
meds_per_admission = interactions.groupby("hadm_id")["medication"].count()
print("\nmedications per admission:")
print(meds_per_admission.describe().round(1).to_string())

# diagnoses per admission (multiple diagnoses per admission)
dx_per_admission = np.array(icd_matrix.sum(axis=1)).flatten()
print(
    f"\ndiagnoses per admission: mean={dx_per_admission.mean():.1f}, median={np.median(dx_per_admission):.1f}, max={dx_per_admission.max()}"
)

# age
print(
    f"\nage at admission: mean={features['age_at_admission'].mean():.1f}, median={features['age_at_admission'].median():.1f}, min={features['age_at_admission'].min()}, max={features['age_at_admission'].max()}"
)

# top 20 medications
print("\ntop 20 most common medications:")
print(interactions["medication"].value_counts().head(20).to_string())

# sparsity
n_admissions = interactions["hadm_id"].nunique()
n_medications = interactions["medication"].nunique()
n_pairs = len(interactions)
sparsity = 1 - (n_pairs / (n_admissions * n_medications))
print(
    f"\ninteraction matrix: {n_admissions:,} admissions x {n_medications:,} medications"
)
print(f"sparsity: {sparsity:.4%}")

# popularity bias (10% of drugs account for most of the interactions)
# this is the challenge. top drugs are given to almost everyone
# so it will be hard to clear that bar. Our model needs to do better.
med_counts = interactions["medication"].value_counts()
top_10pct = int(len(med_counts) * 0.1)
top_10pct_share = med_counts.head(top_10pct).sum() / len(interactions)
print(
    f"\ntop 10% of medications ({top_10pct:,} drugs) account for {top_10pct_share:.1%} of all interactions"
)


print("\nmedication frequency distribution:")
freq_buckets = pd.cut(
    med_counts,
    bins=[0, 10, 100, 1000, 10000, med_counts.max()],
    labels=["1-10", "11-100", "101-1k", "1k-10k", "10k+"],
)
print(freq_buckets.value_counts().sort_index().to_string())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# medications per admission
axes[0, 0].hist(meds_per_admission, bins=50, edgecolor="white")
axes[0, 0].set_title("medications per admission")
axes[0, 0].set_xlabel("number of medications")
axes[0, 0].set_ylabel("admissions")

# diagnoses per admission
axes[0, 1].hist(dx_per_admission, bins=50, edgecolor="white")
axes[0, 1].set_title("diagnoses per admission")
axes[0, 1].set_xlabel("number of diagnoses")
axes[0, 1].set_ylabel("admissions")

# patient age
axes[1, 0].hist(features["age_at_admission"].dropna(), bins=40, edgecolor="white")
axes[1, 0].set_title("age at admission")
axes[1, 0].set_xlabel("age")
axes[1, 0].set_ylabel("admissions")

# top 20 meds
top_meds = med_counts.head(20)
axes[1, 1].barh(top_meds.index[::-1], top_meds.values[::-1])
axes[1, 1].set_title("top 20 medications")
axes[1, 1].set_xlabel("admissions")

# long tail for medications
freq_labels = freq_buckets.value_counts().sort_index()
axes[0, 2].bar(freq_labels.index.astype(str), freq_labels.values, edgecolor="white")
axes[0, 2].set_title("how many admissions each drug appears in")
axes[0, 2].set_xlabel("admissions per drug")
axes[0, 2].set_ylabel("number of drugs")

# popularity bias for medications
cumulative = med_counts.sort_values(ascending=False).cumsum() / med_counts.sum()
x = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
axes[1, 2].plot(x, cumulative.values * 100)
axes[1, 2].set_title("popularity bias")
axes[1, 2].set_xlabel("top X% of drugs")
axes[1, 2].set_ylabel("% of total interactions covered")
axes[1, 2].axhline(80, color="red", linestyle="--", linewidth=0.8)
axes[1, 2].axhline(95, color="orange", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.savefig("outputs/eda_baseline.png", dpi=150)
plt.close()
print("plot saved to outputs/eda_baseline.png")
