from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path("mimic_data/processed")
K = 20
KNN_NEIGHBORS = 50
BATCH_SIZE = 500
TEST_SIZE = 0.20
RANDOM_STATE = 42


# split by patient, so that same patient doesn't show up in train and test
# this would cause leakage, so that all history stays on one side
def split_by_patient(features):
    rng = np.random.default_rng(RANDOM_STATE)
    subject_ids = features["subject_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(subject_ids)

    test_count = int(len(subject_ids) * TEST_SIZE)
    test_subjects = set(subject_ids[:test_count])
    train_subjects = set(subject_ids[test_count:])

    train_ids = features[features["subject_id"].isin(train_subjects)]["hadm_id"]
    test_ids = features[features["subject_id"].isin(test_subjects)]["hadm_id"]
    return set(train_ids), set(test_ids)


# precision@k, recall@k, and ndcg@k
# we are predicting hadm_id -> ranked list of medications
# our test is hadm_id -> medications actually administered
def evaluate(predictions, ground_truth):
    precisions = []
    recalls = []
    ndcgs = []

    for hadm_id, true_drugs in ground_truth.items():
        preds = predictions.get(hadm_id, [])
        top_k = preds[:K]
        hits = set(top_k) & true_drugs

        # out of the K recommended drugs, how many were correct
        precisions.append(len(hits) / K)

        # out of the drugs actually given, how many did we include
        recalls.append(len(hits) / len(true_drugs))

        # rewards putting correct drugs higher
        ndcgs.append(ndcg_at_k(top_k, true_drugs))

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# ndcg@k for an admission
def ndcg_at_k(preds, true_drugs):
    dcg = 0.0
    rank = 1
    for drug in preds:
        if drug in true_drugs:
            dcg += 1 / np.log2(rank + 1)
        rank += 1

    # ideal is if our order is all right
    ideal_hits = min(len(true_drugs), K)
    ideal_dcg = 0.0
    for r in range(1, ideal_hits + 1):
        ideal_dcg += 1 / np.log2(r + 1)

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# for the feature table
def build_dense_feature_matrix(features):
    drop_cols = ["subject_id", "hadm_id", "admittime", "edregtime", "edouttime"]
    dense = features.drop(columns=drop_cols, errors="ignore").copy()
    dense = pd.get_dummies(dense, dummy_na=True)
    dense = dense.fillna(0)
    arr = dense.to_numpy(dtype=np.float32)
    return csr_matrix(arr)


# add in the dense matrices
def build_full_feature_matrix(features):
    current_dx = load_npz(PROCESSED_DIR / "current_dx_matrix.npz")
    prior_dx = load_npz(PROCESSED_DIR / "prior_dx_matrix.npz")
    prior_med = load_npz(PROCESSED_DIR / "prior_med_matrix.npz")
    dense_features = build_dense_feature_matrix(features)

    return hstack([current_dx, prior_dx, prior_med, dense_features], format="csr")


features = pd.read_csv(
    PROCESSED_DIR / "patient_admission_snapshot.csv", low_memory=False
)

labels = pd.read_csv(PROCESSED_DIR / "admission_drug_labels.csv", low_memory=False)

# handle type
features["admittime"] = pd.to_datetime(features["admittime"])
features["hadm_id"] = features["hadm_id"].astype(int)
labels["hadm_id"] = labels["hadm_id"].astype(int)

positive_labels = labels[labels["label"] == 1]
interactions = positive_labels.rename(columns={"candidate_drug": "medication"})

train_ids, test_ids = split_by_patient(features)

train_interactions = interactions[interactions["hadm_id"].isin(train_ids)]
test_interactions = interactions[interactions["hadm_id"].isin(test_ids)]

ground_truth = {}
for hadm_id, grp in test_interactions.groupby("hadm_id"):
    ground_truth[hadm_id] = set(grp["medication"])

med_counts = train_interactions["medication"].value_counts()
top_meds = med_counts.index.tolist()

print("loaded %d admissions" % len(features))
print(f"train admissions: {len(train_ids):,}")
print(f"test admissions: {len(test_ids):,}")

# model 1: overall medication popularity
# same list to every admission
popularity_preds = {}
for hadm_id in ground_truth:
    popularity_preds[hadm_id] = top_meds
p, r, n = evaluate(popularity_preds, ground_truth)
print(f"\npopularity -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


hadm_ids = features["hadm_id"].to_numpy()

train_rows_list = []
test_rows_list = []
for i in range(len(hadm_ids)):
    h = hadm_ids[i]
    if h in train_ids:
        train_rows_list.append(i)
    elif h in test_ids:
        test_rows_list.append(i)
train_rows = np.array(train_rows_list)
test_rows = np.array(test_rows_list)

train_hadm = hadm_ids[train_rows]
test_hadm = hadm_ids[test_rows]

# Model 2: KNN using full admission feature set
print("Building full-feature KNN baseline...")
feature_matrix = build_full_feature_matrix(features)
train_features = feature_matrix[train_rows]
test_features = feature_matrix[test_rows]

train_drugs_by_hadm = {}
for hadm_id, grp in train_interactions.groupby("hadm_id"):
    train_drugs_by_hadm[hadm_id] = list(grp["medication"])

knn_preds = {}
for start in range(0, len(test_rows), BATCH_SIZE):
    batch = test_features[start : start + BATCH_SIZE]
    similarities = cosine_similarity(batch, train_features)

    for i, row_sims in enumerate(similarities):
        hadm_id = test_hadm[start + i]

        order_desc = np.argsort(row_sims)[::-1]
        neighbor_rows = order_desc[:KNN_NEIGHBORS]

        drug_scores = {}
        for neighbor_row in neighbor_rows:
            score = row_sims[neighbor_row]
            if score <= 0:
                continue

            neighbor_hadm_id = train_hadm[neighbor_row]
            if neighbor_hadm_id in train_drugs_by_hadm:
                neighbor_drugs = train_drugs_by_hadm[neighbor_hadm_id]
            else:
                neighbor_drugs = []
            for drug in neighbor_drugs:
                if drug in drug_scores:
                    drug_scores[drug] = drug_scores[drug] + score
                else:
                    drug_scores[drug] = score

        knn_preds[hadm_id] = sorted(drug_scores, key=drug_scores.get, reverse=True)

p, r, n = evaluate(knn_preds, ground_truth)
print(
    f"full-feature KNN -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}"
)
