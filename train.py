from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("mimic_data/processed")

features = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv", low_memory=False)
icd_matrix = load_npz(PROCESSED_DIR / "icd_matrix.npz")
icd_codes = json.load(open(PROCESSED_DIR / "icd_codes.json"))

print(
    f"loaded {len(features):,} admissions, {icd_matrix.shape[1]:,} icd columns, {len(interactions):,} interaction pairs"
)

# train/test split by id, random
all_hadm_ids = features["hadm_id"].values
train_ids, test_ids = train_test_split(all_hadm_ids, test_size=0.2, random_state=42)
train_ids, test_ids = set(train_ids), set(test_ids)

train_interactions = interactions[interactions["hadm_id"].isin(train_ids)]
test_interactions = interactions[interactions["hadm_id"].isin(test_ids)]

# the drugs actually received
ground_truth = test_interactions.groupby("hadm_id")["medication"].apply(set).to_dict()

K = 20

# global popularity ranking
top_meds = train_interactions["medication"].value_counts().index.tolist()


# returns precision@k and recall@k
def evaluate(predictions, ground_truth, k):
    precisions, recalls = [], []
    for hadm_id, true_meds in ground_truth.items():
        # check if our prediction is in the test set
        if hadm_id not in predictions:
            continue
        preds = predictions[hadm_id]
        if len(preds) < k:
            seen = set(preds)
            preds = list(preds) + [d for d in top_meds if d not in seen]
        top_k = preds[:k]
        hits = len(set(top_k) & true_meds)

        # for my own reference:
        # precision is "of our guesses, what fraction were right"
        precisions.append(hits / k)
        # recall is "of the true items, what fraction did we find"
        recalls.append(hits / len(true_meds))
    return np.mean(precisions), np.mean(recalls)


# MODEL 1: POPULARITY BASELINE
# recommend the same top-K most common drugs to every admission
popularity_preds = {hadm_id: top_meds for hadm_id in ground_truth}
p, r = evaluate(popularity_preds, ground_truth, K)
print(f"\npopularity baseline -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}")


# match admissions to dianosis in the sparse matrix
from scipy.sparse import csr_matrix

hadm_ids = features["hadm_id"].values
train_rows = np.array([i for i, h in enumerate(hadm_ids) if h in train_ids])
test_rows = np.array([i for i, h in enumerate(hadm_ids) if h in test_ids])
train_hadm = hadm_ids[train_rows]
test_hadm = hadm_ids[test_rows]
train_icd = icd_matrix[train_rows]
test_icd = icd_matrix[test_rows]

# MODEL 2: POPULARITY BY DIAGNOSIS
# recommend based on the most common drugs for each diagnosis
drugs = train_interactions["medication"].unique().tolist()
drug_to_idx = {d: i for i, d in enumerate(drugs)}
train_row_idx = {h: i for i, h in enumerate(train_hadm)}

rows = train_interactions["hadm_id"].map(train_row_idx).values
cols = train_interactions["medication"].map(drug_to_idx).values
train_drug = csr_matrix(
    (np.ones(len(rows), dtype=np.int32), (rows, cols)),
    shape=(len(train_rows), len(drugs)),
)

drug_icd = train_drug.T @ train_icd
test_scores = test_icd @ drug_icd.T

popularity_by_dx_preds = {}
for i, hadm_id in enumerate(test_hadm):
    row = test_scores[i].toarray().ravel()
    if row.sum() == 0:
        popularity_by_dx_preds[hadm_id] = top_meds
        continue
    order = np.argsort(-row)
    popularity_by_dx_preds[hadm_id] = [drugs[j] for j in order if row[j] > 0]

p, r = evaluate(popularity_by_dx_preds, ground_truth, K)
print(f"popularity by diagnosis -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}")


# MODEL 3: KNN NEAREST NEIGHBOR
# admissions where diagnosis profile is most similar to this admission
from sklearn.metrics.pairwise import cosine_similarity

train_drug_lookup = (
    train_interactions.groupby("hadm_id")["medication"].apply(list).to_dict()
)

KNN_NEIGHBORS = 50

knn_preds = {}
batch_size = 500
for batch_start in range(0, len(test_rows), batch_size):
    batch = test_icd[batch_start : batch_start + batch_size]
    sims = cosine_similarity(batch, train_icd)
    for i, row_sims in enumerate(sims):
        hadm_id = test_hadm[batch_start + i]
        top_neighbors = np.argsort(row_sims)[::-1][:KNN_NEIGHBORS]
        drug_scores = {}
        for neighbor_idx in top_neighbors:
            sim = row_sims[neighbor_idx]
            for drug in train_drug_lookup.get(train_hadm[neighbor_idx], []):
                drug_scores[drug] = drug_scores.get(drug, 0) + sim
        knn_preds[hadm_id] = sorted(drug_scores, key=drug_scores.get, reverse=True)

p, r = evaluate(knn_preds, ground_truth, K)
print(
    f"nearest neighbor (k={KNN_NEIGHBORS}) -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}"
)
