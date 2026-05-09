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
BPR_FACTORS = 64  # latent factors for BPR
BPR_EPOCHS = 10
BPR_LR = 0.03
BPR_REG = 0.01


# split by patient, so that same patient doesn't show up in train and test
# this would cause leakage, so that all history stays on one side
def split_by_patient(features):
    rng = np.random.default_rng(RANDOM_STATE)
    subject_ids = features["subject_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(subject_ids)

    n_test = int(len(subject_ids) * TEST_SIZE)
    test_subjects = set(subject_ids[:n_test].tolist())
    train_subjects = set(subject_ids[n_test:].tolist())

    train_mask = features["subject_id"].isin(train_subjects)
    test_mask = features["subject_id"].isin(test_subjects)
    return set(features.loc[train_mask, "hadm_id"]), set(features.loc[test_mask, "hadm_id"])


# precision@k, recall@k, and ndcg@k
# we are predicting hadm_id -> ranked list of medications
# our test is hadm_id -> medications actually administered
def evaluate(predictions, ground_truth):
    p_list, r_list, n_list = [], [], []

    for hadm_id, true_drugs in ground_truth.items():
        top_k = predictions.get(hadm_id, [])[:K]
        hits = set(top_k) & true_drugs

        # out of the K recommended drugs, how many were correct
        p_list.append(len(hits) / K)
        # out of the drugs actually given, how many did we include
        r_list.append(len(hits) / len(true_drugs))
        # rewards putting correct drugs higher
        n_list.append(ndcg_at_k(top_k, true_drugs))

    return np.mean(p_list), np.mean(r_list), np.mean(n_list)


# ndcg@k for an admission
def ndcg_at_k(preds, true_drugs):
    dcg = 0.0
    for i, drug in enumerate(preds):
        if drug in true_drugs:
            dcg += 1 / np.log2(i + 2)

    # ideal is if our order is all right
    ideal_hits = min(len(true_drugs), K)
    ideal_dcg = sum(1 / np.log2(r + 1) for r in range(1, ideal_hits + 1))

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# for the feature table
def build_dense_feature_matrix(features):
    drop_cols = ["subject_id", "hadm_id", "admittime", "edregtime", "edouttime"]
    dense = features.drop(columns=drop_cols, errors="ignore")
    dense = pd.get_dummies(dense, dummy_na=True).fillna(0)
    return csr_matrix(dense.to_numpy(dtype=np.float32))


# add in the dense matrices
def build_full_feature_matrix(features):
    current_dx = load_npz(PROCESSED_DIR / "current_dx_matrix.npz")
    prior_dx = load_npz(PROCESSED_DIR / "prior_dx_matrix.npz")
    prior_med = load_npz(PROCESSED_DIR / "prior_med_matrix.npz")
    dense_features = build_dense_feature_matrix(features)

    return hstack([current_dx, prior_dx, prior_med, dense_features], format="csr")


# BPR - baysian personalized ranking
def train_bpr(interaction_matrix):
    rng = np.random.default_rng(RANDOM_STATE)
    num_users, num_items = interaction_matrix.shape

    # initialize random initial weights
    # shape is num (users/items, 64)
    users = rng.normal(0, 0.05, size=(num_users, BPR_FACTORS)).astype(np.float32)
    meds = rng.normal(0, 0.05, size=(num_items, BPR_FACTORS)).astype(np.float32)

    meds_by_user = []
    train_users = []
    # get all medications per user (med ids received)
    for u in range(num_users):
        user_meds = interaction_matrix[u].indices
        meds_by_user.append(user_meds)
        # add to train users if they some meds
        if len(user_meds) > 0 and len(user_meds) < num_items:
            train_users.append(u)

    if len(train_users) == 0:
        return meds

    n_samples = len(train_users) * 20
    for _ in range(BPR_EPOCHS):
        for _ in range(n_samples):
            u = rng.choice(train_users)
            user_meds = meds_by_user[u]

            pos = rng.choice(user_meds)
            neg = rng.integers(num_items)
            while neg in user_meds:
                neg = rng.integers(num_items)

            # (u, pos, neg) we are training on
            u_vec = users[u].copy()
            p_vec = meds[pos].copy()
            n_vec = meds[neg].copy()

            # we want this to be big
            diff = u_vec @ (p_vec - n_vec)
            w = 1.0 / (1.0 + np.exp(diff))

            users[u] += BPR_LR * (w * (p_vec - n_vec) - BPR_REG * u_vec)
            meds[pos] += BPR_LR * (w * u_vec - BPR_REG * p_vec)
            meds[neg] += BPR_LR * (-w * u_vec - BPR_REG * n_vec)

    return meds


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


# === MODEL 1: overall medication popularity ===
# same list to every admission
popularity_preds = {}
for hadm_id in ground_truth:
    popularity_preds[hadm_id] = top_meds
p, r, n = evaluate(popularity_preds, ground_truth)
print(f"\npopularity -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


hadm_ids = features["hadm_id"].to_numpy()

tr_rows, te_rows = [], []
for i, h in enumerate(hadm_ids):
    if h in train_ids:
        tr_rows.append(i)
    elif h in test_ids:
        te_rows.append(i)

train_rows = np.array(tr_rows)
test_rows = np.array(te_rows)
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
    sims = cosine_similarity(batch, train_features)

    for i, row_sims in enumerate(sims):
        hadm_id = test_hadm[start + i]
        neighbor_rows = np.argsort(row_sims)[::-1][:KNN_NEIGHBORS]

        drug_scores = {}
        for nr in neighbor_rows:
            s = row_sims[nr]
            if s <= 0:
                continue
            n_hadm = train_hadm[nr]
            for drug in train_drugs_by_hadm.get(n_hadm, []):
                drug_scores[drug] = drug_scores.get(drug, 0) + s

        knn_preds[hadm_id] = sorted(drug_scores, key=drug_scores.get, reverse=True)

p, r, n = evaluate(knn_preds, ground_truth)
print(
    f"full-feature KNN -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}"
)


# === MODEL 3: BPR ===

# all users/subjects in train
train_subject_ids = np.sort(features.loc[train_rows, "subject_id"].drop_duplicates().to_numpy())
med_vocab = np.array(sorted(train_interactions["medication"].unique()))

# subject/med id to index mapping
subject_to_row = {sid: i for i, sid in enumerate(train_subject_ids)}
med_to_col = {m: i for i, m in enumerate(med_vocab)}

# build unique subject/med pairs
hadm_to_subj = features[["hadm_id", "subject_id"]]
train_pairs = train_interactions.merge(hadm_to_subj, on="hadm_id", how="left")
train_pairs = train_pairs[["subject_id", "medication"]].drop_duplicates()

# subject-drug matrxi
# 1 means seen, 0 means not observed
rows_idx = train_pairs["subject_id"].map(subject_to_row).to_numpy()
cols_idx = train_pairs["medication"].map(med_to_col).to_numpy()
data = np.ones(len(train_pairs), dtype=np.float32)
train_patient_item_matrix = csr_matrix(
    (data, (rows_idx, cols_idx)),
    shape=(len(train_subject_ids), len(med_vocab)),
)

print("Training BPR-MF baseline...")
# one learned vector per med
bpr_med_factors = train_bpr(train_patient_item_matrix)
# uses prior med list to predict new medications
prior_med_matrix = load_npz(PROCESSED_DIR / "prior_med_matrix.npz")
all_prior_meds = np.array(sorted(interactions["medication"].unique()))

bpr_preds = {}
warm_start_count = 0
# training loop:
for row_idx, hadm_id in zip(test_rows, test_hadm):
    prior_indices = prior_med_matrix[row_idx].indices

    # build medication Id's this user has seen
    cols = []
    for idx in prior_indices:
        if idx >= len(all_prior_meds):
            continue
        med = all_prior_meds[idx]
        if med in med_to_col:
            cols.append(med_to_col[med])

    if len(cols) == 0:
        bpr_preds[hadm_id] = top_meds
        continue

    warm_start_count += 1
    # just a vector of their seen medications
    user_vec = bpr_med_factors[cols].mean(axis=0)
    scores = bpr_med_factors @ user_vec
    # ranked medication list
    bpr_preds[hadm_id] = med_vocab[np.argsort(scores)[::-1]].tolist()

p, r, n = evaluate(bpr_preds, ground_truth)
print(f"BPR-MF -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")
print(f"BPR-MF warm-start admissions in test: {warm_start_count:,} / {len(test_hadm):,}")
