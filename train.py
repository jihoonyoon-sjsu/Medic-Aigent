from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMRanker
from lightfm import LightFM
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
LIGHTFM_FACTORS = 64
LIGHTFM_EPOCHS = 10
LGBM_BATCH_SIZE = 50
DEEPFM_FACTORS = 32
DEEPFM_EPOCHS = 5
DEEPFM_LR = 0.001
DEEPFM_BATCH_SIZE = 512
DEEPFM_PRED_BATCH = 10


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
    return set(features.loc[train_mask, "hadm_id"]), set(
        features.loc[test_mask, "hadm_id"]
    )


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

top_meds = train_interactions["medication"].value_counts().index.tolist()

print("loaded %d admissions" % len(features))
print(f"train admissions: {len(train_ids):,}")
print(f"test admissions: {len(test_ids):,}")

hadm_ids = features["hadm_id"].to_numpy()
hadm_to_row = {h: i for i, h in enumerate(hadm_ids)}

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

feature_matrix = build_full_feature_matrix(features)
med_vocab = np.array(sorted(train_interactions["medication"].unique()))
med_to_col = {m: i for i, m in enumerate(med_vocab)}
med_ids = np.arange(len(med_vocab), dtype=np.int32)
n_meds = len(med_vocab)


# ========== MODEL 1: overall medication popularity ==========
# same list to every admission
popularity_preds = {}
for hadm_id in ground_truth:
    popularity_preds[hadm_id] = top_meds
p, r, n = evaluate(popularity_preds, ground_truth)
print(f"\npopularity -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


# ========== MODEL 2: KNN ==========
print("Building full-feature KNN baseline...")
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


# ========== MODEL 3: BPR ==========
def train_bpr(interaction_matrix):
    rng = np.random.default_rng(RANDOM_STATE)
    num_users, num_items = interaction_matrix.shape

    # BPR - baysian personalized ranking
    # initialize random initial weights
    # shape is num (users/items, 64)
    users = rng.normal(0, 0.05, size=(num_users, BPR_FACTORS)).astype(np.float32)
    meds = rng.normal(0, 0.05, size=(num_items, BPR_FACTORS)).astype(np.float32)

    meds_by_user = []
    train_users = []
    for u in range(num_users):
        # get all medications per user (med ids received)
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


# all users/subjects in train
train_subject_ids = np.sort(
    features.loc[train_rows, "subject_id"].drop_duplicates().to_numpy()
)

# subject/med id to index mapping
subject_to_row = {sid: i for i, sid in enumerate(train_subject_ids)}

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
print(
    f"BPR-MF warm-start admissions in test: {warm_start_count:,} / {len(test_hadm):,}"
)


# ========== MODEL 4: LightFM ==========
# CF with matrix factorization but w/ all side features
# BPR only sees patient/med, so this works with a lot more context
train_admission_pairs = train_interactions[["hadm_id", "medication"]].drop_duplicates()

# (admission, drug) interaction
interaction_rows = []
interaction_cols = []
for hadm_id, med in zip(
    train_admission_pairs["hadm_id"], train_admission_pairs["medication"]
):
    interaction_rows.append(hadm_to_row[hadm_id])
    interaction_cols.append(med_to_col[med])

interaction_data = np.ones(len(interaction_rows), dtype=np.float32)
lightfm_interactions = csr_matrix(
    (interaction_data, (interaction_rows, interaction_cols)),
    shape=(len(features), len(med_vocab)),
)

print("Training LightFM baseline...")
# loss is warp loss - good for implicit feedbakc and for ranking.
lightfm_model = LightFM(
    no_components=LIGHTFM_FACTORS,  # 64 as before
    loss="warp",
    random_state=RANDOM_STATE,
)
lightfm_model.fit(
    lightfm_interactions,
    user_features=feature_matrix,  # our feature matrix from prepare.py
    epochs=LIGHTFM_EPOCHS,
)

lightfm_preds = {}
for start in range(0, len(test_rows), BATCH_SIZE):
    batch_rows = test_rows[start : start + BATCH_SIZE]
    batch_hadm = test_hadm[start : start + BATCH_SIZE]

    user_ids = np.repeat(batch_rows, n_meds)
    item_ids = np.tile(med_ids, len(batch_rows))

    scores = lightfm_model.predict(user_ids, item_ids, user_features=feature_matrix)
    scores = scores.reshape(len(batch_rows), n_meds)

    for i, hadm_id in enumerate(batch_hadm):
        # sort highest first
        ranking = np.argsort(scores[i])[::-1]
        lightfm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(lightfm_preds, ground_truth)
print(f"LightFM -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


# ========== MODEL 5: LightGBM ==========
# tree models can learn more expressive non-linear patterns
# each row is (admission, drug) pair -> 0/1 label
# no embeddings like previous models
lgbm_train = labels[labels["hadm_id"].isin(train_ids)].copy()
lgbm_train = lgbm_train[lgbm_train["candidate_drug"].isin(med_to_col)]
lgbm_train = lgbm_train.sort_values("hadm_id").reset_index(drop=True)

lgbm_rows = lgbm_train["hadm_id"].map(hadm_to_row).to_numpy()
lgbm_cols = lgbm_train["candidate_drug"].map(med_to_col).to_numpy()
lgbm_y = lgbm_train["label"].to_numpy()
lgbm_group = lgbm_train.groupby("hadm_id", sort=False).size().to_numpy()

# one-hot drugs
lgbm_drugs = csr_matrix(
    (
        np.ones(len(lgbm_train), dtype=np.float32),
        (np.arange(len(lgbm_train)), lgbm_cols),
    ),
    shape=(len(lgbm_train), n_meds),
)
lgbm_x = hstack([feature_matrix[lgbm_rows], lgbm_drugs], format="csr")

print("Training LightGBM LambdaRank model...")
lgbm_model = LGBMRanker(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    objective="lambdarank",  # rank, not classify
    label_gain=[0, 1],
    eval_at=[K],
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
lgbm_model.fit(lgbm_x, lgbm_y, group=lgbm_group)


# prediction
def build_candidate_block(batch_rows):
    n_users = len(batch_rows)
    n_pairs = n_users * n_meds

    user_rows = np.repeat(batch_rows, n_meds)
    drug_cols = np.tile(med_ids, n_users)

    drug_block = csr_matrix(
        (np.ones(n_pairs, dtype=np.float32), (np.arange(n_pairs), drug_cols)),
        shape=(n_pairs, n_meds),
    )
    return hstack([feature_matrix[user_rows], drug_block], format="csr")


lgbm_preds = {}
for start in range(0, len(test_rows), LGBM_BATCH_SIZE):
    batch_rows = test_rows[start : start + LGBM_BATCH_SIZE]
    batch_hadm = test_hadm[start : start + LGBM_BATCH_SIZE]

    batch_x = build_candidate_block(batch_rows)
    scores = lgbm_model.predict(batch_x).reshape(len(batch_rows), n_meds)

    for i, hadm_id in enumerate(batch_hadm):
        ranking = np.argsort(scores[i])[::-1]
        lgbm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(lgbm_preds, ground_truth)
print(f"LightGBM -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


# ========== MODEL 6: DeepFM ==========
# deep learning approach
# takes input and outputs three components: linear/fm/deep parts
# it's like logistic regression + lightFM + MLP
class DeepFM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.EmbeddingBag(
            n_features, 1, mode="sum", include_last_offset=True
        )
        self.fm = nn.Embedding(n_features, DEEPFM_FACTORS)
        self.deep = nn.Sequential(
            nn.Linear(DEEPFM_FACTORS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, idx, offsets, vals):
        # linear
        linear_part = self.linear(idx, offsets, per_sample_weights=vals).squeeze(1)

        # FM
        emb = self.fm(idx) * vals.unsqueeze(1)
        n_rows = len(offsets) - 1
        counts = offsets[1:] - offsets[:-1]
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=idx.device),
            counts,
        )

        summed = torch.zeros(n_rows, DEEPFM_FACTORS, device=idx.device)
        squared = torch.zeros(n_rows, DEEPFM_FACTORS, device=idx.device)
        summed.index_add_(0, row_ids, emb)
        squared.index_add_(0, row_ids, emb * emb)

        # deep
        fm_part = 0.5 * ((summed * summed) - squared).sum(1)
        deep_part = self.deep(summed).squeeze(1)
        return linear_part + fm_part + deep_part


# convert for pytorch
def sparse_to_torch(x):
    x = x.tocsr()
    counts = np.diff(x.indptr)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    idx = torch.from_numpy(x.indices.astype(np.int64))
    vals = torch.from_numpy(x.data.astype(np.float32))
    offsets = torch.from_numpy(offsets)
    return idx, offsets, vals


# trainign loop
def train_deepfm(train_x, train_y):
    torch.manual_seed(RANDOM_STATE)
    model = DeepFM(train_x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=DEEPFM_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    rows = np.arange(train_x.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)

    model.train()
    for _ in range(DEEPFM_EPOCHS):
        rng.shuffle(rows)
        for start in range(0, len(rows), DEEPFM_BATCH_SIZE):
            batch_rows = rows[start : start + DEEPFM_BATCH_SIZE]
            idx, offsets, vals = sparse_to_torch(train_x[batch_rows])
            y = torch.from_numpy(train_y[batch_rows].astype(np.float32))

            opt.zero_grad()
            loss = loss_fn(model(idx, offsets, vals), y)
            loss.backward()
            opt.step()

    return model


def predict_deepfm(model, x):
    model.eval()
    with torch.no_grad():
        idx, offsets, vals = sparse_to_torch(x)
        return model(idx, offsets, vals).numpy()


print("Training DeepFM model...")
# same input shape as LightGBM: admission features to one hot drugs
deepfm_train = labels[labels["hadm_id"].isin(train_ids)].copy()
deepfm_train = deepfm_train[deepfm_train["candidate_drug"].isin(med_to_col)]

deepfm_rows = deepfm_train["hadm_id"].map(hadm_to_row).to_numpy()
deepfm_cols = deepfm_train["candidate_drug"].map(med_to_col).to_numpy()
deepfm_y = deepfm_train["label"].to_numpy()

deepfm_drugs = csr_matrix(
    (
        np.ones(len(deepfm_train), dtype=np.float32),
        (np.arange(len(deepfm_train)), deepfm_cols),
    ),
    shape=(len(deepfm_train), n_meds),
)
deepfm_x = hstack([feature_matrix[deepfm_rows], deepfm_drugs], format="csr")
deepfm_model = train_deepfm(deepfm_x, deepfm_y)

deepfm_preds = {}
for start in range(0, len(test_rows), DEEPFM_PRED_BATCH):
    batch_rows = test_rows[start : start + DEEPFM_PRED_BATCH]
    batch_hadm = test_hadm[start : start + DEEPFM_PRED_BATCH]

    batch_x = build_candidate_block(batch_rows)
    scores = predict_deepfm(deepfm_model, batch_x)
    scores = scores.reshape(len(batch_rows), n_meds)

    for i, hadm_id in enumerate(batch_hadm):
        ranking = np.argsort(scores[i])[::-1]
        deepfm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(deepfm_preds, ground_truth)
print(f"DeepFM -- precision@{K}: {p:.4f}, recall@{K}: {r:.4f}, ndcg@{K}: {n:.4f}")


# ========== MODEL 7: etc - sequential? mixed/hybrid? ==========
