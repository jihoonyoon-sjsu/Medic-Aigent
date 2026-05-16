import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "8"

import time
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MaxAbsScaler

from lightgbm import LGBMRanker, early_stopping, log_evaluation
from lightfm import LightFM
from implicit.als import AlternatingLeastSquares

from config import PROCESSED_DIR, RANDOM_STATE, LAB_SOURCES


K = 20
TEST_SIZE = 0.20
TOP_N_SAVED = 100

MODELS_DIR = PROCESSED_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_PATH = MODELS_DIR / "artifacts.pkl"


# split by patient, so that same patient doesn't show up in train and test
# this would cause leakage, so that all history stays on one side
def split_by_patient(features):
    rng = np.random.default_rng(RANDOM_STATE)
    subjects = features["subject_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(subjects)

    n_test = int(len(subjects) * TEST_SIZE)
    test_subjects = set(subjects[:n_test].tolist())
    train_subjects = set(subjects[n_test:].tolist())

    train_hadm_ids = features.loc[features["subject_id"].isin(train_subjects), "hadm_id"]
    test_hadm_ids = features.loc[features["subject_id"].isin(test_subjects), "hadm_id"]
    return set(train_hadm_ids), set(test_hadm_ids)


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
            dcg += 1.0 / np.log2(i + 2)

    # ideal is if our order is all right
    ideal_hits = min(len(true_drugs), K)
    ideal_dcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# for the feature table
def build_dense_feature_matrix(features):
    drop_cols = ["subject_id", "hadm_id", "admittime"]
    dense = features.drop(columns=drop_cols, errors="ignore")
    dense = pd.get_dummies(dense, dummy_na=True).fillna(0)
    return csr_matrix(dense.to_numpy(dtype=np.float32))


# fill missing values with median of train
def impute_labs(values, train_rows):
    out = values.copy()
    medians = np.nanmedian(values[train_rows], axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for j in range(out.shape[1]):
        mask = np.isnan(out[:, j])
        out[mask, j] = medians[j]
    return out


def load_lab_source(source, train_rows):
    current = np.load(PROCESSED_DIR / f"current_{source}_labs.npz")
    prior = np.load(PROCESSED_DIR / f"prior_{source}_labs.npz")

    current_values = impute_labs(current["values"], train_rows)
    prior_values = impute_labs(prior["values"], train_rows)
    block = np.concatenate(
        [current_values, current["flags"], prior_values, prior["flags"]], axis=1
    )
    return csr_matrix(block.astype(np.float32))


# normalize and scale
def field_normalize(mat):
    nnz = np.diff(mat.indptr).astype(np.float32)
    nnz[nnz == 0] = 1.0
    return (diags(1.0 / nnz) @ mat).astype(np.float32)


def scale_train_fit(mat, train_rows):
    scaler = MaxAbsScaler()
    scaler.fit(mat[train_rows])
    return scaler.transform(mat).astype(np.float32)


# add in the dense matrices
def build_full_feature_matrix(features, train_rows):
    current_dx = field_normalize(load_npz(PROCESSED_DIR / "current_dx_matrix.npz"))
    prior_dx = field_normalize(load_npz(PROCESSED_DIR / "prior_dx_matrix.npz"))
    prior_med = field_normalize(load_npz(PROCESSED_DIR / "prior_med_matrix.npz"))
    current_proc = field_normalize(load_npz(PROCESSED_DIR / "current_proc_matrix.npz"))
    prior_proc = field_normalize(load_npz(PROCESSED_DIR / "prior_proc_matrix.npz"))
    dense = scale_train_fit(build_dense_feature_matrix(features), train_rows)

    blocks = [current_dx, prior_dx, prior_med, current_proc, prior_proc, dense]
    for source in LAB_SOURCES:
        blocks.append(scale_train_fit(load_lab_source(source, train_rows), train_rows))
    return hstack(blocks, format="csr")


metrics = {}
all_preds = {}


print("loading processed tables...")
features = pd.read_csv(PROCESSED_DIR / "patient_admission_snapshot.csv", low_memory=False)
labels = pd.read_csv(PROCESSED_DIR / "admission_drug_labels.csv", low_memory=False)

# handle type
features["admittime"] = pd.to_datetime(features["admittime"])
features["hadm_id"] = features["hadm_id"].astype(int)
labels["hadm_id"] = labels["hadm_id"].astype(int)

interactions = labels[labels["label"] == 1].rename(columns={"candidate_drug": "medication"})

train_ids, test_ids = split_by_patient(features)

train_interactions = interactions[interactions["hadm_id"].isin(train_ids)]
test_interactions = interactions[interactions["hadm_id"].isin(test_ids)]

ground_truth = {}
for hadm_id, grp in test_interactions.groupby("hadm_id"):
    ground_truth[hadm_id] = set(grp["medication"])

top_meds = train_interactions["medication"].value_counts().index.tolist()

print("admissions:", len(features), "train:", len(train_ids), "test:", len(test_ids))

hadm_ids = features["hadm_id"].to_numpy()
hadm_to_row = {h: i for i, h in enumerate(hadm_ids)}

train_rows = np.where(features["hadm_id"].isin(train_ids).to_numpy())[0]
test_rows = np.where(features["hadm_id"].isin(test_ids).to_numpy())[0]
train_hadm = hadm_ids[train_rows]
test_hadm = hadm_ids[test_rows]

print("building feature matrix...")
feature_matrix = build_full_feature_matrix(features, train_rows)
feature_matrix.indptr = feature_matrix.indptr.astype(np.int32)
feature_matrix.indices = feature_matrix.indices.astype(np.int32)

med_vocab = np.array(sorted(train_interactions["medication"].unique()))
med_to_col = {med: i for i, med in enumerate(med_vocab)}
med_ids = np.arange(len(med_vocab), dtype=np.int32)
n_meds = len(med_vocab)


def save_artifacts():
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump({
            "predictions": all_preds,
            "metrics": metrics,
            "ground_truth": ground_truth,
            "train_ids": train_ids,
            "test_ids": test_ids,
        }, f)
    print(f"saved to {ARTIFACTS_PATH}")


def drug_onehot(cols):
    n = len(cols)
    return csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), cols)),
        shape=(n, n_meds),
    )


def build_candidate_block(batch_rows):
    b = len(batch_rows)
    repeat_idx = np.repeat(np.arange(b), n_meds)
    drug_block = drug_onehot(np.tile(med_ids, b))
    return hstack([feature_matrix[batch_rows][repeat_idx], drug_block], format="csr")


def trim_preds(preds):
    return {h: list(v)[:TOP_N_SAVED] for h, v in preds.items()}


# save stuff needed for app.py
def save_pickle(name, obj):
    with open(MODELS_DIR / name, "wb") as f:
        pickle.dump(obj, f)


# ========== MODEL 1: overall medication popularity ==========

# same list to every admission
popularity_preds = {}
for hadm_id in ground_truth:
    popularity_preds[hadm_id] = top_meds
p, r, n = evaluate(popularity_preds, ground_truth)
metrics["popularity"] = (p, r, n)
print(f"\npopularity  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}")
all_preds["popularity"] = trim_preds(popularity_preds)
save_artifacts()

# shared data for all model saves
save_pickle("shared.pkl", {
    "feature_matrix": feature_matrix,
    "med_vocab": med_vocab,
    "hadm_to_row": hadm_to_row,
    "med_to_col": med_to_col,
    "top_meds": top_meds,
    "n_meds": n_meds,
})


# ========== MODEL 2: KNN ==========

KNN_NEIGHBORS = 50
KNN_BATCH_SIZE = 500

t0 = time.time()
print("building full-feature KNN baseline...")
train_features = feature_matrix[train_rows]
test_features = feature_matrix[test_rows]

train_drugs_by_hadm = {}
for hadm_id, grp in train_interactions.groupby("hadm_id"):
    train_drugs_by_hadm[hadm_id] = list(grp["medication"])

drug_rows, drug_cols = [], []
for pos, hadm_id in enumerate(train_hadm):
    for drug in train_drugs_by_hadm.get(hadm_id, []):
        drug_rows.append(pos)
        drug_cols.append(med_to_col[drug])
train_drug_matrix = csr_matrix(
    (np.ones(len(drug_rows), dtype=np.float32), (drug_rows, drug_cols)),
    shape=(len(train_rows), n_meds),
)

knn_preds = {}
for start in range(0, len(test_rows), KNN_BATCH_SIZE):
    batch = test_features[start:start + KNN_BATCH_SIZE]
    sims = cosine_similarity(batch, train_features)

    for i, row_sims in enumerate(sims):
        hadm_id = test_hadm[start + i]
        top_idx = np.argpartition(row_sims, -KNN_NEIGHBORS)[-KNN_NEIGHBORS:]
        top_sims = row_sims[top_idx]
        pos_mask = top_sims > 0
        top_idx = top_idx[pos_mask]
        top_sims = top_sims[pos_mask]

        if len(top_idx) == 0:
            knn_preds[hadm_id] = top_meds
            continue

        drug_scores = np.asarray(top_sims @ train_drug_matrix[top_idx]).ravel()
        knn_preds[hadm_id] = med_vocab[np.argsort(drug_scores)[::-1]].tolist()

p, r, n = evaluate(knn_preds, ground_truth)
metrics["knn"] = (p, r, n)
print(f"knn  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["knn"] = trim_preds(knn_preds)
save_artifacts()

save_pickle("knn.pkl", {
    "train_features": train_features,
    "train_drug_matrix": train_drug_matrix,
})


# ========== MODEL 3: ALS ==========

ALS_FACTORS = 64  # latent factors for ALS
ALS_EPOCHS = 10
ALS_REG = 0.01

# all users/subjects in train
train_subjects = np.sort(
    features.loc[train_rows, "subject_id"].drop_duplicates().to_numpy()
)
# subject/med id to index mapping
subj_to_row = {subj: i for i, subj in enumerate(train_subjects)}

# build unique subject/med pairs
hadm_to_subj = features[["hadm_id", "subject_id"]]
patient_med_pairs = train_interactions.merge(hadm_to_subj, on="hadm_id", how="left")
patient_med_pairs = patient_med_pairs[["subject_id", "medication"]].drop_duplicates()

# subject-drug matrxi
# 1 means seen, 0 means not observed
subj_rows = patient_med_pairs["subject_id"].map(subj_to_row).to_numpy()
med_cols = patient_med_pairs["medication"].map(med_to_col).to_numpy()
patient_item_matrix = csr_matrix(
    (np.ones(len(patient_med_pairs), dtype=np.float32), (subj_rows, med_cols)),
    shape=(len(train_subjects), len(med_vocab)),
)

t0 = time.time()
print("training ALS baseline...")
als_model = AlternatingLeastSquares(
    factors=ALS_FACTORS,
    regularization=ALS_REG,
    iterations=ALS_EPOCHS,
    random_state=RANDOM_STATE,
)
als_model.fit(patient_item_matrix)

# uses prior med list to predict new medications
prior_med_matrix = load_npz(PROCESSED_DIR / "prior_med_matrix.npz")
all_prior_meds = np.array(sorted(interactions["medication"].unique()))

als_preds = {}
for row_idx, hadm_id in zip(test_rows, test_hadm):
    prior_indices = prior_med_matrix[row_idx].indices

    cols = []
    for idx in prior_indices:
        med = all_prior_meds[idx]
        if med in med_to_col:
            cols.append(med_to_col[med])

    if len(cols) == 0:
        als_preds[hadm_id] = top_meds
        continue

    user_items = csr_matrix(
        (np.ones(len(cols), dtype=np.float32),
         (np.zeros(len(cols), dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(1, n_meds),
    )
    ids, _ = als_model.recommend(
        0, user_items, N=n_meds,
        recalculate_user=True,
        filter_already_liked_items=False,
    )
    als_preds[hadm_id] = med_vocab[ids].tolist()

p, r, n = evaluate(als_preds, ground_truth)
metrics["als"] = (p, r, n)
print(f"als  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["als"] = trim_preds(als_preds)
save_artifacts()


# ========== MODEL 4: LightFM ==========

LIGHTFM_FACTORS = 32
LIGHTFM_EPOCHS = 50
LIGHTFM_THREADS = 8
LIGHTFM_BATCH_SIZE = 500

# CF with matrix factorization but w/ all side features
# ALS only sees patient/med, so this works with a lot more context
admission_drug_pairs = train_interactions[["hadm_id", "medication"]].drop_duplicates()

# (admission, drug) interaction
interaction_rows, interaction_cols = [], []
for hadm_id, med in zip(
    admission_drug_pairs["hadm_id"], admission_drug_pairs["medication"]
):
    interaction_rows.append(hadm_to_row[hadm_id])
    interaction_cols.append(med_to_col[med])
lightfm_interactions = csr_matrix(
    (np.ones(len(interaction_rows), dtype=np.float32), (interaction_rows, interaction_cols)),
    shape=(len(features), len(med_vocab)),
)

# scaled, normalized features
lightfm_user_features = feature_matrix

t0 = time.time()
print("training LightFM baseline...")
lightfm_model = LightFM(
    no_components=LIGHTFM_FACTORS,
    loss="warp",
    random_state=RANDOM_STATE,
)
lightfm_model.fit(
    lightfm_interactions,
    user_features=lightfm_user_features,
    epochs=LIGHTFM_EPOCHS,
    num_threads=LIGHTFM_THREADS,
    verbose=True,
)

lightfm_preds = {}
for start in range(0, len(test_rows), LIGHTFM_BATCH_SIZE):
    batch_rows = test_rows[start:start + LIGHTFM_BATCH_SIZE]
    batch_hadm = test_hadm[start:start + LIGHTFM_BATCH_SIZE]

    user_ids = np.repeat(batch_rows, n_meds)
    item_ids = np.tile(med_ids, len(batch_rows))

    scores = lightfm_model.predict(
        user_ids, item_ids,
        user_features=lightfm_user_features,
        num_threads=LIGHTFM_THREADS,
    ).reshape(len(batch_rows), n_meds)

    for i, hadm_id in enumerate(batch_hadm):
        # sort highest first
        ranking = np.argsort(scores[i])[::-1]
        lightfm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(lightfm_preds, ground_truth)
metrics["lightfm"] = (p, r, n)
print(f"lightfm  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["lightfm"] = trim_preds(lightfm_preds)
save_artifacts()

save_pickle("lightfm.pkl", lightfm_model)


# ========== MODEL 5: LightGBM ==========

# tree models can learn more expressive non-linear patterns
# each row is (admission, drug) pair -> 0/1 label
# no embeddings like previous models
LGBM_BATCH_SIZE = 1000

lgbm_all = labels[labels["hadm_id"].isin(train_ids)].copy()
lgbm_all = lgbm_all[lgbm_all["candidate_drug"].isin(med_to_col)]

all_train_hadm = np.array(sorted(train_ids))
np.random.default_rng(RANDOM_STATE + 1).shuffle(all_train_hadm)
n_val = int(len(all_train_hadm) * 0.1)
val_hadm_set = set(all_train_hadm[:n_val].tolist())

train_df = (
    lgbm_all[~lgbm_all["hadm_id"].isin(val_hadm_set)]
    .sort_values("hadm_id")
    .reset_index(drop=True)
)
val_df = (
    lgbm_all[lgbm_all["hadm_id"].isin(val_hadm_set)]
    .sort_values("hadm_id")
    .reset_index(drop=True)
)

train_row_idx = train_df["hadm_id"].map(hadm_to_row).to_numpy()
train_col_idx = train_df["candidate_drug"].map(med_to_col).to_numpy()
train_y = train_df["label"].to_numpy()
train_groups = train_df.groupby("hadm_id", sort=False).size().to_numpy()
# one-hot drugs
train_x = hstack([feature_matrix[train_row_idx], drug_onehot(train_col_idx)], format="csr")
train_x.sort_indices()

val_row_idx = val_df["hadm_id"].map(hadm_to_row).to_numpy()
val_col_idx = val_df["candidate_drug"].map(med_to_col).to_numpy()
val_y = val_df["label"].to_numpy()
val_groups = val_df.groupby("hadm_id", sort=False).size().to_numpy()
val_x = hstack([feature_matrix[val_row_idx], drug_onehot(val_col_idx)], format="csr")
val_x.sort_indices()

t0 = time.time()
print("training LightGBM LambdaRank...")
lgbm_model = LGBMRanker(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=127,
    objective="lambdarank",  # rank, not classify
    label_gain=[0, 1],
    eval_at=[K],
    random_state=RANDOM_STATE,
    n_jobs=8,
    force_col_wise=True,
    max_bin=63,
    verbose=-1,
    device='gpu',
)
lgbm_model.fit(
    train_x, train_y, group=train_groups,
    eval_set=[(val_x, val_y)],
    eval_group=[val_groups],
    callbacks=[early_stopping(50), log_evaluation(100)],
)

# prediction
lgbm_preds = {}
for start in range(0, len(test_rows), LGBM_BATCH_SIZE):
    batch_rows = test_rows[start:start + LGBM_BATCH_SIZE]
    batch_hadm = test_hadm[start:start + LGBM_BATCH_SIZE]
    scores = lgbm_model.predict(build_candidate_block(batch_rows)).reshape(
        len(batch_rows), n_meds
    )
    for i, hadm_id in enumerate(batch_hadm):
        ranking = np.argsort(scores[i])[::-1]
        lgbm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(lgbm_preds, ground_truth)
metrics["lgbm"] = (p, r, n)
print(f"lgbm  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["lgbm"] = trim_preds(lgbm_preds)
save_artifacts()

save_pickle("lgbm.pkl", lgbm_model)


# ========== MODEL 6: DeepFM ==========

DEEPFM_FACTORS = 32
DEEPFM_EPOCHS = 5
DEEPFM_LR = 0.001
DEEPFM_BATCH_SIZE = 512
DEEPFM_PRED_BATCH = 512

# need to import here because of conflicts
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device: {device}")

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
            torch.arange(n_rows, device=idx.device), counts
        )

        summed = torch.zeros(n_rows, DEEPFM_FACTORS, device=idx.device)
        squared = torch.zeros(n_rows, DEEPFM_FACTORS, device=idx.device)
        summed.index_add_(0, row_ids, emb)
        squared.index_add_(0, row_ids, emb * emb)

        # deep
        fm_part = 0.5 * ((summed * summed) - squared).sum(1)
        deep_part = self.deep(summed).squeeze(1)
        return linear_part + fm_part + deep_part


def sparse_to_torch(x):
    x = x.tocsr()
    counts = np.diff(x.indptr)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    idx = torch.from_numpy(x.indices.astype(np.int64)).to(device)
    vals = torch.from_numpy(x.data.astype(np.float32)).to(device)
    return idx, torch.from_numpy(offsets).to(device), vals


def train_torch(cls, feat_matrix, row_ids, drug_cols, train_y, n_features, name, bs, epochs, lr):
    torch.manual_seed(RANDOM_STATE)
    model = cls(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    rows = np.arange(len(row_ids))
    rng = np.random.default_rng(RANDOM_STATE)
    n_batches = int(np.ceil(len(rows) / bs))

    print(f"  {name}: {len(row_ids):,} train pairs, "
          f"{n_batches:,} batches/epoch, {epochs} epochs", flush=True)

    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(rows)
        epoch_loss = 0.0
        epoch_t0 = time.time()

        for start in range(0, len(rows), bs):
            batch_idx = rows[start:start + bs]
            batch_x = hstack([feat_matrix[row_ids[batch_idx]], drug_onehot(drug_cols[batch_idx])], format="csr")
            idx, offsets, vals = sparse_to_torch(batch_x)
            y = torch.from_numpy(train_y[batch_idx].astype(np.float32)).to(device)

            opt.zero_grad()
            loss = loss_fn(model(idx, offsets, vals), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        print(f"  {name} epoch {epoch}/{epochs} done  "
              f"loss={epoch_loss / n_batches:.4f}  ({time.time() - epoch_t0:.0f}s)",
              flush=True)

    return model


def predict_model(model, x):
    chunk = 16384
    model.eval()
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, chunk):
            sub = x[s:s + chunk]
            idx, offsets, vals = sparse_to_torch(sub)
            out[s:s + chunk] = model(idx, offsets, vals).cpu().numpy()
    return out


train_pairs = labels[labels["hadm_id"].isin(train_ids)].copy()
train_pairs = train_pairs[train_pairs["candidate_drug"].isin(med_to_col)]

row_ids = train_pairs["hadm_id"].map(hadm_to_row).to_numpy()
drug_cols = train_pairs["candidate_drug"].map(med_to_col).to_numpy()
y = train_pairs["label"].to_numpy()
n_features = feature_matrix.shape[1] + n_meds

t0 = time.time()
print("training deepfm model...", flush=True)
deepfm_model = train_torch(
    DeepFM, feature_matrix, row_ids, drug_cols, y, n_features,
    "deepfm", DEEPFM_BATCH_SIZE, DEEPFM_EPOCHS, DEEPFM_LR,
)
torch.save(deepfm_model.state_dict(), MODELS_DIR / "deepfm.pt")

print(f"scoring deepfm on {len(test_rows):,} test admissions...", flush=True)
deepfm_preds = {}
for start in range(0, len(test_rows), DEEPFM_PRED_BATCH):
    batch_rows = test_rows[start:start + DEEPFM_PRED_BATCH]
    batch_hadm = test_hadm[start:start + DEEPFM_PRED_BATCH]
    scores = predict_model(deepfm_model, build_candidate_block(batch_rows)).reshape(
        len(batch_rows), n_meds
    )
    for i, hadm_id in enumerate(batch_hadm):
        ranking = np.argsort(scores[i])[::-1]
        deepfm_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(deepfm_preds, ground_truth)
metrics["deepfm"] = (p, r, n)
print(f"deepfm  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["deepfm"] = trim_preds(deepfm_preds)
save_artifacts()


# ========== MODEL 7: DCN-v2 ==========

DCNV2_EMBED_DIM = 64
DCNV2_N_CROSS = 3
DCNV2_EPOCHS = 5
DCNV2_LR = 0.001
DCNV2_BATCH_SIZE = 512
DCNV2_PRED_BATCH = 512

class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0, x):
        return x0 * self.W(x) + x


class DCNV2(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.embed = nn.EmbeddingBag(
            n_features, DCNV2_EMBED_DIM, mode="sum", include_last_offset=True
        )
        self.cross = nn.ModuleList(
            [CrossLayer(DCNV2_EMBED_DIM) for _ in range(DCNV2_N_CROSS)]
        )
        self.deep = nn.Sequential(
            nn.Linear(DCNV2_EMBED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.out = nn.Linear(DCNV2_EMBED_DIM + 64, 1)

    def forward(self, idx, offsets, vals):
        x0 = self.embed(idx, offsets, per_sample_weights=vals)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.out(torch.cat([x, self.deep(x0)], dim=1)).squeeze(1)


t0 = time.time()
print("training dcnv2 model...", flush=True)
dcnv2_model = train_torch(
    DCNV2, feature_matrix, row_ids, drug_cols, y, n_features,
    "dcnv2", DCNV2_BATCH_SIZE, DCNV2_EPOCHS, DCNV2_LR,
)
torch.save(dcnv2_model.state_dict(), MODELS_DIR / "dcnv2.pt")

print(f"scoring dcnv2 on {len(test_rows):,} test admissions...", flush=True)
dcnv2_preds = {}
for start in range(0, len(test_rows), DCNV2_PRED_BATCH):
    batch_rows = test_rows[start:start + DCNV2_PRED_BATCH]
    batch_hadm = test_hadm[start:start + DCNV2_PRED_BATCH]
    scores = predict_model(dcnv2_model, build_candidate_block(batch_rows)).reshape(
        len(batch_rows), n_meds
    )
    for i, hadm_id in enumerate(batch_hadm):
        ranking = np.argsort(scores[i])[::-1]
        dcnv2_preds[hadm_id] = med_vocab[ranking].tolist()

p, r, n = evaluate(dcnv2_preds, ground_truth)
metrics["dcnv2"] = (p, r, n)
print(f"dcnv2  P@{K}={p:.4f}  R@{K}={r:.4f}  NDCG@{K}={n:.4f}  ({time.time() - t0:.0f}s)")
all_preds["dcnv2"] = trim_preds(dcnv2_preds)
save_artifacts()
