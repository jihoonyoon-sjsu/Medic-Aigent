# front end

import os
import pickle
import streamlit as st

CACHE_PATH = "app_cache.pkl"

MODEL_ORDER = [
    ("logreg_ovr","LogReg OvR"),
    ("lgbm","LightGBM"),
    ("deepfm","DeepFM"),
    ("dcnv2","DCN-v2"),
    ("two_tower_llm","Two-tower + LLM"),
    ("two_tower","Two-tower"),
    ("two_tower_bias","Two-tower + bias"),
    ("lightfm","LightFM"),
    ("knn","KNN"),
    ("als","ALS"),
    ("patient_als","Patient-level ALS"),
    ("popularity","Popularity"),
]
DEFAULT_MODEL = "logreg_ovr"
PATIENT_TAB_TOP_K = 10


@st.cache_resource(show_spinner="loading...")
def load_cache():
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def pill(label, bg="#E3F2FD", fg="#1565C0"):
    return f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;font-size:12px;display:inline-block;margin:2px">{label}</span>'


st.set_page_config(page_title="Medic-Aigent", layout="wide")

if not os.path.exists(CACHE_PATH):
    st.error(f"no cache at {CACHE_PATH}\nrun: python3 build_app_cache.py")
    st.stop()

cache = load_cache()
visits = cache["visits"]
by_subject = cache["by_subject"]
pool_all = cache["pool"]
models_available = set(cache["models"])

model_ids = [mid for mid, _ in MODEL_ORDER if mid in models_available]
model_labels = {mid: label for mid, label in MODEL_ORDER}
label_to_id = {label: mid for mid, label in MODEL_ORDER}
ordered_labels = [model_labels[mid] for mid in model_ids]

@st.cache_resource(show_spinner=False)
def rank_pool(pool, by_subject, visits):
    def score(sid):
        meds = 0
        prior = 0
        for h in by_subject[sid]:
            vv = visits[h]
            meds += len(vv.get("actual_meds", []))
            prior += vv.get("n_prior", 0) or 0
        return (meds + prior, meds, prior)
    return sorted(pool, key=score, reverse=True)

pool_all = rank_pool(pool_all, by_subject, visits)

# sidebar
with st.sidebar:
    st.title("Medic-Aigent")
    st.caption("MIMIC-IV medication recommender")

    search = st.text_input("search patient id", placeholder="e.g. 10000032")
    if search.strip():
        s = search.strip()
        pool = [p for p in pool_all if str(p).startswith(s)]
    else:
        pool = pool_all

    if not pool:
        st.warning("no patients found")
        st.stop()

    subject_id = st.selectbox("patient", pool)
    hadm_ids = sorted(by_subject[subject_id], key=lambda h: visits[h]["admittime"], reverse=True)
    labels = [visits[h]["visit_label"] for h in hadm_ids]

    if len(labels) > 1:
        visit_idx = st.selectbox("visit", range(len(labels)), format_func=lambda i: labels[i])
    else:
        visit_idx = 0
        st.markdown(f"**visit:** {labels[0]}")

    hadm_id = hadm_ids[visit_idx]
    v = visits[hadm_id]

    st.divider()

    if v["n_prior"] == 0:
        st.markdown(pill("Cold start", "#FFF3E0", "#E65100"), unsafe_allow_html=True)
        st.markdown("**Prior admissions:** 0")
    else:
        st.markdown(pill("Warm start", "#E8F5E9", "#1B5E20"), unsafe_allow_html=True)
        st.markdown(f"**Prior admissions:** {v['n_prior']}")
        if v["days_since_last"] is not None:
            st.markdown(f"**Days since last visit:** {v['days_since_last']}")

st.header(f"Patient {subject_id}")

c1, c2, c3 = st.columns(3)
c1.metric("Age", v["age"])
c2.metric("Gender", v["gender"])
c3.metric("Service", v["curr_service"])

tags = " ".join(pill(c) for c in v["comorbidities"]) if v["comorbidities"] else "<em style='color:#888'>None</em>"
st.markdown("<strong>Conditions:</strong> " + tags, unsafe_allow_html=True)
st.divider()

actual_meds = set(v["actual_meds"])
n_actual = len(actual_meds)
k_max = min(50, max(2, n_actual))
k_default = min(20, k_max)

def render_recs(recs, k):
    rows = []
    for i, m in enumerate(recs[:k], 1):
        rows.append({"#": i, "Medication": m, "Hit": "yes" if m in actual_meds else ""})
    return rows

tab_patient, tab_model = st.tabs(["Patient View", "Model Explorer"])

with tab_patient:
    default_recs = v["preds"].get(DEFAULT_MODEL, [])
    top_recs = default_recs[:PATIENT_TAB_TOP_K]

    col_recs, col_side = st.columns(2)

    with col_recs:
        st.subheader("Recommended Medications")
        if top_recs:
            st.dataframe(
                [{"#": i, "Medication": m} for i, m in enumerate(top_recs, 1)],
                hide_index=True, use_container_width=True,
                column_config={"#": st.column_config.NumberColumn(width="small")},
            )
        else:
            st.info("no recommendations available")

    with col_side:
        st.subheader("Past Medications")
        if v["prior_meds"]:
            st.dataframe(
                [{"Medication": m} for m in v["prior_meds"]],
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("No prior medications on file.")

    st.subheader("Diagnoses")
    if v["dx"]:
        grouped = {}
        for s, c in v["dx"]:
            grouped.setdefault(s, []).append(c)
        rows = [{"System": s, "Codes": ", ".join(codes)} for s, codes in grouped.items()]
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No diagnosis codes for this admission.")

with tab_model:
    col_l, _ = st.columns([1, 3])
    with col_l:
        default_idx = model_ids.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_ids else 0
        chosen_label = st.selectbox("Model", ordered_labels, index=default_idx)
        name = label_to_id[chosen_label]
        st.caption(f"{n_actual} actual meds, k capped at {k_max}")
        if n_actual < 2:
            k = max(1, n_actual)
            st.markdown(f"**Top K:** {k}")
        else:
            k = st.slider("Top K", 1, k_max, k_default)

    recs = v["preds"].get(name, [])
    hits = [m for m in recs[:k] if m in actual_meds]
    p = len(hits) / k if k else 0.0
    r = len(hits) / n_actual if n_actual else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Precision@{k}", f"{p:.3f}")
    m2.metric(f"Recall@{k}", f"{r:.3f}")
    m3.metric("Hits", len(hits))

    top_k_set = set(recs[:k])
    col1, col2 = st.columns(2)
    col1.subheader("Recommended")
    col1.dataframe(render_recs(recs, k), hide_index=True, use_container_width=True)
    col2.subheader("Actually Prescribed")
    actual_rows = sorted(
        [{"Medication": m, "Captured": "yes" if m in top_k_set else ""} for m in actual_meds],
        key=lambda r: (r["Captured"] != "yes", r["Medication"]),
    )
    col2.dataframe(actual_rows, hide_index=True, use_container_width=True)

    with st.expander("overall test metrics"):
        metrics = cache["metrics"]
        rows = [
            {
                "Model": model_labels[mid],
                "P@20": round(metrics[mid][0], 4),
                "R@20": round(metrics[mid][1], 4),
                "NDCG@20": round(metrics[mid][2], 4),
            }
            for mid in model_ids if mid in metrics
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
