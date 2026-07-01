import json
import logging
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from dataset_prep import clean, eng_features, ENG_COLS

MODEL_DIR = "best_model"
CONF_FLOOR = 0.50

KW_OVERRIDE = [
    ("is_flat_kw", "flat"),
    ("is_comm_kw", "commercial unit"),
    ("is_land_kw", "landparcel"),
]

logging.basicConfig(filename="override_audit.log", level=logging.INFO,
                     format="%(asctime)s %(message)s")
logger = logging.getLogger("override_audit")

st.set_page_config(page_title="Property Address Classifier", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/word_vec.pkl", "rb") as f:
        word_vec = pickle.load(f)
    with open(f"{MODEL_DIR}/char_vec.pkl", "rb") as f:
        char_vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(f"{MODEL_DIR}/meta.json") as f:
        meta = json.load(f)
    return model, word_vec, char_vec, scaler, le, meta


def predict(address, model, word_vec, char_vec, scaler, le):
    t = clean(address)
    feats = eng_features(pd.Series([address]))
    eng = feats.values.astype(float)

    Xw = word_vec.transform([t])
    Xc = char_vec.transform([t])
    Xe = sp.csr_matrix(scaler.transform(eng))
    X = sp.hstack([Xw, Xc, Xe]).tocsr()

    proba = model.predict_proba(X)[0]
    class_proba = dict(zip(le.classes_, proba.tolist()))

    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    label = le.classes_[idx] if conf >= CONF_FLOOR else "others"

    override_label = None
    if label == "others":
        for col, candidate in KW_OVERRIDE:
            if feats[col].iloc[0] == 1:
                override_label = candidate
                break

    if override_label:
        logger.info("OVERRIDE address=%r model_label=%r model_conf=%.4f -> override_label=%r",
                     address, label, conf, override_label)
        label = override_label
        conf = class_proba[label]

    return label, conf, class_proba


st.title("🏠 Property Address Classifier")
st.caption("Classifies a raw Indian property address into: flat · houseorplot · landparcel · commercial unit · others")

try:
    model, word_vec, char_vec, scaler, le, meta = load_model()
    st.success(f"Model: **{meta['model_type'].upper()}**  |  Test Macro F1: **{meta['macro_f1']:.4f}**")
except FileNotFoundError:
    st.error("No artifacts found in best_model/. Run `python train.py` first.")
    st.stop()

address = st.text_area(
    "Enter a property address",
    placeholder="e.g. Flat 3B, Tower C, Green Society, Boisar 401501",
    height=100,
)

if st.button("Classify", type="primary") and address.strip():
    label, conf, all_probs = predict(address, model, word_vec, char_vec, scaler, le)

    c1, c2 = st.columns(2)
    c1.metric("Predicted Category", label)
    c2.metric("Confidence", f"{conf * 100:.1f}%")

    prob_df = (
        pd.DataFrame({"category": list(all_probs.keys()), "probability": list(all_probs.values())})
        .sort_values("probability", ascending=False)
    )
    st.bar_chart(prob_df.set_index("category"))

elif not address.strip():
    st.info("Enter an address above and click Classify.")
