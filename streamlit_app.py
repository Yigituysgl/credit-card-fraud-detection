"""Transaction replay demo.

Steps through a fixed sample of held-out test transactions (data/demo_transactions.csv,
written by train.py) and scores each one with the saved pipeline and its
validation-tuned threshold. This is a replay of historical data, not a live stream.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "models" / "fraud_pipeline.joblib"
DEMO_PATH = ROOT / "data" / "demo_transactions.csv"


@st.cache_resource
def load_bundle():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_PATH)


st.title("💳 Fraud Detection – Transaction Replay Demo")
st.write(
    "Replays held-out test transactions one at a time and scores them with the saved "
    "pipeline. Historical data only; no live stream."
)

if not PIPELINE_PATH.exists() or not DEMO_PATH.exists():
    st.error("Model or demo data missing. Run `python train.py` first (see README).")
    st.stop()

bundle = load_bundle()
pipeline, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]
df = load_demo()

for key in ("index", "tp", "fp", "fn", "tn"):
    st.session_state.setdefault(key, 0)

st.caption(
    f"Model: {bundle['model']} + {bundle['balancing']} · "
    f"decision threshold {threshold:.4f} (tuned on the validation split)"
)

if st.button("Next transaction"):
    i = st.session_state.index
    if i < len(df):
        row = df.iloc[[i]]
        risk = float(pipeline.predict_proba(row[features])[0, 1])
        flagged = risk >= threshold
        actual = int(row["Class"].iloc[0])

        outcome = {(True, 1): "tp", (True, 0): "fp", (False, 1): "fn", (False, 0): "tn"}
        st.session_state[outcome[(flagged, actual)]] += 1
        st.session_state.index = i + 1

        st.write(f"### Transaction {i + 1} of {len(df)}")
        st.dataframe(row[features], hide_index=True)
        if flagged:
            st.error(f"🚨 Flagged as fraud (score {risk:.4f})")
        else:
            st.success(f"✅ Not flagged (score {risk:.4f})")
        st.caption(f"Actual label: {'fraud' if actual else 'normal'}")
    else:
        st.info("All demo transactions have been replayed.")

st.subheader("Running tally")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraud caught (TP)", st.session_state.tp)
c2.metric("False alarms (FP)", st.session_state.fp)
c3.metric("Fraud missed (FN)", st.session_state.fn)
c4.metric("Normal passed (TN)", st.session_state.tn)
st.progress(st.session_state.index / len(df))
