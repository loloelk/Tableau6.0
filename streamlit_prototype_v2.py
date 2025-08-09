"""
Streamlit Prototype v2.1 — Patient Dashboard (working + improved)
-----------------------------------------------------------------
How to run locally:
  1) pip install -r requirements.txt  (or: pip install streamlit pandas plotly pyvis networkx pyyaml)
  2) streamlit run streamlit_prototype_v2.py

Notes:
- Uses ONLY standard modules + Streamlit; no Pydantic.
- Fixes cache errors by caching ONLY module-level functions with hashable args.
- Clean 4-page UI: Overview • Patient • Protocols • Nurse notes.
- Simulated CSVs in ./data are used if present; otherwise auto-generates demo data.
- REDCap-ready: swap CsvDataSource with a future RedcapDataSource returning same DataFrames.

Requirements (add to requirements.txt if missing):
  streamlit
  pandas
  plotly
  pyvis
  networkx
  pyyaml
"""

from __future__ import annotations
import os
from typing import Optional, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import networkx as nx
from pyvis.network import Network

# -------------------------------
# Config & constants
# -------------------------------
DATA_DIR = os.environ.get("DASH_DATA_DIR", "data")
DEFAULT_PATIENT_CSV = os.path.join(DATA_DIR, "patient_data_with_protocol_simulated.csv")
DEFAULT_EMA_CSV = os.path.join(DATA_DIR, "simulated_ema_data.csv")
DEFAULT_NURSE_CSV = os.path.join(DATA_DIR, "nurse_inputs.csv")

MADRS_ITEMS = [
    "apparent_sadness", "reported_sadness", "inner_tension", "reduced_sleep",
    "reduced_appetite", "concentration_difficulty", "lassitude",
    "inability_to_feel", "pessimistic_thoughts", "suicidal_thoughts"
]

PROTOCOL_ORDER = ["HF-10Hz", "iTBS", "BR-18Hz", "Sham"]

# -------------------------------
# Cached CSV loaders (hashable args only)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csv_patients(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "patient_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "patient_id"})
    return df

@st.cache_data(show_spinner=False)
def load_csv_ema(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ("date", "timestamp", "day"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

@st.cache_data(show_spinner=False)
def load_csv_nurse(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ("date", "timestamp"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

# -------------------------------
# Data source abstraction
# -------------------------------
class AbstractDataSource:
    def get_patients(self) -> pd.DataFrame:
        raise NotImplementedError
    def get_ema(self) -> pd.DataFrame:
        raise NotImplementedError
    def get_nurse(self) -> pd.DataFrame:
        raise NotImplementedError

class CsvDataSource(AbstractDataSource):
    def __init__(self, patient_csv: str, ema_csv: str, nurse_csv: str):
        self.patient_csv = patient_csv
        self.ema_csv = ema_csv
        self.nurse_csv = nurse_csv

    def get_patients(self) -> pd.DataFrame:
        df = load_csv_patients(self.patient_csv)
        if df is None:
            return self._make_toy_patients()
        return df

    def get_ema(self) -> pd.DataFrame:
        df = load_csv_ema(self.ema_csv)
        if df is None:
            return self._make_toy_ema()
        return df

    def get_nurse(self) -> pd.DataFrame:
        df = load_csv_nurse(self.nurse_csv)
        if df is None:
            return self._make_toy_nurse()
        return df

    # --- toy data fallbacks for demo environments ---
    def _make_toy_patients(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 25
        df = pd.DataFrame({
            "patient_id": [f"P{1000+i}" for i in range(n)],
            "sex": rng.choice(["F", "M"], size=n),
            "age": rng.integers(18, 75, size=n),
            "protocol": rng.choice(PROTOCOL_ORDER, size=n, p=[0.4,0.4,0.15,0.05]),
            "madrs_baseline": rng.normal(32, 7, size=n).clip(10, 50).round(0),
            "madrs_followup": rng.normal(18, 8, size=n).clip(0, 50).round(0),
            "phq9_baseline": rng.normal(17, 4, size=n).clip(4, 27).round(0),
            "phq9_followup": rng.normal(10, 5, size=n).clip(0, 27).round(0)
        })
        return df

    def _make_toy_ema(self) -> pd.DataFrame:
        rng = np.random.default_rng(123)
        patients = [f"P{1000+i}" for i in range(25)]
        days = pd.date_range(pd.Timestamp.today().date() - pd.Timedelta(days=60), periods=60)
        rows: List[dict] = []
        for pid in patients:
            base = rng.normal(3.5, 0.7)
            for d in days:
                item_scores = {k: float(np.clip(rng.normal(base, 0.5), 0, 6)) for k in MADRS_ITEMS}
                rows.append({
                    "patient_id": pid,
                    "date": d,
                    **item_scores,
                    "anxiety": float(np.clip(rng.normal(2.5,0.7),0,5)),
                    "sleep": float(np.clip(rng.normal(2.5,0.8),0,5)),
                })
        return pd.DataFrame(rows)

    def _make_toy_nurse(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        patients = [f"P{1000+i}" for i in range(25)]
        rows: List[dict] = []
        for pid in patients:
            for i in range(5):
                rows.append({
                    "patient_id": pid,
                    "date": pd.Timestamp.today().normalize() - pd.Timedelta(days=int(i*7)),
                    "objective": f"Increase BA by {10+i*5}%",
                    "task": f"Walk {15+5*i} mins daily",
                    "comment": rng.choice(["Doing exposures", "No side effects", "Mild headache", "Concern re: sleep"]) 
                })
        return pd.DataFrame(rows)

# -------------------------------
# Analytics helpers (no caching on methods)
# -------------------------------
class AnalyticsService:
    def __init__(self, src: AbstractDataSource):
        self.src = src

    def patient_summary(self) -> pd.DataFrame:
        df = self.src.get_patients().copy()
        if not df.empty:
            df["madrs_delta"] = df.get("madrs_baseline", np.nan) - df.get("madrs_followup", np.nan)
            df["phq9_delta"] = df.get("phq9_baseline", np.nan) - df.get("phq9_followup", np.nan)
        return df

    def protocol_effectiveness(self) -> pd.DataFrame:
        df = self.patient_summary()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("protocol")[ ["madrs_delta", "phq9_delta"] ]
              .agg(["mean", "median", "count"])  # simple for demo
              .reset_index()
        )

    def ema_for_patient(self, patient_id: str) -> pd.DataFrame:
        ema = self.src.get_ema()
        sub = ema[ema["patient_id"] == patient_id].sort_values("date")
        if not sub.empty:
            sub["madrs_proxy"] = sub[MADRS_ITEMS].sum(axis=1)
        return sub

# -------------------------------
# Visualization helpers
# -------------------------------
def build_symptom_network(ema_sub: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    if ema_sub is None or ema_sub.empty:
        return G
    corr = ema_sub[MADRS_ITEMS].corr()
    for col in corr.columns:
        G.add_node(col)
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if j <= i:
                continue
            w = corr.loc[a, b]
            if abs(w) >= 0.3:  # threshold for visibility
                G.add_edge(a, b, weight=float(w))
    return G

# no caching here; NetworkX graph is unhashable for Streamlit

def render_pyvis_graph(G: nx.Graph) -> str:
    net = Network(height="500px", width="100%", notebook=False, directed=False)
    for n in G.nodes:
        net.add_node(n, label=n)
    for a, b, data in G.edges(data=True):
        w = data.get("weight", 0.1)
        net.add_edge(a, b, value=abs(w))
    return net.generate_html(notebook=False)

# -------------------------------
# UI — app
# -------------------------------
st.set_page_config(
    page_title="TMS Patient Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Patient", "Protocols", "Nurse notes"], index=0)

    st.markdown("---")
    st.caption("Data source (CSV paths)")
    patient_csv = st.text_input("Patients CSV", DEFAULT_PATIENT_CSV)
    ema_csv = st.text_input("EMA CSV", DEFAULT_EMA_CSV)
    nurse_csv = st.text_input("Nurse CSV", DEFAULT_NURSE_CSV)

# Instantiate data/analytics services
source = CsvDataSource(patient_csv=patient_csv, ema_csv=ema_csv, nurse_csv=nurse_csv)
svc = AnalyticsService(source)

if page == "Overview":
    st.title("Overview")
    df = svc.patient_summary()

    if df.empty:
        st.info("No patient data available.")
    else:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.subheader("Baseline vs Follow-up (MADRS)")
            melted = df.melt(id_vars=["patient_id", "protocol"],
                             value_vars=["madrs_baseline", "madrs_followup"],
                             var_name="timepoint", value_name="madrs")
            fig = px.box(melted, x="timepoint", y="madrs", points="all")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Patients", len(df))
            st.metric("Protocols", int(df["protocol"].nunique()))
        with c3:
            st.metric("Median MADRS Δ", float(np.nanmedian(df["madrs_delta"])) if "madrs_delta" in df else np.nan)
            st.metric("Median PHQ-9 Δ", float(np.nanmedian(df["phq9_delta"])) if "phq9_delta" in df else np.nan)

        st.subheader("Protocol effectiveness (Δ scores)")
        proto = svc.protocol_effectiveness()
        if not proto.empty:
            proto.columns = ["_".join(col).strip("_") for col in proto.columns.values]
            fig2 = px.bar(proto, x="protocol_", y="madrs_delta_mean", hover_data=proto.columns)
            fig2.update_layout(xaxis_title="Protocol", yaxis_title="Mean MADRS Δ (baseline - followup)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Not enough data to compute effectiveness.")

elif page == "Patient":
    st.title("Patient detail")
    df = svc.patient_summary()
    if df.empty:
        st.info("No patient data available.")
    else:
        colA, colB = st.columns([2, 1])
        with colB:
            patient_id = st.selectbox("Patient", df["patient_id"].sort_values().tolist())
            show_network = st.checkbox("Show symptom network", value=True)
        with colA:
            row = df[df["patient_id"] == patient_id].iloc[0]
            st.markdown(f"**Protocol:** {row.get('protocol', 'NA')}  |  **Age:** {row.get('age', 'NA')}  |  **Sex:** {row.get('sex', 'NA')}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MADRS baseline", row.get("madrs_baseline", np.nan))
            k2.metric("MADRS follow-up", row.get("madrs_followup", np.nan))
            k3.metric("PHQ-9 baseline", row.get("phq9_baseline", np.nan))
            k4.metric("PHQ-9 follow-up", row.get("phq9_followup", np.nan))

        ema = svc.ema_for_patient(patient_id)
        if ema.empty:
            st.info("No EMA data for this patient.")
        else:
            t1, t2 = st.tabs(["Time series", "Distribution"])
            with t1:
                fig = px.line(ema, x="date", y="madrs_proxy", title="EMA MADRS proxy over time")
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                figh = px.histogram(ema, x="madrs_proxy")
                st.plotly_chart(figh, use_container_width=True)

        if show_network:
            st.subheader("Symptom network (|r| ≥ 0.3)")
            G = build_symptom_network(ema)
            if len(G.nodes) == 0:
                st.caption("Insufficient data to build network.")
            else:
                html = render_pyvis_graph(G)
                st.components.v1.html(html, height=520, scrolling=True)

elif page == "Protocols":
    st.title("Protocol comparisons")
    df = svc.patient_summary()
    if df.empty:
        st.info("No patient data available.")
    else:
        with st.expander("Filters", expanded=True):
            protos = st.multiselect("Protocol(s)", PROTOCOL_ORDER, default=PROTOCOL_ORDER[:3])
        dfp = df[df["protocol"].isin(protos)] if protos else df

        fig = px.violin(dfp, x="protocol", y="madrs_delta", box=True, points="all")
        fig.update_layout(yaxis_title="MADRS Δ (baseline - followup)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Table")
        st.dataframe(dfp[["patient_id", "protocol", "madrs_baseline", "madrs_followup", "madrs_delta", "phq9_delta"]], use_container_width=True)

elif page == "Nurse notes":
    st.title("Nurse inputs & BA tracker")
    n = source.get_nurse().copy()
    patients = source.get_patients()["patient_id"].sort_values().tolist()

    with st.form("nurse_form"):
        c1, c2, c3 = st.columns(3)
        pid = c1.selectbox("Patient", patients)
        date = c2.date_input("Date", pd.Timestamp.today())
        objective = st.text_input("Objective (SMART)")
        task = st.text_input("Behavioral activation task")
        comment = st.text_area("Comment")
        submitted = st.form_submit_button("Save entry")

    if submitted:
        row = {"patient_id": pid, "date": pd.to_datetime(date), "objective": objective, "task": task, "comment": comment}
        n = pd.concat([n, pd.DataFrame([row])], ignore_index=True)
        # Persist to CSV in demo mode
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            out_path = DEFAULT_NURSE_CSV if nurse_csv == DEFAULT_NURSE_CSV else nurse_csv
            n.to_csv(out_path, index=False)
            st.success("Saved.")
        except Exception as e:
            st.warning(f"Could not save to CSV: {e}")

    st.subheader("Recent entries")
    st.dataframe(n.sort_values("date", ascending=False).head(50), use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Prototype v2.1 • Simulated data • Swap in REDCap by implementing RedcapDataSource(AbstractDataSource)")
