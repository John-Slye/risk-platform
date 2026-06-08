"""Portfolio credit risk: Vasicek + Gaussian/t copula loss distribution."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Portfolio Credit Risk - Credit VaR & Economic Capital")

st.markdown(
    """
Move from loan-by-loan expected loss to a full **portfolio loss
distribution**. Computes Credit VaR (the loss exceeded with probability
alpha) and Economic Capital (Credit VaR - Expected Loss) - the unexpected-
loss reserve a bank holds equity against.

**Method:** Vasicek single-factor (ASRF) model - the analytical foundation
of Basel IRB - paired with Gaussian / Student-t copula Monte Carlo for the
finite-sample loss distribution. 100k+ simulated portfolio paths per run.

**Input:** set the parametric inputs (PD, LGD, asset correlation, EAD,
number of obligors, copula choice). For 2014-2018 LendingClub-style
sub-prime parameters: PD ~20%, LGD ~92%, rho ~0.10.

**Try the comparison:** run once with Gaussian, then again with Student-t
(df=5). The 99.9% VaR and Economic Capital typically rise 15-30% under the
t-copula. That gap is the post-2008 modeling story in one chart.

For heterogeneous portfolios (one PD per obligor from a CSV), use the
Portfolio Upload page.
"""
)

st.sidebar.subheader("Portfolio source")
mode = st.sidebar.radio(
    "Data",
    ["Homogeneous (parameters)", "Upload obligor CSV (pd,lgd,ead)"],
    label_visibility="collapsed",
)

obligors = None
if mode.startswith("Upload"):
    upload = st.sidebar.file_uploader(
        "CSV with `pd`, `lgd`, `ead` columns", type=["csv"],
        help="One row per obligor. PDs must be in (0, 1); LGDs in [0, 1]; "
             "EADs in USD."
    )
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            missing = {"pd", "lgd", "ead"} - set(df.columns)
            if missing:
                st.sidebar.error(f"CSV missing columns: {sorted(missing)}")
            else:
                df = df.dropna(subset=["pd", "lgd", "ead"])
                obligors = df[["pd", "lgd", "ead"]].to_dict(orient="records")
                st.sidebar.success(f"{len(obligors)} obligors loaded.")
        except Exception as exc:
            st.sidebar.error(f"Failed to parse CSV: {exc}")
    else:
        st.sidebar.info("Awaiting CSV. Falls back to homogeneous parameters "
                        "until you upload.")

with st.form("port_form"):
    c1, c2, c3 = st.columns(3)
    pd_rate     = c1.number_input("Avg PD",            min_value=0.001, max_value=0.5, value=0.20, step=0.01)
    rho         = c2.number_input("Asset correlation", min_value=0.01,  max_value=0.99, value=0.10, step=0.01)
    n_obligors  = c3.number_input("Number of obligors", min_value=10,   max_value=100_000, value=1000)
    c4, c5, c6 = st.columns(3)
    lgd         = c4.number_input("LGD",            min_value=0.0,   max_value=1.0, value=0.92, step=0.05)
    ead         = c5.number_input("EAD per obligor ($)", min_value=100, value=15_000)
    n_sims      = c6.slider("MC simulations", 1_000, 100_000, 20_000, step=1_000)
    copula      = st.radio("Copula", ["gaussian", "t"], horizontal=True, index=0)
    df          = st.slider("t-copula degrees of freedom", 3, 30, 5,
                            disabled=(copula == "gaussian"))
    submitted = st.form_submit_button("Simulate portfolio")

if submitted:
    body = dict(pd=pd_rate, rho=rho, n_obligors=n_obligors, lgd=lgd, ead=ead,
                copula=copula, df=df, n_simulations=n_sims)
    if obligors:
        body["obligors"] = obligors
        st.info(f"Using {len(obligors)} uploaded obligors; the homogeneous "
                f"PD/LGD/EAD/n_obligors above are ignored.")
    try:
        r = requests.post(f"{API_URL}/portfolio/credit_var", json=body, timeout=30)
        r.raise_for_status()
        out = r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Loss",       f"${out['expected_loss']:,.0f}")
    c2.metric("Credit VaR (99.9%)",  f"${out['credit_var_99_9']:,.0f}")
    c3.metric("Economic Capital",    f"${out['economic_capital']:,.0f}")
    c4, c5 = st.columns(2)
    c4.metric("Credit VaR (99%)",    f"${out['credit_var_99']:,.0f}")
    if out.get("tail_es_99_9") is not None:
        c5.metric("Tail ES (99.9%)", f"${out['tail_es_99_9']:,.0f}")
    st.caption(f"Copula: **{out['copula']}**  ·  df: {out['df']}  ·  sims: {out['n_simulations']:,}")

    # Loss distribution histogram (from API histogram_bins + histogram_counts)
    if out.get("histogram_bins") and out.get("histogram_counts"):
        edges = np.array(out["histogram_bins"])
        counts = np.array(out["histogram_counts"])
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure()
        fig.add_bar(x=centers, y=counts,
                    marker_color="crimson" if copula == "t" else "steelblue",
                    name=f"Loss distribution ({copula})")
        fig.add_vline(x=out["expected_loss"], line_dash="dash", line_color="black",
                      annotation_text=f"EL ${out['expected_loss']:,.0f}")
        fig.add_vline(x=out["credit_var_99"], line_dash="dot", line_color="darkorange",
                      annotation_text="99%")
        fig.add_vline(x=out["credit_var_99_9"], line_dash="dot", line_color="red",
                      annotation_text="99.9%")
        fig.update_layout(
            title="Simulated portfolio loss distribution",
            xaxis_title="Portfolio loss ($)", yaxis_title="Sim count",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why does Gaussian vs t-copula matter?"):
        st.markdown("""
The 2008 mortgage crisis revealed that the Gaussian copula assumption
(implicit in Vasicek and Basel IRB) underestimates joint extreme defaults.
The Student-t copula has fatter joint tails: when one obligor defaults badly,
others are more likely to default with it.

Run this page twice with the same inputs but Gaussian vs t (df=5). The
99.9% VaR and Economic Capital typically rise 15-30% under t - that gap is
roughly the model-risk severity the post-2008 regulatory framework was built
to address.
""")
