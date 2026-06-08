"""
Demand Forecasting — LightGBM Challenger
Streamlit app: upload data → train → evaluate → view report
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, io, tempfile, traceback
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.block-container { padding-top: 2rem; }

.metric-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; }
.metric-card .value { color: #38bdf8; font-family: 'IBM Plex Mono'; font-size: 2rem; font-weight: 600; }
.metric-card .sub   { color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem; }

.status-box {
    border-radius: 6px; padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono'; font-size: 0.85rem;
}
.status-win  { background:#052e16; border-left:4px solid #22c55e; color:#86efac; }
.status-loss { background:#1c1917; border-left:4px solid #f97316; color:#fed7aa; }
.status-info { background:#0c1445; border-left:4px solid #38bdf8; color:#bae6fd; }

.upload-hint { color:#64748b; font-size:0.8rem; margin-top:-0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants (must match training script) ────────────────────────────────────
WIN_THRESHOLD = 0.10
CAT = ["family", "material", "stock_type", "dow", "month"]
FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_60","lag_90",
            "rmean_7","rstd_7","rmean_30","rstd_30","rmean_90","rstd_90",
            "dow","month","is_month_start","is_month_end","days_since_issue",
            "family","material","stock_type","reorder_qty","reorder_ratio"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2,
              n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbosity=-1)

# ── Helper functions ──────────────────────────────────────────────────────────
def _prep(df):
    df = df.sort_values(["item","date"]).copy()
    for c in CAT:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def train_one(df, target_col, grain, progress_bar, status_text, offset, span):
    df = _prep(df).dropna(subset=[target_col])
    X, y = df[FEATURES], df[target_col]
    cats = [c for c in CAT if c in X.columns]
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mae = []
    for i, (tr, va) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=cats)
        cv_mae.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va])))
        progress_bar.progress(offset + span * (i + 1) / 5)
        status_text.text(f"Training {grain} | horizon {target_col} — CV fold {i+1}/3")
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y, categorical_feature=cats)
    return model, float(np.mean(cv_mae))

def forward_eval(df, target_col, model):
    df = _prep(df)
    cats = [c for c in CAT if c in df.columns]
    rows = []
    for item, g in df.groupby("item", observed=True):
        g = g.dropna(subset=[target_col])
        if len(g) < 5:
            continue
        origin = g.iloc[[-1]][FEATURES].copy()
        for c in cats:
            origin[c] = origin[c].astype(df[c].dtype)
        pred   = float(model.predict(origin)[0])
        actual = float(g.iloc[-1][target_col])
        rows.append({"item": item, "ml_pred": round(pred,2),
                     "actual": actual, "ml_mae": abs(pred - actual)})
    return pd.DataFrame(rows)

def run_training(daily, weekly, base):
    results = []
    prog = st.progress(0)
    txt  = st.empty()
    tasks = [
        (daily,  "target_30", "daily",  30,  0.00, 0.25),
        (daily,  "target_90", "daily",  90,  0.25, 0.25),
        (weekly, "target_4",  "weekly", 30,  0.50, 0.25),
        (weekly, "target_13", "weekly", 90,  0.75, 0.25),
    ]
    models = {}
    for df, tcol, grain, hd, off, span in tasks:
        model, cv = train_one(df, tcol, grain, prog, txt, off, span)
        ev = forward_eval(df, tcol, model)
        ev = ev.assign(grain=grain, horizon=hd, cv_mae=round(cv,1))
        results.append(ev)
        models[f"{grain}_h{hd}"] = model

    prog.progress(1.0)
    txt.text("✅ Training complete!")

    allres = pd.concat(results, ignore_index=True).merge(
        base[["item","best_baseline","baseline_to_beat"]], on="item", how="left")
    allres["improvement"] = (
        (allres["baseline_to_beat"] - allres["ml_mae"]) / allres["baseline_to_beat"]
    )
    allres["ml_wins"] = allres["improvement"] >= WIN_THRESHOLD
    return allres, models

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Demand Forecasting AI")
    st.markdown("**Phase 4 — LightGBM Challenger**")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Upload your 3 data files
2. Click **Train & Evaluate**
3. Review results — AI only ships where it beats your baseline by ≥ 10%

**Files needed:**
- `train_daily.parquet`
- `train_weekly.parquet`
- `baseline_to_beat.parquet`
""")
    st.markdown("---")
    st.markdown('<span style="color:#475569;font-size:0.75rem">Win threshold: 10% MAE improvement</span>', unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 📦 Demand Forecasting AI")
st.markdown("##### LightGBM vs Baseline — which method wins for each item?")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["  📁 Upload & Train  ", "  📊 Results  ", "  📋 Report  "])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Train
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Step 1 — Upload your data files")
    st.markdown('<p class="upload-hint">All three files are required before training can begin.</p>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Daily features**")
        f_daily = st.file_uploader("train_daily.parquet", type=["parquet"], key="daily")
    with col2:
        st.markdown("**Weekly features**")
        f_weekly = st.file_uploader("train_weekly.parquet", type=["parquet"], key="weekly")
    with col3:
        st.markdown("**Baseline benchmarks**")
        f_base = st.file_uploader("baseline_to_beat.parquet", type=["parquet"], key="base")

    st.markdown("---")
    st.markdown("### Step 2 — Train & Evaluate")

    all_ready = f_daily and f_weekly and f_base

    if not all_ready:
        missing = []
        if not f_daily:  missing.append("train_daily.parquet")
        if not f_weekly: missing.append("train_weekly.parquet")
        if not f_base:   missing.append("baseline_to_beat.parquet")
        st.markdown(f'<div class="status-box status-info">⏳ Waiting for: {", ".join(missing)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-win">✅ All files uploaded — ready to train!</div>', unsafe_allow_html=True)
        st.markdown("")

        if st.button("🚀 Train & Evaluate", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading files…"):
                    daily  = pd.read_parquet(io.BytesIO(f_daily.read()))
                    weekly = pd.read_parquet(io.BytesIO(f_weekly.read()))
                    base   = pd.read_parquet(io.BytesIO(f_base.read()))

                allres, models = run_training(daily, weekly, base)
                st.session_state["allres"]  = allres
                st.session_state["models"]  = models
                st.success("Training finished! Go to the **Results** tab to explore.")

            except Exception as e:
                st.error(f"Something went wrong during training.")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Results
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if "allres" not in st.session_state:
        st.markdown('<div class="status-box status-info">ℹ️ No results yet — please upload files and run training first.</div>', unsafe_allow_html=True)
    else:
        allres = st.session_state["allres"]

        wins       = int(allres["ml_wins"].sum())
        total      = len(allres)
        items_win  = allres[allres.ml_wins]["item"].nunique()
        items_tot  = allres["item"].nunique()
        avg_improv = allres[allres.ml_wins]["improvement"].mean() * 100

        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, str(wins),            "ML wins",       f"out of {total} cells"),
            (c2, str(items_win),        "Items → AI",    f"of {items_tot} total items"),
            (c3, f"{avg_improv:.1f}%",  "Avg improvement", "where ML wins"),
            (c4, str(total - wins),     "Keep baseline", "cells"),
        ]
        for col, val, label, sub in metrics:
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### Results by grain & horizon")

        summary = (allres.groupby(["grain","horizon"])["ml_wins"]
                   .agg(wins="sum", total="size").reset_index())
        summary["win_rate"] = (summary["wins"] / summary["total"] * 100).round(1).astype(str) + "%"
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Filter detailed results")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show = st.radio("Show", ["All", "ML wins only", "Baseline wins only"], horizontal=True)
        with col_f2:
            grains = ["All"] + sorted(allres["grain"].unique().tolist())
            grain_filter = st.selectbox("Grain", grains)

        display = allres.copy()
        if show == "ML wins only":      display = display[display.ml_wins]
        elif show == "Baseline wins only": display = display[~display.ml_wins]
        if grain_filter != "All":       display = display[display.grain == grain_filter]

        display = display.rename(columns={
            "ml_mae":"ML MAE","baseline_to_beat":"Baseline MAE",
            "improvement":"Improvement","ml_wins":"ML Wins",
            "grain":"Grain","horizon":"Horizon (days)"
        })
        display["Improvement"] = (display["Improvement"] * 100).round(1).astype(str) + "%"

        st.dataframe(
            display[["item","Grain","Horizon (days)","ML MAE",
                      "Baseline MAE","Improvement","ML Wins","cv_mae"]],
            use_container_width=True, hide_index=True
        )

        csv = allres.to_csv(index=False).encode()
        st.download_button("⬇️ Download full results CSV", csv,
                           "phase4_eval.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Report
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if "allres" not in st.session_state:
        st.markdown('<div class="status-box status-info">ℹ️ No results yet — please upload files and run training first.</div>', unsafe_allow_html=True)
    else:
        allres = st.session_state["allres"]
        wins   = allres[allres.ml_wins]

        report = f"""# Phase 4 — LightGBM Challenger Results

## Overview
- **Candidates:** {allres['item'].nunique()} dense Tier-A items
- **Win bar:** ML must beat each item's baseline by ≥ {int(WIN_THRESHOLD*100)}% MAE
- **ML wins:** {allres['ml_wins'].sum()} of {len(allres)} (item × grain × horizon) cells

## Breakdown by Grain & Horizon

{allres.groupby(['grain','horizon'])['ml_wins'].agg(wins='sum', total='size').reset_index().to_markdown(index=False)}

## Items Where ML Ships (sorted by improvement)

{wins[['item','grain','horizon','ml_mae','baseline_to_beat','best_baseline','improvement']].sort_values('improvement',ascending=False).round(3).to_markdown(index=False) if len(wins) else '_None — baselines win everywhere._'}

## Verdict

Ship ML **only** for the rows listed above.  
Every other item keeps its Phase-3 baseline (MA-90 / MA-7 / Croston).  
This honours the principle: **baselines are the floor**.
"""

        st.markdown(report)
        st.download_button("⬇️ Download report (.md)", report.encode(),
                           "phase4_report.md", "text/markdown")
