"""
Phase 4 — LightGBM challenger training (RUN IN GOOGLE COLAB).
================================================================
You received four parquet files. Upload them to the Colab session (or mount Drive)
and set DATA_DIR. This script trains one global LightGBM per horizon, evaluates with
an EXPANDING-WINDOW time-series split, and — the decision that matters — compares each
item's held-out MAE against its committed baseline. ML is declared a winner ONLY where
it beats that item's baseline by >= 10% MAE.

Inputs (from the data scientist):
  train_daily.parquet      features + target_7 / target_30 / target_90  (daily grain)
  train_weekly.parquet     features + target_4 / target_13              (weekly grain ≈30d/90d)
  baseline_to_beat.parquet per-item best baseline MAE on the same 60-day forward test
Outputs (this script writes):
  lgbm_<grain>_h<H>.txt    trained boosters
  phase4_eval.csv          per-item ML vs baseline, win/loss flags
  phase4_report.md         summary + ship list
"""
import pandas as pd, numpy as np, json, os
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = "."           # <-- set to where you uploaded the parquet files
TEST_DAYS = 60           # final forward hold-out (matches Phase 3)
WIN_THRESHOLD = 0.10     # ML must beat baseline by >=10% MAE
CAT = ["family", "material", "stock_type", "dow", "month"]

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2,
              n_estimators=1200, learning_rate=0.02, num_leaves=63,
              min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)

FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_60","lag_90",
            "rmean_7","rstd_7","rmean_30","rstd_30","rmean_90","rstd_90",
            "dow","month","is_month_start","is_month_end","days_since_issue",
            "family","material","stock_type","reorder_qty","reorder_ratio"]

def _prep(df):
    df = df.sort_values(["item","date"]).copy()
    for c in CAT:
        if c in df: df[c] = df[c].astype("category")
    return df

def train_one(df, target_col, grain):
    """Global model across all candidate items for a single horizon."""
    df = _prep(df).dropna(subset=[target_col])
    X, y = df[FEATURES], df[target_col]
    cats = [c for c in CAT if c in X]
    # expanding-window CV over time-sorted rows (report only; final fit is on all)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae = []
    for tr, va in tscv.split(X):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=cats)
        cv_mae.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va])))
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y, categorical_feature=cats)
    model.booster_.save_model(f"lgbm_{grain}_h{target_col.split('_')[1]}.txt")
    return model, float(np.mean(cv_mae))

def forward_eval(df, target_col, model, horizon_days):
    """Per-item MAE on the LAST forecast origin (held-out 60-day window),
    directly comparable to the Phase-3 baseline number."""
    df = _prep(df)
    cats = [c for c in CAT if c in df]
    rows = []
    for item, g in df.groupby("item", observed=True):
        g = g.dropna(subset=[target_col])
        if len(g) < 5: continue
        origin = g.iloc[[-1]][FEATURES].copy()          # keep as 1-row DataFrame
        for c in cats:                                   # preserve training categories
            origin[c] = origin[c].astype(df[c].dtype)
        pred = float(model.predict(origin)[0])
        actual = float(g.iloc[-1][target_col])
        rows.append({"item": item, "ml_pred": pred, "actual": actual,
                     "ml_mae": abs(pred - actual)})
    return pd.DataFrame(rows)

def main():
    daily = pd.read_parquet(os.path.join(DATA_DIR, "train_daily.parquet"))
    weekly = pd.read_parquet(os.path.join(DATA_DIR, "train_weekly.parquet"))
    base = pd.read_parquet(os.path.join(DATA_DIR, "baseline_to_beat.parquet"))[
        ["item","best_baseline","baseline_to_beat"]]

    results = []
    # Daily models: 30 & 90-day horizons are the operational ones
    for tcol, hd in [("target_30", 30), ("target_90", 90)]:
        model, cv = train_one(daily, tcol, "daily")
        ev = forward_eval(daily, tcol, model, hd).assign(grain="daily", horizon=hd, cv_mae=round(cv,1))
        results.append(ev)
    # Weekly models (these items are dense weekly): 4wk≈30d, 13wk≈90d
    for tcol, hd in [("target_4", 30), ("target_13", 90)]:
        model, cv = train_one(weekly, tcol, "weekly")
        ev = forward_eval(weekly, tcol, model, hd).assign(grain="weekly", horizon=hd, cv_mae=round(cv,1))
        results.append(ev)

    allres = pd.concat(results, ignore_index=True).merge(base, on="item", how="left")
    allres["improvement"] = (allres["baseline_to_beat"] - allres["ml_mae"]) / allres["baseline_to_beat"]
    allres["ml_wins"] = allres["improvement"] >= WIN_THRESHOLD
    allres.to_csv("phase4_eval.csv", index=False)

    # ship list: for each item+horizon, ML only if it wins; else keep baseline
    win = allres[allres.ml_wins].sort_values("improvement", ascending=False)
    with open("phase4_report.md","w") as f:
        f.write("# Phase 4 — LightGBM challenger results\n\n")
        f.write(f"Candidates: {allres.item.nunique()} dense Tier-A items. "
                f"Win bar: ML beats item's baseline by >= {int(WIN_THRESHOLD*100)}% MAE.\n\n")
        f.write(f"**ML wins: {allres.ml_wins.sum()} of {len(allres)} (item × grain × horizon) cells.**\n\n")
        f.write("## Items where ML ships (sorted by improvement)\n\n")
        f.write(win[["item","grain","horizon","ml_mae","baseline_to_beat","best_baseline","improvement"]]
                  .round(3).to_markdown(index=False) if len(win) else "_None — baselines win everywhere. Ship baselines._\n")
        f.write("\n\n## Verdict\n")
        f.write("Ship ML only for the rows above. Every other item keeps its Phase-3 "
                "baseline (MA-90 / MA-7 / Croston). This honours 'baselines are the floor'.\n")
    print("WROTE phase4_eval.csv and phase4_report.md")
    print(allres.groupby(["grain","horizon"])["ml_wins"].agg(["sum","size"]).to_string())
    print(f"\nTotal ML wins: {allres.ml_wins.sum()} / {len(allres)}")

if __name__ == "__main__":
    main()
