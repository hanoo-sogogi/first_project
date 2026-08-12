"""
IBM AML (HI-Small) baseline reproduction.

Follows the methodology described in "Realistic Synthetic Financial Transactions
for Anti-Money Laundering Models" (Altman et al., NeurIPS 2023 D&B):
  - Graph-derived account features (in/out degree, unique counterparties,
    repeat-pair history) in place of the paper's proprietary Graph Feature
    Preprocessor (GFP).
  - Temporal 60/20/20 train/val/test split.
  - GBT baselines (XGBoost, LightGBM) with minority-class F1 / PR-AUC / P@K,
    matching Table 2 / Appendix F of the paper.
  - All account-level aggregates are computed from the TRAIN period only to
    avoid temporal leakage (transactions from val/test never leak into a
    feature seen at train time or vice versa).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"C:\Users\aica_\Documents\CLaude")
OUT_DIR = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
DATASET = "HI-Small"
SEEDS = [1, 2, 3]

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
log("Loading transactions...")
df = pd.read_csv(DATA_DIR / f"{DATASET}_Trans.csv")
df.columns = [
    "Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount",
    "AmountReceived", "ReceivingCurrency", "AmountPaid", "PaymentCurrency",
    "PaymentFormat", "IsLaundering",
]
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
df = df.sort_values("Timestamp").reset_index(drop=True)
log(f"Loaded {len(df):,} transactions, {df['IsLaundering'].sum():,} laundering "
    f"({df['IsLaundering'].mean()*100:.4f}%)")

# Composite node keys: (bank, account) — account numbers repeat across banks.
df["FromNode"] = df["FromBankID"].astype(str) + "_" + df["FromAccount"]
df["ToNode"] = df["ToBankID"].astype(str) + "_" + df["ToAccount"]

# ---------------------------------------------------------------------------
# 2. Currency normalization -> USD
# ---------------------------------------------------------------------------
log("Deriving FX table from mismatched-currency transactions...")
fx = df[df["ReceivingCurrency"] != df["PaymentCurrency"]].copy()
fx["rate_recv_per_paid"] = fx["AmountReceived"] / fx["AmountPaid"]

# Build a directed median rate table (currency_paid -> currency_recv), then
# anchor everything to "US Dollar" via a short BFS since not all currencies
# trade directly against USD in the mismatched-currency subset.
pair_rate = (
    fx.groupby(["PaymentCurrency", "ReceivingCurrency"])["rate_recv_per_paid"]
    .median()
    .to_dict()
)
graph = {}
for (a, b), r in pair_rate.items():
    graph.setdefault(a, {})[b] = r
    graph.setdefault(b, {})[a] = 1.0 / r

usd_per = {"US Dollar": 1.0}
frontier = ["US Dollar"]
while frontier:
    nxt = []
    for cur in frontier:
        for nb, r in graph.get(cur, {}).items():
            if nb not in usd_per:
                # rate is recv_per_paid for (cur -> nb); we want USD per unit of nb
                usd_per[nb] = usd_per[cur] / r
                nxt.append(nb)
    frontier = nxt

all_currencies = set(df["PaymentCurrency"]) | set(df["ReceivingCurrency"])
missing = all_currencies - set(usd_per)
if missing:
    log(f"WARNING: no FX path for {missing}, defaulting to 1.0")
    for c in missing:
        usd_per[c] = 1.0

df["AmountUSD"] = df["AmountPaid"] * df["PaymentCurrency"].map(usd_per)
df["LogAmountUSD"] = np.log1p(df["AmountUSD"])
log(f"FX table resolved for {len(usd_per)} currencies")

# ---------------------------------------------------------------------------
# 3. Simple structural / temporal features (leak-free, row-local)
# ---------------------------------------------------------------------------
df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(int)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(int)
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["PaymentFormat"] = df["PaymentFormat"].astype("category")

# ---------------------------------------------------------------------------
# 4. Temporal 60/20/20 split
# ---------------------------------------------------------------------------
n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2
log(f"Split -> train {train_mask.sum():,} (pos {df.loc[train_mask,'IsLaundering'].sum():,}) | "
    f"val {val_mask.sum():,} (pos {df.loc[val_mask,'IsLaundering'].sum():,}) | "
    f"test {test_mask.sum():,} (pos {df.loc[test_mask,'IsLaundering'].sum():,})")

# ---------------------------------------------------------------------------
# 5. Graph aggregate features — computed from TRAIN period only
# ---------------------------------------------------------------------------
log("Building leak-free graph aggregate features from train period...")
train_df = df.loc[train_mask]

out_agg = train_df.groupby("FromNode").agg(
    out_degree=("ToNode", "count"),
    out_unique_cp=("ToNode", "nunique"),
    out_avg_amt=("LogAmountUSD", "mean"),
)
in_agg = train_df.groupby("ToNode").agg(
    in_degree=("FromNode", "count"),
    in_unique_cp=("FromNode", "nunique"),
    in_avg_amt=("LogAmountUSD", "mean"),
    in_unique_banks=("FromBankID", "nunique"),
)
pair_count = (
    train_df.groupby(["FromNode", "ToNode"]).size().rename("pair_prior_count")
)

df = df.merge(out_agg, left_on="FromNode", right_index=True, how="left")
df = df.merge(in_agg, left_on="ToNode", right_index=True, how="left")
df = df.merge(pair_count, on=["FromNode", "ToNode"], how="left")

agg_cols = [
    "out_degree", "out_unique_cp", "out_avg_amt",
    "in_degree", "in_unique_cp", "in_avg_amt", "in_unique_banks",
    "pair_prior_count",
]
for c in agg_cols:
    df[c] = df[c].fillna(0)

log("Graph features attached: " + ", ".join(agg_cols))

# ---------------------------------------------------------------------------
# 6. Assemble feature matrix
# ---------------------------------------------------------------------------
FEATURES = [
    "LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank",
    "Hour", "DayOfWeek",
] + agg_cols
LABEL = "IsLaundering"

X = df[FEATURES]
y = df[LABEL]

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

# Downsample negatives in the TRAIN split only (validated in prior EDA to cut
# training time drastically with no measurable effect on held-out metrics).
rng = np.random.RandomState(0)
pos_idx = y_train[y_train == 1].index
neg_idx = y_train[y_train == 0].index
neg_sample = rng.choice(neg_idx, size=min(400_000, len(neg_idx)), replace=False)
keep_idx = pos_idx.union(pd.Index(neg_sample))
X_train_ds = X_train.loc[keep_idx]
y_train_ds = y_train.loc[keep_idx]
log(f"Downsampled train set: {len(X_train_ds):,} rows "
    f"({y_train_ds.sum():,} positive)")

scale_pos_weight = (y_train_ds == 0).sum() / max((y_train_ds == 1).sum(), 1)
log(f"train pos:neg ratio after downsampling = 1:{scale_pos_weight:.1f} "
    f"(scale_pos_weight NOT applied on top of this - see note below)")


# ---------------------------------------------------------------------------
# 7. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(y_true, scores, name):
    preds = (scores >= 0.5).astype(int)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)

    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    p_at_k = {}
    for k in (100, 500, 1000, 2000):
        if len(y_sorted) >= k:
            p_at_k[k] = float(y_sorted[:k].sum() / k)

    p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
    recall_at_90 = 0.0
    mask90 = p_curve[:-1] >= 0.9
    if mask90.any():
        recall_at_90 = float(r_curve[:-1][mask90].max())

    result = {
        "model": name, "f1": float(f1), "precision": float(prec),
        "recall": float(rec), "pr_auc": float(pr_auc),
        "precision_at_k": p_at_k, "recall_at_precision90": recall_at_90,
    }
    log(f"{name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} PR-AUC={pr_auc:.4f} "
        f"P@500={p_at_k.get(500, float('nan')):.3f}")
    return result


# ---------------------------------------------------------------------------
# 8. Train models across seeds
# ---------------------------------------------------------------------------
import xgboost as xgb
import lightgbm as lgb

all_results = {"xgboost": [], "lightgbm": []}
feature_importance = {"xgboost": None, "lightgbm": None}

for seed in SEEDS:
    log(f"--- Seed {seed}: XGBoost ---")
    # NOTE: applying scale_pos_weight ON TOP OF an already-downsampled train
    # set double-corrects for imbalance and pushes predicted probabilities
    # far past 0.5, collapsing precision at the paper's F1 threshold (0.5)
    # from ~0.25 to ~0.11 while barely moving PR-AUC (verified via
    # debug_xgb_spw.py). Downsampling alone already yields a train ratio of
    # 1:{scale_pos_weight:.0f}, so no additional weighting is applied.
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_ds, y_train_ds)
    scores = xgb_model.predict_proba(X_test)[:, 1]
    all_results["xgboost"].append(evaluate(y_test, scores, f"xgboost_seed{seed}"))
    if feature_importance["xgboost"] is None:
        feature_importance["xgboost"] = dict(
            zip(FEATURES, xgb_model.feature_importances_.tolist())
        )

    log(f"--- Seed {seed}: LightGBM ---")
    # NOTE: LightGBM's leaf-wise growth badly overfits/miscalibrates when
    # scale_pos_weight is combined with an already-downsampled train set
    # (verified via debug_lgb2.py: PR-AUC collapsed from 0.24 to 0.01 with
    # scale_pos_weight=174 on top of downsampling). Rely on downsampling
    # alone for class balance and regularize leaf count / min samples
    # instead, matching XGBoost's more conservative depth-wise growth.
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    X_train_lgb = X_train_ds.copy()
    X_test_lgb = X_test.copy()
    lgb_model.fit(X_train_lgb, y_train_ds, categorical_feature=["PaymentFormat"])
    scores = lgb_model.predict_proba(X_test_lgb)[:, 1]
    all_results["lightgbm"].append(evaluate(y_test, scores, f"lightgbm_seed{seed}"))
    if feature_importance["lightgbm"] is None:
        feature_importance["lightgbm"] = dict(
            zip(FEATURES, lgb_model.feature_importances_.tolist())
        )

# ---------------------------------------------------------------------------
# 9. Aggregate + save
# ---------------------------------------------------------------------------
def summarize(results_list):
    keys = ["f1", "precision", "recall", "pr_auc", "recall_at_precision90"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results_list]
        summary[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                       "max": float(np.max(vals))}
    for k in (100, 500, 1000, 2000):
        vals = [r["precision_at_k"].get(k) for r in results_list if k in r["precision_at_k"]]
        if vals:
            summary[f"precision_at_{k}"] = {"mean": float(np.mean(vals))}
    return summary


final = {
    "dataset": DATASET,
    "n_transactions": int(n),
    "n_positive": int(df["IsLaundering"].sum()),
    "split": {
        "train": int(train_mask.sum()), "val": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "train_pos": int(df.loc[train_mask, "IsLaundering"].sum()),
        "test_pos": int(df.loc[test_mask, "IsLaundering"].sum()),
    },
    "features": FEATURES,
    "seeds": SEEDS,
    "raw_results": all_results,
    "summary": {
        "xgboost": summarize(all_results["xgboost"]),
        "lightgbm": summarize(all_results["lightgbm"]),
    },
    "feature_importance": feature_importance,
}

with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
