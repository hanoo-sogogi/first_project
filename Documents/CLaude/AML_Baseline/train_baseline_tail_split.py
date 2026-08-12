"""
Re-evaluation of the HI-Small GBT baseline, splitting the test set into:
  - "normal" test period  : test-split transactions from before 2022-09-11
  - "tail" test period     : test-split transactions from 2022-09-11 onward

Motivation: daily EDA showed transaction volume collapses after 2022-09-10
(from ~200K-1.1M/day down to 10-400/day) while the local laundering rate in
that sparse tail jumps to 56-73% (vs 0.03-0.21% in the normal period). The
temporal 80% split boundary lands mid-day on 2022-09-08, so the test set
absorbs the entire anomalous tail: 655 of the test set's 1,798 positive
labels (36%) come from just 1,108 tail transactions (0.02% of all data).
This script checks whether the previously reported test-set metrics are
being driven disproportionately by that tail, by scoring it separately.

Everything else (feature engineering, downsampling, model configs) is
identical to train_baseline.py so the numbers are directly comparable.
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
TAIL_START = pd.Timestamp("2022-09-11")

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load (identical to train_baseline.py)
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
log(f"Loaded {len(df):,} transactions, {df['IsLaundering'].sum():,} laundering")

df["FromNode"] = df["FromBankID"].astype(str) + "_" + df["FromAccount"]
df["ToNode"] = df["ToBankID"].astype(str) + "_" + df["ToAccount"]

# --- FX -> USD (identical) ---
fx = df[df["ReceivingCurrency"] != df["PaymentCurrency"]].copy()
fx["rate_recv_per_paid"] = fx["AmountReceived"] / fx["AmountPaid"]
pair_rate = fx.groupby(["PaymentCurrency", "ReceivingCurrency"])["rate_recv_per_paid"].median().to_dict()
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
                usd_per[nb] = usd_per[cur] / r
                nxt.append(nb)
    frontier = nxt
for c in set(df["PaymentCurrency"]) | set(df["ReceivingCurrency"]):
    usd_per.setdefault(c, 1.0)
df["AmountUSD"] = df["AmountPaid"] * df["PaymentCurrency"].map(usd_per)
df["LogAmountUSD"] = np.log1p(df["AmountUSD"])
log("FX normalization done")

df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(int)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(int)
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["PaymentFormat"] = df["PaymentFormat"].astype("category")

# ---------------------------------------------------------------------------
# 2. Temporal 60/20/20 split (identical), plus a tail sub-split of test
# ---------------------------------------------------------------------------
n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2

test_normal_mask = test_mask & (df["Timestamp"] < TAIL_START)
test_tail_mask = test_mask & (df["Timestamp"] >= TAIL_START)

log(f"Split -> train {train_mask.sum():,} (pos {df.loc[train_mask,'IsLaundering'].sum():,}) | "
    f"val {val_mask.sum():,} (pos {df.loc[val_mask,'IsLaundering'].sum():,}) | "
    f"test {test_mask.sum():,} (pos {df.loc[test_mask,'IsLaundering'].sum():,})")
log(f"  test split -> normal(<9/11) {test_normal_mask.sum():,} "
    f"(pos {df.loc[test_normal_mask,'IsLaundering'].sum():,}) | "
    f"tail(>=9/11) {test_tail_mask.sum():,} "
    f"(pos {df.loc[test_tail_mask,'IsLaundering'].sum():,})")

# ---------------------------------------------------------------------------
# 3. Graph aggregate features from TRAIN period only (identical)
# ---------------------------------------------------------------------------
log("Building leak-free graph aggregate features from train period...")
train_df = df.loc[train_mask]
out_agg = train_df.groupby("FromNode").agg(
    out_degree=("ToNode", "count"), out_unique_cp=("ToNode", "nunique"),
    out_avg_amt=("LogAmountUSD", "mean"),
)
in_agg = train_df.groupby("ToNode").agg(
    in_degree=("FromNode", "count"), in_unique_cp=("FromNode", "nunique"),
    in_avg_amt=("LogAmountUSD", "mean"), in_unique_banks=("FromBankID", "nunique"),
)
pair_count = train_df.groupby(["FromNode", "ToNode"]).size().rename("pair_prior_count")
df = df.merge(out_agg, left_on="FromNode", right_index=True, how="left")
df = df.merge(in_agg, left_on="ToNode", right_index=True, how="left")
df = df.merge(pair_count, on=["FromNode", "ToNode"], how="left")
agg_cols = ["out_degree", "out_unique_cp", "out_avg_amt", "in_degree",
            "in_unique_cp", "in_avg_amt", "in_unique_banks", "pair_prior_count"]
for c in agg_cols:
    df[c] = df[c].fillna(0)
log("Graph features attached")

# ---------------------------------------------------------------------------
# 4. Feature matrix + downsampled train (identical)
# ---------------------------------------------------------------------------
FEATURES = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank",
            "Hour", "DayOfWeek"] + agg_cols
LABEL = "IsLaundering"
X = df[FEATURES]
y = df[LABEL]

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test_all, y_test_all = X.loc[test_mask], y.loc[test_mask]
X_test_normal, y_test_normal = X.loc[test_normal_mask], y.loc[test_normal_mask]
X_test_tail, y_test_tail = X.loc[test_tail_mask], y.loc[test_tail_mask]

rng = np.random.RandomState(0)
pos_idx = y_train[y_train == 1].index
neg_idx = y_train[y_train == 0].index
neg_sample = rng.choice(neg_idx, size=min(400_000, len(neg_idx)), replace=False)
keep_idx = pos_idx.union(pd.Index(neg_sample))
X_train_ds = X_train.loc[keep_idx]
y_train_ds = y_train.loc[keep_idx]
log(f"Downsampled train set: {len(X_train_ds):,} rows ({y_train_ds.sum():,} positive)")


# ---------------------------------------------------------------------------
# 5. Evaluation helper (identical)
# ---------------------------------------------------------------------------
def evaluate(y_true, scores, name):
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        # degenerate subset (e.g. handful of rows) - report what we can
        return {"model": name, "n": int(len(y_true)), "n_pos": n_pos,
                "f1": None, "precision": None, "recall": None,
                "pr_auc": None, "precision_at_k": {}, "recall_at_precision90": None,
                "note": "degenerate subset (no both classes present)"}

    preds = (scores >= 0.5).astype(int)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    p_at_k = {}
    for k in (10, 50, 100, 500, 1000):
        if len(y_sorted) >= k:
            p_at_k[k] = float(y_sorted[:k].sum() / k)

    p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
    recall_at_90 = 0.0
    mask90 = p_curve[:-1] >= 0.9
    if mask90.any():
        recall_at_90 = float(r_curve[:-1][mask90].max())

    result = {
        "model": name, "n": int(len(y_true)), "n_pos": n_pos,
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "pr_auc": float(pr_auc), "precision_at_k": p_at_k,
        "recall_at_precision90": recall_at_90,
    }
    log(f"{name}: n={len(y_true):,} pos={n_pos} F1={f1:.4f} P={prec:.4f} "
        f"R={rec:.4f} PR-AUC={pr_auc:.4f}")
    return result


# ---------------------------------------------------------------------------
# 6. Train + evaluate on all 3 test subsets, per seed, per model
# ---------------------------------------------------------------------------
import xgboost as xgb
import lightgbm as lgb

all_results = {"xgboost": [], "lightgbm": []}

for seed in SEEDS:
    log(f"--- Seed {seed}: XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_ds, y_train_ds)
    seed_result = {"seed": seed}
    for subset_name, Xs, ys in [
        ("combined", X_test_all, y_test_all),
        ("normal", X_test_normal, y_test_normal),
        ("tail", X_test_tail, y_test_tail),
    ]:
        scores = xgb_model.predict_proba(Xs)[:, 1]
        seed_result[subset_name] = evaluate(ys, scores, f"xgboost_seed{seed}_{subset_name}")
    all_results["xgboost"].append(seed_result)

    log(f"--- Seed {seed}: LightGBM ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_ds, y_train_ds, categorical_feature=["PaymentFormat"])
    seed_result = {"seed": seed}
    for subset_name, Xs, ys in [
        ("combined", X_test_all, y_test_all),
        ("normal", X_test_normal, y_test_normal),
        ("tail", X_test_tail, y_test_tail),
    ]:
        scores = lgb_model.predict_proba(Xs)[:, 1]
        seed_result[subset_name] = evaluate(ys, scores, f"lightgbm_seed{seed}_{subset_name}")
    all_results["lightgbm"].append(seed_result)


# ---------------------------------------------------------------------------
# 7. Aggregate + save
# ---------------------------------------------------------------------------
def summarize_subset(seed_results, subset_name):
    vals_list = [sr[subset_name] for sr in seed_results if sr[subset_name]["f1"] is not None]
    if not vals_list:
        return {"note": "all seeds degenerate for this subset"}
    keys = ["f1", "precision", "recall", "pr_auc", "recall_at_precision90"]
    summary = {}
    for k in keys:
        vals = [v[k] for v in vals_list]
        summary[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    for k in (10, 50, 100, 500, 1000):
        vals = [v["precision_at_k"].get(k) for v in vals_list if k in v["precision_at_k"]]
        if vals:
            summary[f"precision_at_{k}"] = {"mean": float(np.mean(vals))}
    summary["n"] = vals_list[0]["n"]
    summary["n_pos"] = vals_list[0]["n_pos"]
    summary["n_seeds_used"] = len(vals_list)
    return summary


final = {
    "dataset": DATASET,
    "tail_start": str(TAIL_START.date()),
    "split": {
        "test_combined": {"n": int(test_mask.sum()), "n_pos": int(df.loc[test_mask, "IsLaundering"].sum())},
        "test_normal": {"n": int(test_normal_mask.sum()), "n_pos": int(df.loc[test_normal_mask, "IsLaundering"].sum())},
        "test_tail": {"n": int(test_tail_mask.sum()), "n_pos": int(df.loc[test_tail_mask, "IsLaundering"].sum())},
    },
    "seeds": SEEDS,
    "raw_results": all_results,
    "summary": {
        "xgboost": {
            "combined": summarize_subset(all_results["xgboost"], "combined"),
            "normal": summarize_subset(all_results["xgboost"], "normal"),
            "tail": summarize_subset(all_results["xgboost"], "tail"),
        },
        "lightgbm": {
            "combined": summarize_subset(all_results["lightgbm"], "combined"),
            "normal": summarize_subset(all_results["lightgbm"], "normal"),
            "tail": summarize_subset(all_results["lightgbm"], "tail"),
        },
    },
}

with open(OUT_DIR / "results_tail_split.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_tail_split.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
