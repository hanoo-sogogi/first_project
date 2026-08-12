"""
IBM AML (HI-Small) GBT baseline, v3 - NO GRAPH AGGREGATES (ablation).

Directly tests Framework.html's STAGE 1 proposal ("그래프 미사용, 9개 기본
피처만") against our own v2 pipeline (results_v2.json), by removing ONLY the
8 account-level graph aggregate columns (out_degree, out_unique_cp,
out_avg_amt, in_degree, in_unique_cp, in_avg_amt, in_unique_banks,
pair_prior_count) - i.e. the features that require traversing/aggregating
the transaction graph across many rows.

Everything else is byte-for-byte identical to train_baseline_v2.py: same FX
normalization, same new-account flags, same BankCountry features, same
temporal split, same downsampling, same model configs, same 3 seeds. This
isolates the effect of graph aggregation as the only variable.

Kept (NOT considered "graph" features - derivable from a single row without
traversing other transactions): IsSelfLoop, IsSameBank. These only compare
two columns of the same row, unlike out_degree etc. which require scanning
the whole account history.
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
# 1. Load (identical to v2)
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
log(f"FX table resolved for {len(usd_per)} currencies")

df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(int)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(int)
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["PaymentFormat"] = df["PaymentFormat"].astype("category")

# --- new-account flags (identical to v2) ---
log("Computing leak-free new-account flags...")
from_events = pd.DataFrame({"idx": df.index, "node": df["FromNode"].to_numpy()})
to_events = pd.DataFrame({"idx": df.index, "node": df["ToNode"].to_numpy()})
events = pd.concat([from_events, to_events], ignore_index=True).sort_values("idx")
first_seen_idx = events.groupby("node")["idx"].min()
df["FromIsNew"] = (df["FromNode"].map(first_seen_idx) == df.index).astype(int)
df["ToIsNew"] = (df["ToNode"].map(first_seen_idx) == df.index).astype(int)

# --- BankCountry (identical to v2) ---
log("Extracting BankCountry from accounts.csv...")
accounts = pd.read_csv(DATA_DIR / f"{DATASET}_accounts.csv")
bank_country = accounts[["Bank ID", "Bank Name"]].drop_duplicates("Bank ID").copy()
bank_country["Country"] = bank_country["Bank Name"].str.extract(
    r"^([A-Za-z ]+?) Bank #\d+$"
)[0].fillna("US_domestic_style")
bank_id_to_country = dict(zip(bank_country["Bank ID"].astype(str), bank_country["Country"]))
df["FromCountry"] = df["FromBankID"].astype(str).map(bank_id_to_country).fillna("Unknown")
df["ToCountry"] = df["ToBankID"].astype(str).map(bank_id_to_country).fillna("Unknown")
df["IsCrossBorderBank"] = (df["FromCountry"] != df["ToCountry"]).astype(int)
df["FromCountry"] = df["FromCountry"].astype("category")
df["ToCountry"] = df["ToCountry"].astype("category")

# --- temporal split (identical) ---
n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2
log(f"Split -> train {train_mask.sum():,} | val {val_mask.sum():,} | test {test_mask.sum():,}")

# ---------------------------------------------------------------------------
# NO graph aggregate step here - this is the entire ablation.
# ---------------------------------------------------------------------------
log("SKIPPING graph aggregate features (out_degree, pair_prior_count, etc.) - this is the ablation")

# ---------------------------------------------------------------------------
# Feature matrix - 11 features, no graph aggregation required
# ---------------------------------------------------------------------------
FEATURES = [
    "LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank",
    "Hour", "DayOfWeek",
    "FromIsNew", "ToIsNew",
    "FromCountry", "ToCountry", "IsCrossBorderBank",
]
LABEL = "IsLaundering"

X = df[FEATURES]
y = df[LABEL]

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

rng = np.random.RandomState(0)
pos_idx = y_train[y_train == 1].index
neg_idx = y_train[y_train == 0].index
neg_sample = rng.choice(neg_idx, size=min(400_000, len(neg_idx)), replace=False)
keep_idx = pos_idx.union(pd.Index(neg_sample))
X_train_ds = X_train.loc[keep_idx]
y_train_ds = y_train.loc[keep_idx]
log(f"Downsampled train set: {len(X_train_ds):,} rows ({y_train_ds.sum():,} positive)")


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


import xgboost as xgb
import lightgbm as lgb

all_results = {"xgboost": [], "lightgbm": []}
feature_importance = {"xgboost": None, "lightgbm": None}

for seed in SEEDS:
    log(f"--- Seed {seed}: XGBoost (no graph) ---")
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

    log(f"--- Seed {seed}: LightGBM (no graph) ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_ds.copy(), y_train_ds,
                  categorical_feature=["PaymentFormat", "FromCountry", "ToCountry"])
    scores = lgb_model.predict_proba(X_test.copy())[:, 1]
    all_results["lightgbm"].append(evaluate(y_test, scores, f"lightgbm_seed{seed}"))
    if feature_importance["lightgbm"] is None:
        feature_importance["lightgbm"] = dict(
            zip(FEATURES, lgb_model.feature_importances_.tolist())
        )


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
    "ablation": "no_graph_aggregates",
    "n_transactions": int(n),
    "n_positive": int(df["IsLaundering"].sum()),
    "features": FEATURES,
    "seeds": SEEDS,
    "raw_results": all_results,
    "summary": {
        "xgboost": summarize(all_results["xgboost"]),
        "lightgbm": summarize(all_results["lightgbm"]),
    },
    "feature_importance": feature_importance,
}

with open(OUT_DIR / "results_v3_nograph.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_v3_nograph.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
