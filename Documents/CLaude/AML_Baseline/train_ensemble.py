"""
Round 7: multi-seed ensemble. experiments6.py (3.11 in the write-up) found
that BOTH pair_only and remove2 have a seed that collapses XGBoost (seed 6)
and/or LightGBM (seed 4) - a data-scale problem (2,297 positive examples),
not a feature-selection problem. This script tests whether averaging
predicted probabilities across multiple seeds actually rescues those
collapses, using the recommended pair_only (12-feature) set.

Trains XGBoost + LightGBM on seeds 1-6 (the same 6 seeds already
individually tested in experiments.py/experiments2.py/experiments6.py, so
results are directly comparable), keeps the raw test-set probability
arrays, then evaluates several ensembles:
  - xgb_all6 / lgb_all6: same-model-type ensemble across all 6 seeds
  - combined_all12: all 12 models (both types, all seeds) averaged
  - xgb_incremental_k / lgb_incremental_k: first K seeds averaged, K=1..6,
    to see how many seeds are needed before a collapsed seed stops hurting
  - xgb_worst_pair / lgb_worst_pair: averaging just the two seeds that
    included the known-bad one (seed6 for XGB, seed4 for LGB) with a
    known-good neighbor, to see if even 2-way averaging is enough

Preprocessing is identical to the previous experiments*.py scripts
(copy-pasted, not imported, to keep this a standalone reproducible script).
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
SEEDS = [1, 2, 3, 4, 5, 6]

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


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

log("Computing leak-free new-account flags...")
from_events = pd.DataFrame({"idx": df.index, "node": df["FromNode"].to_numpy()})
to_events = pd.DataFrame({"idx": df.index, "node": df["ToNode"].to_numpy()})
events = pd.concat([from_events, to_events], ignore_index=True).sort_values("idx")
first_seen_idx = events.groupby("node")["idx"].min()
df["FromIsNew"] = (df["FromNode"].map(first_seen_idx) == df.index).astype(int)
df["ToIsNew"] = (df["ToNode"].map(first_seen_idx) == df.index).astype(int)

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

n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2
log(f"Split -> train {train_mask.sum():,} | val {val_mask.sum():,} | test {test_mask.sum():,}")

log("Building leak-free graph aggregate features (self-loop excluded)...")
train_df_full = df.loc[train_mask]
train_df = train_df_full[train_df_full["FromNode"] != train_df_full["ToNode"]]
pair_count = train_df.groupby(["FromNode", "ToNode"]).size().rename("pair_prior_count")
df = df.merge(pair_count, on=["FromNode", "ToNode"], how="left")
df["pair_prior_count"] = df["pair_prior_count"].fillna(0)
log("pair_prior_count attached")

FEATURES = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek",
            "FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank",
            "pair_prior_count"]
CAT_COLS = ["PaymentFormat", "FromCountry", "ToCountry"]

y_all = df["IsLaundering"]
X_test_full = df.loc[test_mask]
y_test = y_all.loc[test_mask]
X_train_full = df.loc[train_mask]
y_train_full = y_all.loc[train_mask]

rng = np.random.RandomState(0)
pos_idx = y_train_full[y_train_full == 1].index
neg_idx = y_train_full[y_train_full == 0].index
neg_sample = rng.choice(neg_idx, size=min(400_000, len(neg_idx)), replace=False)
keep_idx = pos_idx.union(pd.Index(neg_sample))
log(f"Downsample index fixed: {len(keep_idx):,} rows ({y_train_full.loc[keep_idx].sum():,} positive)")

X_train_ds = X_train_full.loc[keep_idx, FEATURES]
y_train_ds = y_train_full.loc[keep_idx]
X_test_v = X_test_full[FEATURES]

y_test_np = y_test.to_numpy()


def evaluate(y_true, scores, name):
    preds = (scores >= 0.5).astype(int)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)
    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    p_at_k = {k: float(y_sorted[:k].sum() / k) for k in (100, 500, 1000, 2000) if len(y_sorted) >= k}
    p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
    mask90 = p_curve[:-1] >= 0.9
    recall_at_90 = float(r_curve[:-1][mask90].max()) if mask90.any() else 0.0
    result = {"model": name, "f1": float(f1), "precision": float(prec), "recall": float(rec),
              "pr_auc": float(pr_auc), "precision_at_k": p_at_k, "recall_at_precision90": recall_at_90}
    log(f"{name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} PR-AUC={pr_auc:.4f}")
    return result


import xgboost as xgb
import lightgbm as lgb

xgb_scores = {}
lgb_scores = {}
individual_results = {"xgboost": [], "lightgbm": []}

for seed in SEEDS:
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_ds, y_train_ds)
    scores = xgb_model.predict_proba(X_test_v)[:, 1]
    xgb_scores[seed] = scores
    individual_results["xgboost"].append(evaluate(y_test_np, scores, f"xgb_s{seed}"))

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_ds.copy(), y_train_ds, categorical_feature=CAT_COLS)
    scores = lgb_model.predict_proba(X_test_v.copy())[:, 1]
    lgb_scores[seed] = scores
    individual_results["lightgbm"].append(evaluate(y_test_np, scores, f"lgb_s{seed}"))

log("=== Ensembles ===")
ensemble_results = {}

xgb_avg_all6 = np.mean([xgb_scores[s] for s in SEEDS], axis=0)
ensemble_results["xgb_all6"] = evaluate(y_test_np, xgb_avg_all6, "xgb_all6")

lgb_avg_all6 = np.mean([lgb_scores[s] for s in SEEDS], axis=0)
ensemble_results["lgb_all6"] = evaluate(y_test_np, lgb_avg_all6, "lgb_all6")

combined_all12 = np.mean([xgb_scores[s] for s in SEEDS] + [lgb_scores[s] for s in SEEDS], axis=0)
ensemble_results["combined_all12"] = evaluate(y_test_np, combined_all12, "combined_all12")

log("--- Incremental (first K seeds, in order 1,2,3,4,5,6) ---")
for k in range(1, 7):
    ks = SEEDS[:k]
    avg = np.mean([xgb_scores[s] for s in ks], axis=0)
    ensemble_results[f"xgb_first{k}"] = evaluate(y_test_np, avg, f"xgb_first{k}")
for k in range(1, 7):
    ks = SEEDS[:k]
    avg = np.mean([lgb_scores[s] for s in ks], axis=0)
    ensemble_results[f"lgb_first{k}"] = evaluate(y_test_np, avg, f"lgb_first{k}")

log("--- Worst-seed rescue pairs ---")
# XGBoost: seed 6 is the known collapse (from experiments6.py). Pair it with
# a known-good neighbor (seed 5) to see if 2-way averaging is enough.
avg = np.mean([xgb_scores[5], xgb_scores[6]], axis=0)
ensemble_results["xgb_pair_5_6"] = evaluate(y_test_np, avg, "xgb_pair_5_6")
avg = np.mean([xgb_scores[1], xgb_scores[6]], axis=0)
ensemble_results["xgb_pair_1_6"] = evaluate(y_test_np, avg, "xgb_pair_1_6")

# LightGBM: seed 4 is the known collapse (remove2 variant, but same seed
# also tested here on pair_only for a fair like-for-like check).
avg = np.mean([lgb_scores[3], lgb_scores[4]], axis=0)
ensemble_results["lgb_pair_3_4"] = evaluate(y_test_np, avg, "lgb_pair_3_4")
avg = np.mean([lgb_scores[4], lgb_scores[5]], axis=0)
ensemble_results["lgb_pair_4_5"] = evaluate(y_test_np, avg, "lgb_pair_4_5")

out = {
    "features": FEATURES,
    "seeds": SEEDS,
    "individual": individual_results,
    "ensembles": ensemble_results,
}
with open(OUT_DIR / "results_ensemble.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_ensemble.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
