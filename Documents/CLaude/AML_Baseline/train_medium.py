"""
HI-Medium run of the pair_only pipeline (AML_실험기록.md 4절 최종 권장 피처셋),
memory-optimized for a 32M-row transaction file on a 15.6GB-RAM machine.

Key differences from the HI-Small scripts (train_baseline_experiments*.py):
  - No FromNode/ToNode STRING columns at all. Accounts are factorized once
    (From+To jointly, so IDs are consistent across both columns) into int32
    codes, then packed with the int32 bank ID into a single int64 node ID
    via `node_id = bank_id * (n_accounts+1) + account_code`. This avoids
    ever materializing a 32M-row Python string array (the biggest memory
    cost in the original approach).
  - dtype-optimized CSV read (int32/float32/category) instead of pandas
    defaults (int64/float64/object).
  - Explicit `del` + `gc.collect()` after every large intermediate is no
    longer needed, with psutil memory-usage logging at each checkpoint so
    a problem shows up in the log before an OOM kill.
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
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
DATASET = "HI-Medium"
SEEDS = [1, 2, 3]
NEG_CAP = 400_000  # same cap as the HI-Small pair_only runs, for comparability

t0 = time.time()
proc = psutil.Process()


def log(msg):
    mem_gb = proc.memory_info().rss / 1e9
    sys_avail_gb = psutil.virtual_memory().available / 1e9
    print(f"[{time.time() - t0:7.1f}s | proc {mem_gb:5.2f}GB | sys_avail {sys_avail_gb:5.2f}GB] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load with compact dtypes
# ---------------------------------------------------------------------------
log("Loading transactions (dtype-optimized)...")
COLS = ["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount",
        "AmountReceived", "ReceivingCurrency", "AmountPaid", "PaymentCurrency",
        "PaymentFormat", "IsLaundering"]
DTYPES = {
    "FromBankID": "int32", "ToBankID": "int32",
    "AmountReceived": "float32", "AmountPaid": "float32",
    "ReceivingCurrency": "category", "PaymentCurrency": "category",
    "PaymentFormat": "category", "IsLaundering": "int8",
}
df = pd.read_csv(DATA_DIR / f"{DATASET}_Trans.csv", header=0, names=COLS, dtype=DTYPES)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
log(f"Loaded {len(df):,} transactions, {int(df['IsLaundering'].sum()):,} laundering")

log("Sorting by timestamp (stable)...")
df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
gc.collect()
log("Sorted")

# ---------------------------------------------------------------------------
# 2. Node IDs without ever building a full string node column
# ---------------------------------------------------------------------------
log("Factorizing accounts (From+To jointly, no string node columns)...")
combined_accounts = pd.concat(
    [df["FromAccount"].astype(str), df["ToAccount"].astype(str)], ignore_index=True
)
acct_codes, acct_uniques = pd.factorize(combined_accounts, sort=False)
del combined_accounts
gc.collect()
n = len(df)
df["FromAcctCode"] = acct_codes[:n].astype(np.int32)
df["ToAcctCode"] = acct_codes[n:].astype(np.int32)
del acct_codes
n_accounts = len(acct_uniques)
del acct_uniques
gc.collect()
log(f"{n_accounts:,} unique account strings factorized")

pack = np.int64(n_accounts + 1)
df["FromNode"] = df["FromBankID"].astype(np.int64) * pack + df["FromAcctCode"].astype(np.int64)
df["ToNode"] = df["ToBankID"].astype(np.int64) * pack + df["ToAcctCode"].astype(np.int64)
df = df.drop(columns=["FromAccount", "ToAccount", "FromAcctCode", "ToAcctCode"])
gc.collect()
log("Packed node IDs built, raw account strings dropped")

# ---------------------------------------------------------------------------
# 3. FX normalization (BFS over currency graph, same as HI-Small scripts)
# ---------------------------------------------------------------------------
log("Resolving FX table...")
fx = df.loc[df["ReceivingCurrency"] != df["PaymentCurrency"], ["PaymentCurrency", "ReceivingCurrency", "AmountReceived", "AmountPaid"]].copy()
fx["rate_recv_per_paid"] = fx["AmountReceived"].astype(np.float64) / fx["AmountPaid"].astype(np.float64)
pair_rate = fx.groupby(["PaymentCurrency", "ReceivingCurrency"], observed=True)["rate_recv_per_paid"].median().to_dict()
del fx
gc.collect()
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
for c in set(df["PaymentCurrency"].cat.categories) | set(df["ReceivingCurrency"].cat.categories):
    usd_per.setdefault(c, 1.0)
df["AmountUSD"] = df["AmountPaid"].astype(np.float64) * df["PaymentCurrency"].map(usd_per).astype(np.float64)
df["LogAmountUSD"] = np.log1p(df["AmountUSD"]).astype(np.float32)
df = df.drop(columns=["AmountUSD", "AmountReceived", "AmountPaid"])
log(f"FX table resolved for {len(usd_per)} currencies")

# ---------------------------------------------------------------------------
# 4. Cheap flags
# ---------------------------------------------------------------------------
df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(np.int8)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(np.int8)
df["Hour"] = df["Timestamp"].dt.hour.astype(np.int8)
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype(np.int8)
log("Flags computed (IsSelfLoop, IsSameBank, Hour, DayOfWeek)")

# ---------------------------------------------------------------------------
# 5. Leak-free new-account flags
# ---------------------------------------------------------------------------
log("Computing leak-free new-account flags...")
idx_arr = np.arange(n, dtype=np.int64)
events = pd.DataFrame({
    "idx": np.concatenate([idx_arr, idx_arr]),
    "node": np.concatenate([df["FromNode"].to_numpy(), df["ToNode"].to_numpy()]),
})
first_seen_idx = events.groupby("node")["idx"].min()
del events
gc.collect()
df["FromIsNew"] = (df["FromNode"].map(first_seen_idx).to_numpy() == idx_arr).astype(np.int8)
df["ToIsNew"] = (df["ToNode"].map(first_seen_idx).to_numpy() == idx_arr).astype(np.int8)
del first_seen_idx, idx_arr
gc.collect()
log("New-account flags computed")

# ---------------------------------------------------------------------------
# 6. BankCountry
# ---------------------------------------------------------------------------
log("Extracting BankCountry from accounts.csv...")
accounts = pd.read_csv(DATA_DIR / f"{DATASET}_accounts.csv", usecols=["Bank ID", "Bank Name"])
bank_country = accounts.drop_duplicates("Bank ID").copy()
bank_country["Country"] = bank_country["Bank Name"].str.extract(
    r"^([A-Za-z ]+?) Bank #\d+$"
)[0].fillna("US_domestic_style")
bank_id_to_country = dict(zip(bank_country["Bank ID"].astype(np.int32), bank_country["Country"]))
del accounts, bank_country
df["FromCountry"] = df["FromBankID"].map(bank_id_to_country).fillna("Unknown").astype("category")
df["ToCountry"] = df["ToBankID"].map(bank_id_to_country).fillna("Unknown").astype("category")
df["IsCrossBorderBank"] = (df["FromCountry"] != df["ToCountry"]).astype(np.int8)
gc.collect()
log("BankCountry attached")

# ---------------------------------------------------------------------------
# 7. Temporal split
# ---------------------------------------------------------------------------
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = (df["Timestamp"] < t1).to_numpy()
test_mask = (df["Timestamp"] >= t2).to_numpy()
log(f"Split -> train {train_mask.sum():,} | val {n - train_mask.sum() - test_mask.sum():,} | test {test_mask.sum():,}")

# ---------------------------------------------------------------------------
# 8. pair_prior_count (leak-free, self-loop excluded) -- only graph feature
#    kept, per AML_실험기록.md 4절 (pair_only is the final recommended set)
# ---------------------------------------------------------------------------
log("Building pair_prior_count (train-period, self-loop excluded)...")
train_from = df["FromNode"].to_numpy()[train_mask]
train_to = df["ToNode"].to_numpy()[train_mask]
not_self = train_from != train_to
pair_df = pd.DataFrame({"FromNode": train_from[not_self], "ToNode": train_to[not_self]})
pair_count = pair_df.groupby(["FromNode", "ToNode"]).size().rename("pair_prior_count")
del train_from, train_to, not_self, pair_df
gc.collect()
df = df.merge(pair_count, on=["FromNode", "ToNode"], how="left")
df["pair_prior_count"] = df["pair_prior_count"].fillna(0).astype(np.float32)
del pair_count
gc.collect()
log("pair_prior_count attached")

FEATURES = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek",
            "FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank",
            "pair_prior_count"]
CAT_COLS = ["PaymentFormat", "FromCountry", "ToCountry"]

y_all = df["IsLaundering"].to_numpy()
X_test_v = df.loc[test_mask, FEATURES].reset_index(drop=True)
y_test = y_all[test_mask]
X_train_full = df.loc[train_mask, FEATURES].reset_index(drop=True)
y_train_full = y_all[train_mask]
del df
gc.collect()
log(f"Feature frames built. Train candidates {len(X_train_full):,}, test {len(X_test_v):,}")

rng = np.random.RandomState(0)
pos_idx = np.where(y_train_full == 1)[0]
neg_idx = np.where(y_train_full == 0)[0]
neg_sample = rng.choice(neg_idx, size=min(NEG_CAP, len(neg_idx)), replace=False)
keep_idx = np.concatenate([pos_idx, neg_sample])
X_train_ds = X_train_full.iloc[keep_idx].reset_index(drop=True)
y_train_ds = y_train_full[keep_idx]
del X_train_full, y_train_full, pos_idx, neg_idx, neg_sample, keep_idx
gc.collect()
log(f"Downsample index fixed: {len(X_train_ds):,} rows ({int(y_train_ds.sum()):,} positive)")


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


def summarize(results_list):
    keys = ["f1", "precision", "recall", "pr_auc", "recall_at_precision90"]
    return {k: {"mean": float(np.mean([r[k] for r in results_list])),
                "min": float(np.min([r[k] for r in results_list])),
                "max": float(np.max([r[k] for r in results_list]))} for k in keys}


import xgboost as xgb
import lightgbm as lgb

results = {"xgboost": [], "lightgbm": []}
for seed in SEEDS:
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_ds, y_train_ds)
    scores = xgb_model.predict_proba(X_test_v)[:, 1]
    results["xgboost"].append(evaluate(y_test, scores, f"xgb_s{seed}"))
    del xgb_model, scores
    gc.collect()

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_ds.copy(), y_train_ds, categorical_feature=CAT_COLS)
    scores = lgb_model.predict_proba(X_test_v.copy())[:, 1]
    results["lightgbm"].append(evaluate(y_test, scores, f"lgb_s{seed}"))
    del lgb_model, scores
    gc.collect()

out = {
    "dataset": DATASET,
    "features": FEATURES,
    "neg_cap": NEG_CAP,
    "n_train_candidates": int(train_mask.sum()),
    "n_test": int(test_mask.sum()),
    "n_train_downsampled": int(len(X_train_ds)),
    "raw": results,
    "summary": {"xgboost": summarize(results["xgboost"]), "lightgbm": summarize(results["lightgbm"])},
}
with open(OUT_DIR / "results_medium.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_medium.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
