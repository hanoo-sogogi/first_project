"""
IBM AML (HI-Small) GBT feature-ablation harness.

Preprocessing (FX, self-loop, new-account, BankCountry, graph aggregates,
temporal split, downsampling) is done ONCE, then several named feature-set
variants are trained and evaluated (XGBoost + LightGBM, 3 seeds each) to
isolate which pieces actually drive performance.

Variants:
  full          - everything (== v2)
  no_graph      - no graph aggregates (== v3)
  pair_only     - no-graph baseline + pair_prior_count only
  degree_only   - no-graph baseline + out_degree/in_degree only
  no_selfloop   - full minus IsSelfLoop (how much is that one feature doing?)
  lgb_reg_heavy - full features, LightGBM with heavier regularization
                   (only affects the lightgbm run, xgboost result reused from full)
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
# Shared preprocessing (identical to v2, done once)
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

BASE = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek",
        "FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank"]

VARIANTS = {
    "full": BASE + agg_cols,
    "no_graph": BASE,
    "pair_only": BASE + ["pair_prior_count"],
    "degree_only": BASE + ["out_degree", "in_degree"],
    "no_selfloop": [f for f in BASE if f != "IsSelfLoop"] + agg_cols,
}

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
    summary = {k: {"mean": float(np.mean([r[k] for r in results_list])),
                    "min": float(np.min([r[k] for r in results_list])),
                    "max": float(np.max([r[k] for r in results_list]))} for k in keys}
    return summary


import xgboost as xgb
import lightgbm as lgb

all_variant_results = {}

for variant_name, features in VARIANTS.items():
    log(f"=== Variant: {variant_name} ({len(features)} features) ===")
    cat_cols = [f for f in ["PaymentFormat", "FromCountry", "ToCountry"] if f in features]

    X_train_ds = X_train_full.loc[keep_idx, features]
    y_train_ds = y_train_full.loc[keep_idx]
    X_test_v = X_test_full[features]

    variant_results = {"xgboost": [], "lightgbm": []}
    for seed in SEEDS:
        xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            tree_method="hist", enable_categorical=True, random_state=seed,
        )
        xgb_model.fit(X_train_ds, y_train_ds)
        scores = xgb_model.predict_proba(X_test_v)[:, 1]
        variant_results["xgboost"].append(evaluate(y_test, scores, f"{variant_name}_xgb_s{seed}"))

        if variant_name == "lgb_reg_heavy":
            lgb_model = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=7, min_child_samples=200,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                reg_lambda=5.0, reg_alpha=1.0, random_state=seed, verbosity=-1,
            )
        else:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=15, min_child_samples=50,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=seed, verbosity=-1,
            )
        lgb_model.fit(X_train_ds.copy(), y_train_ds, categorical_feature=cat_cols)
        scores = lgb_model.predict_proba(X_test_v.copy())[:, 1]
        variant_results["lightgbm"].append(evaluate(y_test, scores, f"{variant_name}_lgb_s{seed}"))

    all_variant_results[variant_name] = {
        "features": features,
        "summary": {"xgboost": summarize(variant_results["xgboost"]),
                    "lightgbm": summarize(variant_results["lightgbm"])},
        "raw": variant_results,
    }

# extra: lgb_reg_heavy uses the SAME features as "full" but heavier regularization
variant_name = "lgb_reg_heavy"
features = VARIANTS["full"]
log(f"=== Variant: {variant_name} ({len(features)} features, LightGBM only) ===")
cat_cols = [f for f in ["PaymentFormat", "FromCountry", "ToCountry"] if f in features]
X_train_ds = X_train_full.loc[keep_idx, features]
y_train_ds = y_train_full.loc[keep_idx]
X_test_v = X_test_full[features]
variant_results = {"lightgbm": []}
for seed in SEEDS:
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=7, min_child_samples=200,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5.0, reg_alpha=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_ds.copy(), y_train_ds, categorical_feature=cat_cols)
    scores = lgb_model.predict_proba(X_test_v.copy())[:, 1]
    variant_results["lightgbm"].append(evaluate(y_test, scores, f"{variant_name}_lgb_s{seed}"))
all_variant_results[variant_name] = {
    "features": features,
    "summary": {"lightgbm": summarize(variant_results["lightgbm"])},
    "raw": variant_results,
}

with open(OUT_DIR / "results_experiments.json", "w", encoding="utf-8") as f:
    json.dump(all_variant_results, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_experiments.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
