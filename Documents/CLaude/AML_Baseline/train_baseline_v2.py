"""
IBM AML (HI-Small) GBT baseline, v2 - low-cost improvement bundle.

Adds three changes on top of train_baseline.py, each independently verified
in this project's own analysis (not speculative):

  1. is_new_account flags (FromIsNew / ToIsNew): computed leak-free as
     "is this the very first time this account appears, anywhere in the
     time-sorted data up to and including this row". EDA found test-period
     transactions involving a brand-new (never-seen) account have a 3.30%
     laundering rate vs ~0.11-0.17% otherwise (~20-30x) - but the old
     pipeline gave these accounts all-zero graph features and no explicit
     signal that they were new.

  2. BankCountry / IsCrossBorderBank: extracted from HI-Small_accounts.csv's
     "Bank Name" field, which encodes a country for 97.7% of banks (e.g.
     "Saudi Arabia Bank #1234"). Verified laundering rate varies up to ~5x
     across countries (Saudi Arabia 0.45% vs baseline ~0.10%) and this
     signal was previously discarded entirely (only Bank ID was used, as a
     high-cardinality join key with no direct predictive use).

  3. Self-loop-excluded graph aggregates: out_degree/in_degree/pair_prior_count
     etc. previously counted self-loop transactions (11.64% of all rows,
     81% "Reinvestment", laundering rate 0.0019% - near-zero signal). This
     inflated out_degree by a median +25% for 74.5% of accounts, diluting a
     feature that was the single strongest predictor (importance 0.202).
     Self-loop rows are still scored (kept in train/val/test) - they are
     just excluded from OTHER accounts' aggregate statistics.

Everything else (FX normalization, temporal 60/20/20 split, downsampling,
model configs) is unchanged from train_baseline.py so the effect of these
three changes is directly attributable.
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
log(f"Loaded {len(df):,} transactions, {df['IsLaundering'].sum():,} laundering")

df["FromNode"] = df["FromBankID"].astype(str) + "_" + df["FromAccount"]
df["ToNode"] = df["ToBankID"].astype(str) + "_" + df["ToAccount"]

# ---------------------------------------------------------------------------
# 2. Currency normalization -> USD (unchanged)
# ---------------------------------------------------------------------------
log("Deriving FX table from mismatched-currency transactions...")
fx = df[df["ReceivingCurrency"] != df["PaymentCurrency"]].copy()
fx["rate_recv_per_paid"] = fx["AmountReceived"] / fx["AmountPaid"]
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
                usd_per[nb] = usd_per[cur] / r
                nxt.append(nb)
    frontier = nxt
for c in set(df["PaymentCurrency"]) | set(df["ReceivingCurrency"]):
    usd_per.setdefault(c, 1.0)
df["AmountUSD"] = df["AmountPaid"] * df["PaymentCurrency"].map(usd_per)
df["LogAmountUSD"] = np.log1p(df["AmountUSD"])
log(f"FX table resolved for {len(usd_per)} currencies")

# ---------------------------------------------------------------------------
# 3. Simple structural / temporal features (unchanged)
# ---------------------------------------------------------------------------
df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(int)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(int)
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["PaymentFormat"] = df["PaymentFormat"].astype("category")

# ---------------------------------------------------------------------------
# 4. NEW: is_new_account flags (leak-free, temporally streaming)
# ---------------------------------------------------------------------------
log("Computing leak-free new-account flags...")
from_events = pd.DataFrame({"idx": df.index, "node": df["FromNode"].to_numpy()})
to_events = pd.DataFrame({"idx": df.index, "node": df["ToNode"].to_numpy()})
events = pd.concat([from_events, to_events], ignore_index=True).sort_values("idx")
first_seen_idx = events.groupby("node")["idx"].min()
df["FromIsNew"] = (df["FromNode"].map(first_seen_idx) == df.index).astype(int)
df["ToIsNew"] = (df["ToNode"].map(first_seen_idx) == df.index).astype(int)
log(f"FromIsNew rate: {df['FromIsNew'].mean()*100:.3f}%, "
    f"ToIsNew rate: {df['ToIsNew'].mean()*100:.3f}%")

# ---------------------------------------------------------------------------
# 5. NEW: BankCountry / IsCrossBorderBank from accounts.csv Bank Name
# ---------------------------------------------------------------------------
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
log(f"{df['FromCountry'].nunique()} unique FromCountry categories, "
    f"cross-border rate: {df['IsCrossBorderBank'].mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 6. Temporal 60/20/20 split (unchanged)
# ---------------------------------------------------------------------------
n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2
log(f"Split -> train {train_mask.sum():,} | val {val_mask.sum():,} | test {test_mask.sum():,}")

# ---------------------------------------------------------------------------
# 7. Graph aggregate features from TRAIN period, EXCLUDING self-loops (fix #3)
# ---------------------------------------------------------------------------
log("Building leak-free graph aggregate features from train period (self-loop excluded)...")
train_df_full = df.loc[train_mask]
train_df = train_df_full[train_df_full["FromNode"] != train_df_full["ToNode"]]
log(f"  {len(train_df_full) - len(train_df):,} self-loop rows excluded from aggregation "
    f"({(len(train_df_full) - len(train_df)) / len(train_df_full) * 100:.1f}% of train)")

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

log("Graph features attached (self-loop-excluded): " + ", ".join(agg_cols))

# ---------------------------------------------------------------------------
# 8. Assemble feature matrix
# ---------------------------------------------------------------------------
FEATURES = [
    "LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank",
    "Hour", "DayOfWeek",
    "FromIsNew", "ToIsNew",                              # NEW
    "FromCountry", "ToCountry", "IsCrossBorderBank",      # NEW
] + agg_cols
LABEL = "IsLaundering"

X = df[FEATURES]
y = df[LABEL]

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

rng = np.random.RandomState(0)
pos_idx = y_train[y_train == 1].index
neg_idx = y_train[y_train == 0].index
neg_sample = rng.choice(neg_idx, size=min(400_000, len(neg_idx)), replace=False)
keep_idx = pos_idx.union(pd.Index(neg_sample))
X_train_ds = X_train.loc[keep_idx]
y_train_ds = y_train.loc[keep_idx]
log(f"Downsampled train set: {len(X_train_ds):,} rows ({y_train_ds.sum():,} positive)")


# ---------------------------------------------------------------------------
# 9. Evaluation helper (unchanged)
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
# 10. Train models across seeds
# ---------------------------------------------------------------------------
import xgboost as xgb
import lightgbm as lgb

all_results = {"xgboost": [], "lightgbm": []}
feature_importance = {"xgboost": None, "lightgbm": None}

for seed in SEEDS:
    log(f"--- Seed {seed}: XGBoost ---")
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
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    X_train_lgb = X_train_ds.copy()
    X_test_lgb = X_test.copy()
    lgb_model.fit(X_train_lgb, y_train_ds,
                  categorical_feature=["PaymentFormat", "FromCountry", "ToCountry"])
    scores = lgb_model.predict_proba(X_test_lgb)[:, 1]
    all_results["lightgbm"].append(evaluate(y_test, scores, f"lightgbm_seed{seed}"))
    if feature_importance["lightgbm"] is None:
        feature_importance["lightgbm"] = dict(
            zip(FEATURES, lgb_model.feature_importances_.tolist())
        )

# ---------------------------------------------------------------------------
# 11. Aggregate + save
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
    "features": FEATURES,
    "new_features": ["FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank"],
    "self_loop_excluded_from_aggregates": True,
    "seeds": SEEDS,
    "raw_results": all_results,
    "summary": {
        "xgboost": summarize(all_results["xgboost"]),
        "lightgbm": summarize(all_results["lightgbm"]),
    },
    "feature_importance": feature_importance,
}

with open(OUT_DIR / "results_v2.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_v2.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
