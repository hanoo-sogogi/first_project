"""
Cross-dataset generalization: train the pair_only GBT pipeline on one
AMLworld "world" (one Kaggle dataset file) and score it on a DIFFERENT
world's own held-out test split.

Answers: does HI(train)->LI(test) generalize better than LI(train)->HI(test)?
Does HI-Medium(train)->HI-Small(test) (cross-scale) work at all?

Key design point (see AML_실험기록.md 3.16): pair_prior_count/FromIsNew/
ToIsNew are computed from within-dataset history. HI-Medium and LI-Medium
have disjoint account universes, so a source dataset's history is useless
for a target dataset's accounts. Each dataset is therefore preprocessed
INDEPENDENTLY end-to-end (its own train-period graph aggregates), and only
the trained MODEL WEIGHTS travel from source to target -- scoring always
uses the target's own leak-free features. This mirrors real deployment:
a model trained on dataset A scores a live stream from system B using only
B's own accumulated history.

HI-Large/LI-Large are excluded from this round (17GB each; the joint
account-factorize step that works fine at Medium scale (32M rows -> 3.86GB
peak, see train_medium.py) would need ~180M rows -> a much bigger temporary
string array and risks exceeding this machine's 15.6GB RAM). Doing Large
safely needs a chunked/incremental factorization, not yet implemented.
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
DATASETS = ["HI-Medium", "LI-Medium", "HI-Small"]
SEEDS = [1, 2, 3]
NEG_CAP = 400_000

t0 = time.time()
proc = psutil.Process()


def log(msg):
    mem_gb = proc.memory_info().rss / 1e9
    print(f"[{time.time() - t0:7.1f}s | proc {mem_gb:5.2f}GB] {msg}", flush=True)


COLS = ["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount",
        "AmountReceived", "ReceivingCurrency", "AmountPaid", "PaymentCurrency",
        "PaymentFormat", "IsLaundering"]
DTYPES = {
    "FromBankID": "int32", "ToBankID": "int32",
    "AmountReceived": "float32", "AmountPaid": "float32",
    "ReceivingCurrency": "category", "PaymentCurrency": "category",
    "PaymentFormat": "category", "IsLaundering": "int8",
}
FEATURES = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek",
            "FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank",
            "pair_prior_count"]
CAT_COLS = ["PaymentFormat", "FromCountry", "ToCountry"]
FORMAT_VOCAB = ["Cheque", "Credit Card", "ACH", "Cash", "Reinvestment", "Wire", "Bitcoin"]

# ---------------------------------------------------------------------------
# Step 0: build a shared FromCountry/ToCountry vocabulary across all 3
# datasets so a model trained on one dataset's categorical codes can safely
# score another dataset's data (mismatched category sets would silently
# miscode otherwise).
# ---------------------------------------------------------------------------
log("Building shared country vocabulary across datasets...")
country_vocab = {"Unknown"}
for name in DATASETS:
    accounts = pd.read_csv(DATA_DIR / f"{name}_accounts.csv", usecols=["Bank Name"])
    c = accounts["Bank Name"].str.extract(r"^([A-Za-z ]+?) Bank #\d+$")[0].fillna("Unknown")
    country_vocab.update(c.unique().tolist())
    del accounts
COUNTRY_VOCAB = sorted(country_vocab)
FORMAT_DTYPE = pd.CategoricalDtype(categories=FORMAT_VOCAB)
COUNTRY_DTYPE = pd.CategoricalDtype(categories=COUNTRY_VOCAB)
log(f"Shared vocab: {len(COUNTRY_VOCAB)} countries, {len(FORMAT_VOCAB)} payment formats")


def preprocess(name):
    """Full leak-free pair_only pipeline for one dataset, self-contained.
    Returns dict with X_train_ds, y_train_ds, X_test, y_test."""
    log(f"[{name}] loading...")
    df = pd.read_csv(DATA_DIR / f"{name}_Trans.csv", header=0, names=COLS, dtype=DTYPES)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    log(f"[{name}] loaded {n:,} rows, {int(df['IsLaundering'].sum()):,} laundering")

    combined_accounts = pd.concat(
        [df["FromAccount"].astype(str), df["ToAccount"].astype(str)], ignore_index=True
    )
    acct_codes, acct_uniques = pd.factorize(combined_accounts, sort=False)
    del combined_accounts
    gc.collect()
    df["FromAcctCode"] = acct_codes[:n].astype(np.int32)
    df["ToAcctCode"] = acct_codes[n:].astype(np.int32)
    del acct_codes
    pack = np.int64(len(acct_uniques) + 1)
    del acct_uniques
    df["FromNode"] = df["FromBankID"].astype(np.int64) * pack + df["FromAcctCode"].astype(np.int64)
    df["ToNode"] = df["ToBankID"].astype(np.int64) * pack + df["ToAcctCode"].astype(np.int64)
    df = df.drop(columns=["FromAccount", "ToAccount", "FromAcctCode", "ToAcctCode"])
    gc.collect()

    fx = df.loc[df["ReceivingCurrency"] != df["PaymentCurrency"],
                ["PaymentCurrency", "ReceivingCurrency", "AmountReceived", "AmountPaid"]].copy()
    fx["rate_recv_per_paid"] = fx["AmountReceived"].astype(np.float64) / fx["AmountPaid"].astype(np.float64)
    pair_rate = fx.groupby(["PaymentCurrency", "ReceivingCurrency"], observed=True)["rate_recv_per_paid"].median().to_dict()
    del fx
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
    df["LogAmountUSD"] = np.log1p(df["AmountPaid"].astype(np.float64) * df["PaymentCurrency"].map(usd_per).astype(np.float64)).astype(np.float32)
    df = df.drop(columns=["AmountReceived", "AmountPaid"])

    df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(np.int8)
    df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(np.int8)
    df["Hour"] = df["Timestamp"].dt.hour.astype(np.int8)
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype(np.int8)
    df["PaymentFormat"] = df["PaymentFormat"].astype(str).astype(FORMAT_DTYPE)

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

    accounts = pd.read_csv(DATA_DIR / f"{name}_accounts.csv", usecols=["Bank ID", "Bank Name"])
    bank_country = accounts.drop_duplicates("Bank ID").copy()
    bank_country["Country"] = bank_country["Bank Name"].str.extract(
        r"^([A-Za-z ]+?) Bank #\d+$")[0].fillna("Unknown")
    bank_id_to_country = dict(zip(bank_country["Bank ID"].astype(np.int32), bank_country["Country"]))
    del accounts, bank_country
    df["FromCountry"] = (df["FromBankID"].map(bank_id_to_country).fillna("Unknown")).astype(COUNTRY_DTYPE)
    df["ToCountry"] = (df["ToBankID"].map(bank_id_to_country).fillna("Unknown")).astype(COUNTRY_DTYPE)
    df["IsCrossBorderBank"] = (df["FromCountry"] != df["ToCountry"]).astype(np.int8)
    gc.collect()

    t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
    t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
    train_mask = (df["Timestamp"] < t1).to_numpy()
    test_mask = (df["Timestamp"] >= t2).to_numpy()

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

    y_all = df["IsLaundering"].to_numpy()
    X_test = df.loc[test_mask, FEATURES].reset_index(drop=True)
    y_test = y_all[test_mask]
    X_train_full = df.loc[train_mask, FEATURES].reset_index(drop=True)
    y_train_full = y_all[train_mask]
    del df
    gc.collect()

    rng = np.random.RandomState(0)
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    neg_sample = rng.choice(neg_idx, size=min(NEG_CAP, len(neg_idx)), replace=False)
    keep_idx = np.concatenate([pos_idx, neg_sample])
    X_train_ds = X_train_full.iloc[keep_idx].reset_index(drop=True)
    y_train_ds = y_train_full[keep_idx]
    del X_train_full, y_train_full
    gc.collect()
    log(f"[{name}] ready: train_ds {len(X_train_ds):,} ({int(y_train_ds.sum()):,} pos), test {len(X_test):,} ({int(y_test.sum()):,} pos)")
    return {"X_train_ds": X_train_ds, "y_train_ds": y_train_ds, "X_test": X_test, "y_test": y_test}


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


# ---------------------------------------------------------------------------
# Preprocess every dataset once
# ---------------------------------------------------------------------------
data = {}
for name in DATASETS:
    data[name] = preprocess(name)
    with open(OUT_DIR / "_cross_ds_checkpoint.json", "w") as f:
        json.dump({"done": list(data.keys())}, f)

import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Train GBT (both models, 3 seeds) on each SOURCE dataset
# ---------------------------------------------------------------------------
models = {}
for source in DATASETS:
    Xtr, ytr = data[source]["X_train_ds"], data[source]["y_train_ds"]
    for seed in SEEDS:
        log(f"=== Training on {source}, seed {seed} ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            tree_method="hist", enable_categorical=True, random_state=seed,
        )
        xgb_model.fit(Xtr, ytr)
        models[(source, seed, "xgb")] = xgb_model

        lgb_model = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=15, min_child_samples=50,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, verbosity=-1,
        )
        lgb_model.fit(Xtr.copy(), ytr, categorical_feature=CAT_COLS)
        models[(source, seed, "lgb")] = lgb_model

# ---------------------------------------------------------------------------
# Score every trained model against every dataset's own test split
# (diagonal = within-dataset baseline, off-diagonal = cross-dataset transfer)
# ---------------------------------------------------------------------------
results = []
for source in DATASETS:
    for target in DATASETS:
        Xte, yte = data[target]["X_test"], data[target]["y_test"]
        for seed in SEEDS:
            for mtype in ("xgb", "lgb"):
                model = models[(source, seed, mtype)]
                scores = model.predict_proba(Xte)[:, 1]
                tag = "within" if source == target else "cross"
                r = evaluate(yte, scores, f"{source}_to_{target}[{tag}]_{mtype}_s{seed}")
                r["source"] = source
                r["target"] = target
                r["model_type"] = mtype
                r["seed"] = seed
                r["kind"] = tag
                results.append(r)

with open(OUT_DIR / "results_cross_dataset.json", "w", encoding="utf-8") as f:
    json.dump({"datasets": DATASETS, "features": FEATURES, "results": results}, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_cross_dataset.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
