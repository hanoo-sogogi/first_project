"""
Capstone GBT-track experiment: combine every validated improvement from today
into one final candidate model before handing the track to GNN.
  - 3-way merged training set: HI-Medium + LI-Medium + HI-Small (3.17 used only
    the first two; this extends it with HI-Small, previously held out as the
    "unseen world" bonus check -- now folded into training since we've already
    used it that way once and want the best achievable pooled model)
  - Multi-seed ensemble: 6 seeds each for XGBoost and LightGBM (12 models total),
    matching the 3.12 ensemble methodology (combined_all12)
  - Calibration-split threshold: reuse the 60-80% window as calibration (3.18),
    but now find the threshold on the ENSEMBLE-AVERAGED probability, not a
    single model's score

Evaluated per target dataset (HI-Medium/LI-Medium/HI-Small) at both the default
0.5 threshold and the calibrated threshold, for three ensemble variants:
xgb_ens6 (6 XGBoost seeds averaged), lgb_ens6 (6 LightGBM seeds averaged),
combined_all12 (all 12 averaged).
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
MERGE_SOURCES = ["HI-Medium", "LI-Medium", "HI-Small"]  # 3-way now
SEEDS = [1, 2, 3, 4, 5, 6]
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
    calib_mask = ((df["Timestamp"] >= t1) & (df["Timestamp"] < t2)).to_numpy()
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
    X_calib = df.loc[calib_mask, FEATURES].reset_index(drop=True)
    y_calib = y_all[calib_mask]
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
    log(f"[{name}] ready: train_ds {len(X_train_ds):,} ({int(y_train_ds.sum()):,} pos), "
        f"calib {len(X_calib):,} ({int(y_calib.sum()):,} pos), "
        f"test {len(X_test):,} ({int(y_test.sum()):,} pos)")
    return {"X_train_ds": X_train_ds, "y_train_ds": y_train_ds,
            "X_calib": X_calib, "y_calib": y_calib,
            "X_test": X_test, "y_test": y_test}


def best_f1_threshold(y_true, scores):
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    if len(f1) == 0 or not np.isfinite(f1).any():
        return 0.5, 0.0
    best_idx = int(np.nanargmax(f1))
    return float(thr[best_idx]), float(f1[best_idx])


def evaluate(y_true, scores, threshold, name):
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)
    result = {"model": name, "threshold": float(threshold), "f1": float(f1),
              "precision": float(prec), "recall": float(rec), "pr_auc": float(pr_auc)}
    log(f"{name} @thr={threshold:.3f}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} PR-AUC={pr_auc:.4f}")
    return result


data = {}
for name in DATASETS:
    data[name] = preprocess(name)

X_train_merged = pd.concat([data[s]["X_train_ds"] for s in MERGE_SOURCES], ignore_index=True)
y_train_merged = np.concatenate([data[s]["y_train_ds"] for s in MERGE_SOURCES])
log(f"3-way merged train set ({'+'.join(MERGE_SOURCES)}): {len(X_train_merged):,} rows, "
    f"{int(y_train_merged.sum()):,} pos ({100*y_train_merged.mean():.2f}%)")

import xgboost as xgb
import lightgbm as lgb

xgb_models, lgb_models = [], []
for seed in SEEDS:
    log(f"=== Training seed {seed} ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_merged, y_train_merged)
    xgb_models.append(xgb_model)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_merged.copy(), y_train_merged, categorical_feature=CAT_COLS)
    lgb_models.append(lgb_model)

log("All 12 models trained. Scoring ensembles per target...")

ENSEMBLES = {
    "xgb_ens6": xgb_models,
    "lgb_ens6": lgb_models,
    "combined_all12": xgb_models + lgb_models,
}

results = []
for target_name in DATASETS:
    Xcal, ycal = data[target_name]["X_calib"], data[target_name]["y_calib"]
    Xte, yte = data[target_name]["X_test"], data[target_name]["y_test"]
    for ens_name, models in ENSEMBLES.items():
        cal_scores = np.mean([m.predict_proba(Xcal)[:, 1] for m in models], axis=0)
        calib_thr, calib_f1 = best_f1_threshold(ycal, cal_scores)

        test_scores = np.mean([m.predict_proba(Xte)[:, 1] for m in models], axis=0)
        r_default = evaluate(yte, test_scores, 0.5, f"{target_name}_{ens_name}_thr0.5")
        r_calib = evaluate(yte, test_scores, calib_thr, f"{target_name}_{ens_name}_thrCAL")
        row = {"target": target_name, "ensemble": ens_name, "n_models": len(models),
               "calib_threshold": calib_thr, "calib_set_f1": calib_f1,
               "default": r_default, "calibrated": r_calib}
        results.append(row)
        with open(OUT_DIR / "results_final_ensemble.json", "w", encoding="utf-8") as f:
            json.dump({"merge_sources": MERGE_SOURCES, "seeds": SEEDS, "features": FEATURES,
                       "results": results}, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_final_ensemble.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
