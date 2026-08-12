"""
Test the revised hypothesis from 3.20: the hub-involved blind spot (F1=0.0000
in every cell, even with hub-relative features) is a training-set class-balance
problem, not a feature problem. Uniform random negative downsampling keeps
hub-involved negatives at their full-population share (~9%) while hub-involved
positives are a tiny fraction of all positives (~2.6%) -- so the hub subgroup's
effective positive rate in training is LOWER than the overall training set's,
giving trees no incentive to branch on it.

Fix tested here: downsample hub-involved NEGATIVES much more aggressively than
non-hub negatives (keep only HUB_NEG_RATIO x hub-positive-count), while keeping
ALL positives as before (hub and non-hub) and the same overall NEG_CAP budget
for non-hub negatives. This directly inflates the hub subgroup's local positive
rate in the training set without touching the test set at all.

Uses the hub_dev feature set (pair_only + IsHubInvolved + HubAmountDeviation)
from 3.20, since that's the richest feature set available -- if rebalancing
alone doesn't help even with those features present, that's a strong signal
the problem is deeper than either lever alone.
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"C:\Users\aica_\Documents\CLaude")
OUT_DIR = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
DATASETS = ["HI-Medium", "LI-Medium", "HI-Small"]
MERGE_SOURCES = ["HI-Medium", "LI-Medium"]
SEEDS = [1, 2, 3]
NEG_CAP = 400_000
HUB_BANK_ID = 70
HUB_NEG_RATIO = 5  # keep only this many hub-negatives per hub-positive, instead of ~90:1 natural share

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
            "pair_prior_count", "IsHubInvolved", "HubAmountDeviation"]
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


def parse_patterns(name):
    path = DATA_DIR / f"{name}_Patterns.txt"
    pattern_type = None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                pattern_type = line.split(" - ", 1)[1].split(":")[0].strip()
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                pattern_type = None
                continue
            parts = line.split(",")
            ts, fbank, facct, tbank, tacct, _ar, _cr, _ap, _cp, _fmt, _lbl = parts
            rows.append((ts, int(fbank), facct, int(tbank), tacct, pattern_type))
    pdf = pd.DataFrame(rows, columns=["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount", "PatternType"])
    pdf["Timestamp"] = pd.to_datetime(pdf["Timestamp"], format="%Y/%m/%d %H:%M")
    pdf = pdf.drop_duplicates(subset=["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount"])
    return pdf


def preprocess(name):
    log(f"[{name}] loading...")
    df = pd.read_csv(DATA_DIR / f"{name}_Trans.csv", header=0, names=COLS, dtype=DTYPES)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    log(f"[{name}] loaded {n:,} rows, {int(df['IsLaundering'].sum()):,} laundering")

    is_from_hub = (df["FromBankID"] == HUB_BANK_ID) & (df["FromAccount"].astype(str).str[0] != "8")
    is_to_hub = (df["ToBankID"] == HUB_BANK_ID) & (df["ToAccount"].astype(str).str[0] != "8")
    df["IsHubInvolved"] = (is_from_hub | is_to_hub).astype(np.int8)

    pdf = parse_patterns(name)
    n_before = len(df)
    df = df.merge(pdf, on=["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount"], how="left")
    assert len(df) == n_before
    df["InPattern"] = df["PatternType"].notna().astype(np.int8)
    log(f"[{name}] hub-involved rows: {int(df['IsHubInvolved'].sum()):,}, "
        f"in-pattern laundering: {int(df['InPattern'].sum()):,}/{int(df['IsLaundering'].sum()):,}")

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
    df["HubNode"] = np.where(is_from_hub, df["FromNode"], np.where(is_to_hub, df["ToNode"], -1)).astype(np.int64)
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

    hub_train_mask = train_mask & (df["HubNode"].to_numpy() >= 0)
    hub_median = df.loc[hub_train_mask].groupby("HubNode")["LogAmountUSD"].median()
    hub_rows = df["HubNode"] >= 0
    df["HubAmountDeviation"] = np.float32(0.0)
    df.loc[hub_rows, "HubAmountDeviation"] = (
        df.loc[hub_rows, "LogAmountUSD"] - df.loc[hub_rows, "HubNode"].map(hub_median)
    ).fillna(0.0).astype(np.float32)

    y_all = df["IsLaundering"].to_numpy()
    X_test = df.loc[test_mask, FEATURES].reset_index(drop=True)
    y_test = y_all[test_mask]
    X_train_full = df.loc[train_mask, FEATURES].reset_index(drop=True)
    y_train_full = y_all[train_mask]
    del df
    gc.collect()

    rng = np.random.RandomState(0)
    hub_flag_train = X_train_full["IsHubInvolved"].to_numpy().astype(bool)
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]

    hub_pos_count = int(hub_flag_train[pos_idx].sum())
    hub_neg_idx = neg_idx[hub_flag_train[neg_idx]]
    nonhub_neg_idx = neg_idx[~hub_flag_train[neg_idx]]

    hub_neg_keep_n = min(len(hub_neg_idx), max(HUB_NEG_RATIO * hub_pos_count, 1))
    hub_neg_sample = rng.choice(hub_neg_idx, size=hub_neg_keep_n, replace=False)
    nonhub_neg_budget = max(NEG_CAP - hub_neg_keep_n, 0)
    nonhub_neg_sample = rng.choice(nonhub_neg_idx, size=min(nonhub_neg_budget, len(nonhub_neg_idx)), replace=False)

    keep_idx = np.concatenate([pos_idx, hub_neg_sample, nonhub_neg_sample])
    X_train_ds = X_train_full.iloc[keep_idx].reset_index(drop=True)
    y_train_ds = y_train_full[keep_idx]
    del X_train_full, y_train_full
    gc.collect()
    hub_pos_rate = hub_pos_count / max(hub_pos_count + hub_neg_keep_n, 1)
    log(f"[{name}] train_ds {len(X_train_ds):,} ({int(y_train_ds.sum()):,} pos) | "
        f"hub subgroup in train_ds: {hub_pos_count} pos / {hub_neg_keep_n} neg kept "
        f"(natural hub-neg pool {len(hub_neg_idx):,}) -> hub pos rate {100*hub_pos_rate:.1f}% "
        f"(overall train_ds pos rate {100*y_train_ds.mean():.1f}%)")
    log(f"[{name}] ready: test {len(X_test):,} ({int(y_test.sum()):,} pos)")
    return {"X_train_ds": X_train_ds, "y_train_ds": y_train_ds, "X_test": X_test, "y_test": y_test}


def stratum_metrics(y_true, scores, mask, label):
    y_s = np.asarray(y_true)[mask]
    s_s = np.asarray(scores)[mask]
    n_pos = int(y_s.sum())
    if n_pos == 0 or mask.sum() == 0:
        return {"stratum": label, "n": int(mask.sum()), "n_pos": n_pos, "note": "no positives"}
    preds = (s_s >= 0.5).astype(int)
    return {
        "stratum": label, "n": int(mask.sum()), "n_pos": n_pos,
        "f1": float(f1_score(y_s, preds)),
        "precision": float(precision_score(y_s, preds, zero_division=0)),
        "recall": float(recall_score(y_s, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_s, s_s)) if n_pos < len(y_s) else float("nan"),
    }


data = {}
for name in DATASETS:
    data[name] = preprocess(name)

X_train_merged = pd.concat([data[s]["X_train_ds"] for s in MERGE_SOURCES], ignore_index=True)
y_train_merged = np.concatenate([data[s]["y_train_ds"] for s in MERGE_SOURCES])
log(f"Merged train set: {len(X_train_merged):,} rows, {int(y_train_merged.sum()):,} pos")

import xgboost as xgb
import lightgbm as lgb

results = []
for seed in SEEDS:
    log(f"=== Training rebalanced model, seed {seed} ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", enable_categorical=True, random_state=seed,
    )
    xgb_model.fit(X_train_merged, y_train_merged)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=15, min_child_samples=50,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    lgb_model.fit(X_train_merged.copy(), y_train_merged, categorical_feature=CAT_COLS)

    for target_name in DATASETS:
        Xte = data[target_name]["X_test"]
        yte = data[target_name]["y_test"]
        hub_mask = Xte["IsHubInvolved"].to_numpy().astype(bool)

        for mtype, model in (("xgb", xgb_model), ("lgb", lgb_model)):
            scores = model.predict_proba(Xte)[:, 1]
            pooled = stratum_metrics(yte, scores, np.ones(len(yte), dtype=bool), "pooled")
            hub_res = stratum_metrics(yte, scores, hub_mask, "hub_involved")
            non_hub_res = stratum_metrics(yte, scores, ~hub_mask, "non_hub")
            log(f"{target_name}_{mtype}_s{seed}: pooled F1={pooled.get('f1',float('nan')):.4f} "
                f"| hub F1={hub_res.get('f1',float('nan'))} P={hub_res.get('precision',float('nan'))} "
                f"R={hub_res.get('recall',float('nan'))} PR-AUC={hub_res.get('pr_auc',float('nan'))}")
            row = {"target": target_name, "model_type": mtype, "seed": seed,
                   "pooled": pooled, "hub_involved": hub_res, "non_hub": non_hub_res}
            results.append(row)
            with open(OUT_DIR / "results_hub_rebalanced.json", "w", encoding="utf-8") as f:
                json.dump({"hub_neg_ratio": HUB_NEG_RATIO, "features": FEATURES,
                           "results": results}, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_hub_rebalanced.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
