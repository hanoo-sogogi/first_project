"""
Test whether hub-aware features close the hub-involved blind spot found in 3.19
(pair_only merged model: F1=0.0000 on hub-involved laundering, 18/18 cells).

Two candidate feature sets, both added on top of pair_only (12 features):
  - hub_flag: + IsHubInvolved (binary) -- the minimal fix (architecture doc's 처리안 C)
  - hub_dev:  + IsHubInvolved + HubAmountDeviation -- tests the sharper hypothesis
    that a bare flag isn't enough because pair_prior_count is saturated for hub
    edges (huge count regardless of legit/illicit); HubAmountDeviation asks "is
    THIS transaction's amount unusual for THIS specific hub account's own
    baseline", computed causally from that hub's own train-period transactions.

Same merged (HI-Medium+LI-Medium) training + stratified evaluation protocol as
3.17/3.19, so results are directly comparable to the pair_only baseline already
recorded (in_pattern/normal_embedded/hub_involved/non_hub, per target dataset).
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
BASE_FEATURES = ["LogAmountUSD", "PaymentFormat", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek",
                  "FromIsNew", "ToIsNew", "FromCountry", "ToCountry", "IsCrossBorderBank",
                  "pair_prior_count"]
FEATURE_SETS = {
    "hub_flag": BASE_FEATURES + ["IsHubInvolved"],
    "hub_dev": BASE_FEATURES + ["IsHubInvolved", "HubAmountDeviation"],
}
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
    # which specific hub node is involved (for the per-hub causal amount baseline below)
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

    # causal per-hub amount baseline: each hub's own median LogAmountUSD, computed
    # from TRAIN-period transactions touching that hub only (leak-free, same
    # principle as pair_prior_count above)
    hub_train_mask = train_mask & (df["HubNode"].to_numpy() >= 0)
    hub_median = df.loc[hub_train_mask].groupby("HubNode")["LogAmountUSD"].median()
    hub_rows = df["HubNode"] >= 0
    df["HubAmountDeviation"] = np.float32(0.0)
    df.loc[hub_rows, "HubAmountDeviation"] = (
        df.loc[hub_rows, "LogAmountUSD"] - df.loc[hub_rows, "HubNode"].map(hub_median)
    ).fillna(0.0).astype(np.float32)
    log(f"[{name}] hub baseline computed from {len(hub_median)} hub accounts seen in train period")

    y_all = df["IsLaundering"].to_numpy()
    all_extra_cols = sorted(set(sum(FEATURE_SETS.values(), [])) | {"IsHubInvolved", "InPattern"})
    X_test = df.loc[test_mask, all_extra_cols].reset_index(drop=True)
    y_test = y_all[test_mask]
    X_train_full = df.loc[train_mask, all_extra_cols].reset_index(drop=True)
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

import xgboost as xgb
import lightgbm as lgb

results = []
for fset_name, FEATURES in FEATURE_SETS.items():
    cat_cols_here = [c for c in CAT_COLS if c in FEATURES]
    X_train_merged = pd.concat([data[s]["X_train_ds"][FEATURES] for s in MERGE_SOURCES], ignore_index=True)
    y_train_merged = np.concatenate([data[s]["y_train_ds"] for s in MERGE_SOURCES])
    log(f"=== Feature set {fset_name} ({len(FEATURES)} features): merged train {len(X_train_merged):,} rows ===")

    for seed in SEEDS:
        log(f"--- Training {fset_name}, seed {seed} ---")
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
        lgb_model.fit(X_train_merged.copy(), y_train_merged, categorical_feature=cat_cols_here)

        for target_name in DATASETS:
            Xte_full = data[target_name]["X_test"]
            yte = data[target_name]["y_test"]
            Xte = Xte_full[FEATURES]
            hub_mask = Xte_full["IsHubInvolved"].to_numpy().astype(bool)
            pat_mask = Xte_full["InPattern"].to_numpy().astype(bool)

            for mtype, model in (("xgb", xgb_model), ("lgb", lgb_model)):
                scores = model.predict_proba(Xte)[:, 1]
                pooled = stratum_metrics(yte, scores, np.ones(len(yte), dtype=bool), "pooled")
                strata = {
                    "in_pattern": stratum_metrics(yte, scores, pat_mask, "in_pattern"),
                    "normal_embedded": stratum_metrics(yte, scores, ~pat_mask, "normal_embedded"),
                    "hub_involved": stratum_metrics(yte, scores, hub_mask, "hub_involved"),
                    "non_hub": stratum_metrics(yte, scores, ~hub_mask, "non_hub"),
                }
                hub_f1 = strata["hub_involved"].get("f1", float("nan"))
                log(f"{fset_name}_{target_name}_{mtype}_s{seed}: pooled F1={pooled.get('f1',float('nan')):.4f} "
                    f"| hub_involved F1={hub_f1}")
                row = {"feature_set": fset_name, "target": target_name, "model_type": mtype,
                       "seed": seed, "pooled": pooled, **strata}
                results.append(row)
                with open(OUT_DIR / "results_hub_features.json", "w", encoding="utf-8") as f:
                    json.dump({"feature_sets": FEATURE_SETS, "hub_bank_id": HUB_BANK_ID,
                               "results": results}, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_hub_features.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
