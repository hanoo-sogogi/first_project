"""
Pattern-membership / hub-involvement stratified evaluation, per the success-metric
design in the team architecture doc (jiwon, section 4): "1단계 모델... 패턴 내부/외부와
허브 관여 여부를 분리 평가". All prior results (3.14-3.17) report a single pooled
PR-AUC/F1 -- this breaks that down into 4 strata:
  - in a named pattern (FAN-OUT/CYCLE/GATHER-SCATTER/STACK/RANDOM/BIPARTITE/
    FAN-IN/SCATTER-GATHER) vs. "normal-embedded" laundering (not in any of the 8)
  - hub account involved (FromBankID==70 or ToBankID==70, account number's first
    char != '8' -- confirmed against our own accounts.csv; see ML-08 in the team
    EDA doc) vs. not

Uses the 3.17 merged model (train on HI-Medium+LI-Medium) since that's the
current best production candidate. Reports pooled + per-stratum metrics for
each target dataset's test split.
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
            ts, fbank, facct, tbank, tacct, _amt_recv, _cur_recv, amt_paid, cur_paid, _fmt, _lbl = parts
            rows.append((ts, int(fbank), facct, int(tbank), tacct, float(amt_paid), cur_paid, pattern_type))
    pdf = pd.DataFrame(rows, columns=["Timestamp", "FromBankID", "FromAccount", "ToBankID",
                                       "ToAccount", "AmountPaid", "PaymentCurrency", "PatternType"])
    pdf["Timestamp"] = pd.to_datetime(pdf["Timestamp"], format="%Y/%m/%d %H:%M")
    # Join key deliberately excludes AmountPaid/PaymentCurrency: df reads AmountPaid as
    # float32 while this parses it as float64 from text, so float32's precision loss on
    # large amounts breaks exact matches. Timestamp+accounts alone is already unique
    # enough (this is a same-minute directed edge between two specific accounts).
    pdf = pdf.drop_duplicates(subset=["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount"])
    return pdf


def preprocess(name):
    log(f"[{name}] loading...")
    df = pd.read_csv(DATA_DIR / f"{name}_Trans.csv", header=0, names=COLS, dtype=DTYPES)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    log(f"[{name}] loaded {n:,} rows, {int(df['IsLaundering'].sum()):,} laundering")

    # hub flag + pattern-membership must happen BEFORE account strings are dropped
    df["IsHubInvolved"] = (
        ((df["FromBankID"] == HUB_BANK_ID) & (df["FromAccount"].astype(str).str[0] != "8")) |
        ((df["ToBankID"] == HUB_BANK_ID) & (df["ToAccount"].astype(str).str[0] != "8"))
    ).astype(np.int8)

    pdf = parse_patterns(name).drop(columns=["AmountPaid", "PaymentCurrency"])
    n_before = len(df)
    df = df.merge(pdf, on=["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount"], how="left")
    assert len(df) == n_before, f"merge fan-out detected: {len(df)} vs {n_before}"
    df["InPattern"] = df["PatternType"].notna().astype(np.int8)
    n_matched = int(df["InPattern"].sum())
    n_pos = int(df["IsLaundering"].sum())
    log(f"[{name}] pattern match: {n_matched:,}/{n_pos:,} laundering rows matched to a named pattern "
        f"({100*n_matched/max(n_pos,1):.1f}%), hub-involved rows: {int(df['IsHubInvolved'].sum()):,}")

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
    strat_cols = FEATURES + ["IsHubInvolved", "InPattern"]
    X_test = df.loc[test_mask, strat_cols].reset_index(drop=True)
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


def stratum_metrics(y_true, scores, mask, label):
    y_s = np.asarray(y_true)[mask]
    s_s = np.asarray(scores)[mask]
    n_pos = int(y_s.sum())
    if n_pos == 0 or mask.sum() == 0:
        return {"stratum": label, "n": int(mask.sum()), "n_pos": n_pos, "note": "no positives, metrics undefined"}
    preds = (s_s >= 0.5).astype(int)
    result = {
        "stratum": label, "n": int(mask.sum()), "n_pos": n_pos,
        "f1": float(f1_score(y_s, preds)),
        "precision": float(precision_score(y_s, preds, zero_division=0)),
        "recall": float(recall_score(y_s, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_s, s_s)) if n_pos < len(y_s) else float("nan"),
    }
    return result


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
    log(f"=== Training merged model, seed {seed} ===")
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
            log(f"{target_name}_{mtype}_s{seed} pooled: F1={pooled.get('f1',float('nan')):.4f} PR-AUC={pooled.get('pr_auc',float('nan')):.4f}")
            for k, v in strata.items():
                if "f1" in v:
                    log(f"  {k}: n={v['n']} n_pos={v['n_pos']} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} PR-AUC={v['pr_auc']:.4f}")
                else:
                    log(f"  {k}: n={v['n']} n_pos={v['n_pos']} (no positives)")
            row = {"target": target_name, "model_type": mtype, "seed": seed, "pooled": pooled, **strata}
            results.append(row)
            with open(OUT_DIR / "results_stratified.json", "w", encoding="utf-8") as f:
                json.dump({"merge_sources": MERGE_SOURCES, "hub_bank_id": HUB_BANK_ID,
                           "results": results}, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_stratified.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
