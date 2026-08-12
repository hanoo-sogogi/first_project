"""
Comparative EDA across all 6 IBM AMLworld Kaggle datasets (HI/LI x Small/Medium/Large),
in preparation for cross-dataset generalization experiments (train on one dataset's
"world", score a different dataset's "world" — e.g. HI-Large -> HI-Small, HI<->LI).

Memory-safe by design: HI-Large/LI-Large are ~17GB / 175-180M rows each, far too big
to load with pandas.read_csv() directly on a 15.6GB-RAM machine. Trans.csv files are
processed in chunks with only small running aggregates (counts, sums, sets of unique
account/bank IDs) kept in memory — the per-chunk DataFrame itself is discarded after
each chunk. accounts.csv and Patterns.txt are small (<150MB) and read directly.
"""
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

DATA_DIR = Path(r"C:\Users\aica_\Documents\CLaude")
OUT_DIR = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
DATASETS = ["HI-Small", "LI-Small", "HI-Medium", "LI-Medium", "HI-Large", "LI-Large"]
CHUNKSIZE = 5_000_000

t0 = time.time()
proc = psutil.Process()


def log(msg):
    mem_gb = proc.memory_info().rss / 1e9
    print(f"[{time.time() - t0:7.1f}s | proc {mem_gb:5.2f}GB] {msg}", flush=True)


COLS = ["Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount",
        "AmountReceived", "ReceivingCurrency", "AmountPaid", "PaymentCurrency",
        "PaymentFormat", "IsLaundering"]
DTYPES = {
    "FromBankID": "str", "ToBankID": "str",
    "FromAccount": "str", "ToAccount": "str",
    "AmountReceived": "float32", "AmountPaid": "float32",
    "ReceivingCurrency": "str", "PaymentCurrency": "str",
    "PaymentFormat": "str", "IsLaundering": "int8",
}


def eda_accounts(name):
    accounts = pd.read_csv(DATA_DIR / f"{name}_accounts.csv",
                            usecols=["Bank Name", "Bank ID", "Entity Name"], dtype=str)
    n_accounts = len(accounts)
    n_banks = accounts["Bank ID"].nunique()
    country = accounts["Bank Name"].str.extract(r"^([A-Za-z ]+?) Bank #\d+$")[0].fillna("Unknown")
    country_dist = country.value_counts().head(10).to_dict()
    entity_type = accounts["Entity Name"].str.extract(r"^([A-Za-z ]+?) #")[0].fillna("Unknown")
    entity_dist = entity_type.value_counts().to_dict()
    del accounts
    gc.collect()
    return {"n_accounts": int(n_accounts), "n_banks": int(n_banks),
            "top_countries": country_dist, "entity_type_dist": entity_dist}


def eda_patterns(name):
    path = DATA_DIR / f"{name}_Patterns.txt"
    if not path.exists():
        return {"error": "file not found"}
    pattern_counts = Counter()
    pattern_trans = Counter()
    cur_pattern = None
    cur_lines = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                m = re.search(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)", line)
                cur_pattern = m.group(1) if m else "UNKNOWN"
                cur_lines = 0
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if cur_pattern is not None:
                    pattern_counts[cur_pattern] += 1
                    pattern_trans[cur_pattern] += cur_lines
                cur_pattern = None
            elif cur_pattern is not None and line.strip():
                cur_lines += 1
    return {"pattern_instance_counts": dict(pattern_counts),
            "pattern_transaction_counts": dict(pattern_trans),
            "total_pattern_instances": sum(pattern_counts.values()),
            "total_pattern_transactions": sum(pattern_trans.values())}


def eda_transactions(name):
    path = DATA_DIR / f"{name}_Trans.csv"
    n_rows = 0
    n_pos = 0
    fmt_counts = Counter()
    pay_cur_counts = Counter()
    recv_cur_counts = Counter()
    self_loop = 0
    same_bank = 0
    cross_currency = 0
    amt_sum = 0.0
    amt_min = None
    amt_max = None
    ts_min = None
    ts_max = None
    accounts_seen = set()
    banks_seen = set()
    n_chunks = 0

    for chunk in pd.read_csv(path, header=0, names=COLS, dtype=DTYPES, chunksize=CHUNKSIZE):
        n_chunks += 1
        n_rows += len(chunk)
        n_pos += int(chunk["IsLaundering"].sum())
        fmt_counts.update(chunk["PaymentFormat"].value_counts().to_dict())
        pay_cur_counts.update(chunk["PaymentCurrency"].value_counts().to_dict())
        recv_cur_counts.update(chunk["ReceivingCurrency"].value_counts().to_dict())

        from_node = chunk["FromBankID"] + "_" + chunk["FromAccount"]
        to_node = chunk["ToBankID"] + "_" + chunk["ToAccount"]
        self_loop += int((from_node == to_node).sum())
        same_bank += int((chunk["FromBankID"] == chunk["ToBankID"]).sum())
        cross_currency += int((chunk["PaymentCurrency"] != chunk["ReceivingCurrency"]).sum())

        amt_sum += float(chunk["AmountPaid"].sum())
        cmin, cmax = float(chunk["AmountPaid"].min()), float(chunk["AmountPaid"].max())
        amt_min = cmin if amt_min is None else min(amt_min, cmin)
        amt_max = cmax if amt_max is None else max(amt_max, cmax)

        ts = pd.to_datetime(chunk["Timestamp"], format="%Y/%m/%d %H:%M")
        cts_min, cts_max = ts.min(), ts.max()
        ts_min = cts_min if ts_min is None else min(ts_min, cts_min)
        ts_max = cts_max if ts_max is None else max(ts_max, cts_max)

        # cap the running unique-account/bank sets' growth cost by sampling only
        # every chunk (sets naturally dedupe; final size bounded by true cardinality)
        accounts_seen.update(from_node.to_numpy())
        accounts_seen.update(to_node.to_numpy())
        banks_seen.update(chunk["FromBankID"].to_numpy())
        banks_seen.update(chunk["ToBankID"].to_numpy())

        del chunk, from_node, to_node, ts
        log(f"{name}: chunk {n_chunks} done, {n_rows:,} rows so far "
            f"({len(accounts_seen):,} unique accts seen)")

    days_spanned = (ts_max - ts_min).days + 1 if ts_min is not None else None
    return {
        "n_transactions": n_rows,
        "n_laundering": n_pos,
        "laundering_rate_pct": 100.0 * n_pos / n_rows if n_rows else None,
        "laundering_ratio_1_per_n": n_rows / n_pos if n_pos else None,
        "n_unique_accounts_in_trans": len(accounts_seen),
        "n_unique_banks_in_trans": len(banks_seen),
        "date_min": str(ts_min), "date_max": str(ts_max), "days_spanned": days_spanned,
        "payment_format_dist": dict(fmt_counts),
        "payment_currency_dist": dict(pay_cur_counts.most_common(10)),
        "receiving_currency_dist": dict(recv_cur_counts.most_common(10)),
        "self_loop_rate_pct": 100.0 * self_loop / n_rows if n_rows else None,
        "same_bank_rate_pct": 100.0 * same_bank / n_rows if n_rows else None,
        "cross_currency_rate_pct": 100.0 * cross_currency / n_rows if n_rows else None,
        "avg_amount_paid_raw": amt_sum / n_rows if n_rows else None,
        "min_amount_paid": amt_min, "max_amount_paid": amt_max,
    }


results = {}
for name in DATASETS:
    log(f"=== {name} ===")
    results[name] = {
        "accounts": eda_accounts(name),
        "patterns": eda_patterns(name),
        "transactions": eda_transactions(name),
    }
    with open(OUT_DIR / "eda_compare_datasets.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log(f"{name} done, results checkpointed")
    gc.collect()

log("All datasets done.")
log(f"Total runtime: {time.time() - t0:.1f}s")
