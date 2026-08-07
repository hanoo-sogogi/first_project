"""
Preprocess IBM AML transaction data (AMLworld / Kaggle ealtman2019) for the
GBT + Graph Feature Preprocessor (GFP) pipeline described in:

  Altman et al., "Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models", NeurIPS 2023 (arXiv:2306.16424)

Pipeline (matches paper Sections 3.5, 4 and Appendix D):
  1. Load raw transaction CSV (Timestamp, From/To Bank, Account, Amounts,
     Currencies, Payment Format, Is Laundering).
  2. Encode categorical fields and derive basic transaction-level features.
  3. Map (Bank, Account) pairs to a single integer vertex ID space, shared
     between source and target endpoints (required by GFP's edge-list format).
  4. Sort transactions by timestamp (required by GFP) and run IBM's
     GraphFeaturePreprocessor (snapml) to compute graph-based features:
     scatter-gather patterns, temporal cycles, length-constrained simple
     cycles (<=10 hops), and vertex statistics -- using the same time
     windows as the paper (6h for scatter-gather, 1 day for the rest).
  5. Apply a temporal 60/20/20 train/val/test split (by transaction order,
     since transactions are already timestamp-sorted -- same scheme as
     Section 4 "Data Split").
  6. Persist train/val/test Parquet files plus a metadata JSON (feature
     names, category mappings, split boundaries) ready for
     LightGBM / XGBoost training.

Notes / deviations from the paper made explicit here:
  - The paper streams edges through GFP in batches of 128 for its online
    simulation; we instead call `fit_transform` once on the whole
    (already time-sorted) array. Because GFP's sliding time windows only
    ever look backward in time, this is numerically equivalent to batched
    streaming for a static, already-sorted dataset. Set --gfp-batch-size to
    process in chunks via partial_fit/transform instead, e.g. for memory
    reasons on the Large datasets.
  - Source/target account IDs are used only to build the transaction graph
    for GFP; they are dropped from the final ML feature matrix (paper:
    "Source and destination account IDs ... are not used as features").
  - IBM does not publicly document the exact per-column semantics of GFP's
    output feature matrix beyond the feature families it computes (fan-in,
    fan-out, degree-in, degree-out, scatter-gather, temporal cycle,
    length-constrained simple cycles, vertex statistics), so output columns
    are named generically (gfp_00, gfp_01, ...) rather than guessing exact
    per-bin semantics.

Usage:
  .venv/bin/python scripts/preprocess_gbt_gfp.py --dataset HI-Small
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from snapml import GraphFeaturePreprocessor

RAW_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = RAW_DIR / "processed"

# GFP config, matching Appendix D of the paper.
SCATTER_GATHER_TW_SECONDS = 6 * 3600     # 6 hours
DAILY_TW_SECONDS = 24 * 3600             # 1 day
LC_CYCLE_MAX_LEN = 10                    # simple cycles up to length 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="HI-Small",
                    help="Dataset prefix, e.g. HI-Small, LI-Small, HI-Medium ...")
    p.add_argument("--input-dir", default=str(RAW_DIR), help="Directory with <dataset>_Trans.csv")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Directory to write processed Parquet files")
    p.add_argument("--gfp-batch-size", type=int, default=0,
                    help="If >0, feed GFP in chunks of this many edges via partial_fit/transform "
                         "instead of a single fit_transform call (useful to bound memory on Large datasets).")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--nrows", type=int, default=None,
                    help="Debug: only read this many raw rows from the CSV")
    return p.parse_args()


def load_raw(dataset: str, input_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    path = input_dir / f"{dataset}_Trans.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[load] reading {path}")
    df = pd.read_csv(
        path,
        dtype={
            "From Bank": str,
            "Account": str,
            "To Bank": str,
            "Account.1": str,
            "Receiving Currency": "category",
            "Payment Currency": "category",
            "Payment Format": "category",
            "Is Laundering": np.int8,
        },
        parse_dates=["Timestamp"],
        nrows=nrows,
    )
    print(f"[load] {len(df):,} transactions")
    return df


def build_vertex_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map (Bank, Account) pairs to a shared integer vertex-ID space."""
    src_key = df["From Bank"] + "_" + df["Account"]
    dst_key = df["To Bank"] + "_" + df["Account.1"]
    codes, uniques = pd.factorize(pd.concat([src_key, dst_key], ignore_index=True), sort=False)
    n = len(df)
    src_id = codes[:n].astype(np.int64)
    dst_id = codes[n:].astype(np.int64)
    meta = {"num_accounts": int(len(uniques))}
    return src_id, dst_id, meta


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categoricals + derive simple transaction-level features."""
    cat_maps = {}

    payment_format_codes, payment_format_cats = pd.factorize(df["Payment Format"], sort=True)
    payment_currency_codes, payment_currency_cats = pd.factorize(df["Payment Currency"], sort=True)
    receiving_currency_codes, receiving_currency_cats = pd.factorize(df["Receiving Currency"], sort=True)
    cat_maps["Payment Format"] = list(payment_format_cats)
    cat_maps["Payment Currency"] = list(payment_currency_cats)
    cat_maps["Receiving Currency"] = list(receiving_currency_cats)

    ts = df["Timestamp"]
    feat = pd.DataFrame({
        "timestamp_epoch": ts.astype("int64") // 10**9,  # seconds since epoch, required by GFP
        "amount_paid": df["Amount Paid"].astype(np.float64),
        "amount_received": df["Amount Received"].astype(np.float64),
        "amount_diff": (df["Amount Paid"] - df["Amount Received"]).astype(np.float64),
        "payment_format": payment_format_codes.astype(np.float64),
        "payment_currency": payment_currency_codes.astype(np.float64),
        "receiving_currency": receiving_currency_codes.astype(np.float64),
        "currency_mismatch": (payment_currency_codes != receiving_currency_codes).astype(np.float64),
        "hour_of_day": ts.dt.hour.astype(np.float64),
        "day_of_week": ts.dt.dayofweek.astype(np.float64),
    })
    return feat, cat_maps


def configure_gfp() -> GraphFeaturePreprocessor:
    gfp = GraphFeaturePreprocessor()
    gfp.set_params({
        "num_threads": 12,
        # Vertex statistics computed on the "amount_paid" raw column (index 4
        # in the assembled edge array -- see build_gfp_input below), using
        # the default statistic set (fan, degree, ratio, avg, sum, var,
        # skew, kurtosis) over a 1-day window, as in the paper.
        "vertex_stats": True,
        "vertex_stats_tw": DAILY_TW_SECONDS,
        "vertex_stats_cols": [4],
        # Scatter-gather patterns, 6h window (paper Section 3.5).
        "scatter-gather": True,
        "scatter-gather_tw": SCATTER_GATHER_TW_SECONDS,
        # Temporal cycles, 1-day window.
        "temp-cycle": True,
        "temp-cycle_tw": DAILY_TW_SECONDS,
        # Length-constrained simple cycles, <=10 hops, 1-day window.
        "lc-cycle": True,
        "lc-cycle_tw": DAILY_TW_SECONDS,
        "lc-cycle_len": LC_CYCLE_MAX_LEN,
        # Not called out explicitly in the paper's GFP config -> disabled to
        # keep the feature set aligned with what Appendix D describes.
        "fan": False,
        "degree": False,
    })
    return gfp


def build_gfp_input(feat: pd.DataFrame, src_id: np.ndarray, dst_id: np.ndarray) -> np.ndarray:
    n = len(feat)
    edge_id = np.arange(n, dtype=np.float64)
    mandatory = np.column_stack([edge_id, src_id.astype(np.float64), dst_id.astype(np.float64),
                                  feat["timestamp_epoch"].to_numpy(dtype=np.float64)])
    other = feat.drop(columns=["timestamp_epoch"]).to_numpy(dtype=np.float64)
    return np.column_stack([mandatory, other]), list(feat.drop(columns=["timestamp_epoch"]).columns)


def run_gfp(gfp: GraphFeaturePreprocessor, gfp_input: np.ndarray, batch_size: int) -> np.ndarray:
    n = len(gfp_input)
    if batch_size <= 0:
        print(f"[gfp] fit_transform on {n:,} edges in a single call")
        return gfp.fit_transform(gfp_input)

    print(f"[gfp] streaming {n:,} edges in batches of {batch_size:,}")
    outputs = []
    for start in range(0, n, batch_size):
        chunk = gfp_input[start:start + batch_size]
        gfp.partial_fit(chunk)
        outputs.append(gfp.transform(chunk))
        if (start // batch_size) % 50 == 0:
            print(f"[gfp]   processed {start + len(chunk):,}/{n:,}")
    return np.vstack(outputs)


def temporal_split(n: int, train_frac: float, val_frac: float) -> dict:
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return {"train": (0, t1), "val": (t1, t2), "test": (t2, n)}


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_raw(args.dataset, input_dir, nrows=args.nrows)

    print("[step] sorting by timestamp (required by GFP)")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    print("[step] building vertex ID space from (Bank, Account) pairs")
    src_id, dst_id, vertex_meta = build_vertex_ids(df)

    print("[step] encoding categoricals / deriving base features")
    feat, cat_maps = build_feature_frame(df)

    print("[step] assembling GFP edge-list input")
    gfp_input, base_feature_names = build_gfp_input(feat, src_id, dst_id)
    print(f"[step] gfp input shape = {gfp_input.shape}")

    gfp = configure_gfp()
    gfp_output = run_gfp(gfp, gfp_input, args.gfp_batch_size)
    n_eng_features = gfp_output.shape[1] - gfp_input.shape[1]
    print(f"[gfp] produced {n_eng_features} additional graph-based features "
          f"(output shape = {gfp_output.shape})")

    gfp_feature_names = [f"gfp_{i:03d}" for i in range(n_eng_features)]
    all_columns = ["edge_id", "src_id", "dst_id", "timestamp_epoch"] + base_feature_names + gfp_feature_names
    full = pd.DataFrame(gfp_output, columns=all_columns)
    full["is_laundering"] = df["Is Laundering"].to_numpy()

    # Feature columns actually used for ML: drop IDs so models can't just
    # memorize account identities (paper, Section 4).
    ml_feature_cols = base_feature_names + gfp_feature_names
    drop_cols = ["edge_id", "src_id", "dst_id"]
    full_ml = full.drop(columns=drop_cols)

    print("[step] temporal 60/20/20 split")
    bounds = temporal_split(len(full_ml), args.train_frac, args.val_frac)
    for split_name, (lo, hi) in bounds.items():
        part = full_ml.iloc[lo:hi]
        out_path = output_dir / f"{split_name}.parquet"
        part.to_parquet(out_path, index=False)
        rate = part["is_laundering"].mean() * 100
        print(f"[save] {split_name}: {len(part):,} rows -> {out_path} "
              f"(laundering rate {rate:.4f}%)")

    metadata = {
        "dataset": args.dataset,
        "num_transactions": int(len(df)),
        "num_accounts": vertex_meta["num_accounts"],
        "feature_columns": ml_feature_cols,
        "label_column": "is_laundering",
        "category_mappings": cat_maps,
        "split_bounds": {k: [int(v[0]), int(v[1])] for k, v in bounds.items()},
        "gfp_config": gfp.get_params(),
        "elapsed_seconds": time.time() - t0,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[save] metadata -> {meta_path}")
    print(f"[done] total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
