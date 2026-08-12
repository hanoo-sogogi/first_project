"""
Train GBT (LightGBM / XGBoost) models on GFP-preprocessed IBM AML data,
following the evaluation protocol of:

  Altman et al., "Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models", NeurIPS 2023 (arXiv:2306.16424), Section 4
  and Appendix C.

Pipeline:
  1. Load train/val/test Parquet produced by preprocess_gbt_gfp.py.
  2. Random-search hyperparameters (sampled from the same ranges as the
     paper's Table 10 "Range-large") on the train set, model-selecting by
     minority-class (is_laundering=1) F1 on the validation set.
  3. Retrain the best configuration with several random seeds (paper: 4)
     and report mean +/- std Precision / Recall / F1 for the minority class
     on the held-out test set -- the same metric reported in the paper's
     Table 2.
  4. Save the best hyperparameters, per-seed models, and a results.json.

Parallelism: search trials (and, separately, the final per-seed retrains)
are independent of each other, so they are fanned out across a
ProcessPoolExecutor instead of run sequentially. Each worker process loads
the train/val/test arrays exactly once (via an initializer) and reuses them
for every task it is handed, so the (train, val, test) split is not
repickled per task -- only the small params dict / metrics go over the
process-pool pipe. Each individual LightGBM/XGBoost fit still uses its own
internal thread pool (--threads-per-trial); --max-workers x
--threads-per-trial should stay close to the machine's core count.

Deviations from the paper made explicit here:
  - The paper uses successive-halving hyperparameter search (Appendix C,
    Table 9) with up to 1000 initial configurations evaluated on
    increasing fractions of the training set. We use a flat random search
    over the same parameter ranges (Table 10) with a configurable number
    of trials (--n-search, default 15) evaluated on the full training set
    directly, which is tractable at Small-dataset scale and simpler to
    reason about. Pass --n-search higher for a closer (but slower)
    approximation.
  - Everything else (parameter ranges, minority-class F1 model selection,
    4-seed test evaluation, 0.5 decision threshold) mirrors the paper.

Usage:
  .venv/bin/python scripts/train_gbt.py --dataset HI-Small --model both \
      --n-search 15 --max-workers 15 --threads-per-trial 5
"""

import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

RAW_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = RAW_DIR / "processed"
RESULTS_DIR = RAW_DIR / "results"

RNG_SEEDS = [0, 1, 2, 3]  # paper: "four models initialized with different random seeds"

# Populated once per worker process by _worker_init; reused across all tasks
# handed to that worker (avoids re-pickling/re-sending the full train/val/
# test arrays for every single trial).
_DATA = {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="HI-Small")
    p.add_argument("--model", choices=["lightgbm", "xgboost", "both"], default="both")
    p.add_argument("--n-search", type=int, default=15, help="Number of random-search trials")
    p.add_argument("--seeds", type=int, nargs="+", default=RNG_SEEDS)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--threads-per-trial", type=int, default=5,
                    help="LightGBM/XGBoost internal thread count per single fit")
    p.add_argument("--max-workers", type=int, default=None,
                    help="Process pool size. Default: min(n_search, cpu_count // threads-per-trial)")
    return p.parse_args()


def load_split(dataset_dir: Path, split: str, feature_cols: list, label_col: str):
    df = pd.read_parquet(dataset_dir / f"{split}.parquet", columns=feature_cols + [label_col])
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int32)
    return X, y


def log_uniform(rng, low_exp, high_exp, size=None):
    return 10 ** rng.uniform(low_exp, high_exp, size=size)


# --- Parameter samplers, ranges from paper Table 10 ("Range-large"), capped
# for our 27GB cgroup memory limit. The paper's raw ranges (num_leaves up to
# 16384, max_depth up to 15) were tuned for IBM's research cluster and blew
# through our container's memory budget within minutes of concurrent
# search (observed 92% cgroup usage with only 4 workers x 8 threads before
# we killed it). num_leaves is capped at 1024 and max_depth at 10, which
# still covers highly expressive trees while keeping per-trial histogram
# memory bounded.

MAX_NUM_LEAVES = 1024
MAX_TREE_DEPTH = 10


def sample_lightgbm_params(rng: np.random.Generator) -> dict:
    return {
        "num_round": int(rng.integers(10, 1000)),
        "num_leaves": int(rng.integers(1, MAX_NUM_LEAVES)),
        "learning_rate": float(log_uniform(rng, -2.5, -1)),
        "lambda_l2": float(log_uniform(rng, -2, 2)),
        "scale_pos_weight": float(rng.uniform(1, 10)),
        "lambda_l1": float(log_uniform(rng, 0.01, 0.5)),
    }


def sample_xgboost_params(rng: np.random.Generator) -> dict:
    return {
        "num_round": int(rng.integers(10, 1000)),
        "max_depth": int(rng.integers(1, MAX_TREE_DEPTH)),
        "learning_rate": float(log_uniform(rng, -2.5, -1)),
        "lambda": float(log_uniform(rng, -2, 2)),
        "scale_pos_weight": float(rng.uniform(1, 10)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "subsample": float(rng.uniform(0.5, 1.0)),
    }


# --- Train / predict wrappers ---

def train_lightgbm(params: dict, X_train, y_train, seed: int, threads: int):
    import lightgbm as lgb
    booster_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": params["num_leaves"],
        "learning_rate": params["learning_rate"],
        "lambda_l1": params["lambda_l1"],
        "lambda_l2": params["lambda_l2"],
        "scale_pos_weight": params["scale_pos_weight"],
        "seed": seed,
        "verbose": -1,
        "num_threads": threads,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(booster_params, train_set, num_boost_round=params["num_round"])
    return model


def predict_lightgbm(model, X):
    return model.predict(X)


def train_xgboost(params: dict, X_train, y_train, seed: int, threads: int):
    import xgboost as xgb
    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": params["max_depth"],
        "eta": params["learning_rate"],
        "lambda": params["lambda"],
        "scale_pos_weight": params["scale_pos_weight"],
        "colsample_bytree": params["colsample_bytree"],
        "subsample": params["subsample"],
        "seed": seed,
        "nthread": threads,
        "tree_method": "hist",
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(booster_params, dtrain, num_boost_round=params["num_round"])
    return model


def predict_xgboost(model, X):
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X))


MODEL_FNS = {
    "lightgbm": (sample_lightgbm_params, train_lightgbm, predict_lightgbm, "txt"),
    "xgboost": (sample_xgboost_params, train_xgboost, predict_xgboost, "json"),
}


def minority_f1(y_true, y_prob, threshold=0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def evaluate(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }


# --- Worker-process machinery ---

def _worker_init(dataset_dir_str: str, feature_cols: list, label_col: str, threads: int):
    """Runs once per worker process; caches the splits in module globals."""
    global _DATA
    # Cap BLAS/OMP threads used by numpy/pandas inside the worker so they
    # don't oversubscribe on top of the GBT library's own thread pool.
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    dataset_dir = Path(dataset_dir_str)
    _DATA["X_train"], _DATA["y_train"] = load_split(dataset_dir, "train", feature_cols, label_col)
    _DATA["X_val"], _DATA["y_val"] = load_split(dataset_dir, "val", feature_cols, label_col)
    _DATA["X_test"], _DATA["y_test"] = load_split(dataset_dir, "test", feature_cols, label_col)


def _search_trial_task(task):
    name, trial_idx, params, threads = task
    _, train_fn, predict_fn, _ = MODEL_FNS[name]
    t0 = time.time()
    model = train_fn(params, _DATA["X_train"], _DATA["y_train"], seed=0, threads=threads)
    y_val_prob = predict_fn(model, _DATA["X_val"])
    val_f1 = minority_f1(_DATA["y_val"], y_val_prob)
    return {"trial": trial_idx, "params": params, "val_f1": val_f1, "seconds": time.time() - t0}


def _seed_eval_task(task):
    name, seed, params, threads, model_path, ext = task
    _, train_fn, predict_fn, _ = MODEL_FNS[name]
    model = train_fn(params, _DATA["X_train"], _DATA["y_train"], seed=seed, threads=threads)
    y_test_prob = predict_fn(model, _DATA["X_test"])
    metrics = evaluate(_DATA["y_test"], y_test_prob)
    model.save_model(str(model_path))
    return {"seed": seed, "metrics": metrics}


def run_model_family(name: str, dataset: str, dataset_dir: Path, feature_cols: list,
                      label_col: str, n_search: int, seeds: list, results_dir: Path,
                      threads_per_trial: int, max_workers: int):
    sample_fn, _, _, ext = MODEL_FNS[name]
    print(f"\n{'=' * 70}\n[{name}] launching {max_workers} worker process(es) "
          f"({threads_per_trial} threads each)\n{'=' * 70}")

    out_dir = results_dir / dataset / name
    out_dir.mkdir(parents=True, exist_ok=True)

    search_rng = np.random.default_rng(42)
    trials = [(name, i, sample_fn(search_rng), threads_per_trial) for i in range(n_search)]

    search_log = [None] * n_search
    t_search0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(str(dataset_dir), feature_cols, label_col, threads_per_trial),
    ) as pool:
        print(f"[{name}] random search: {n_search} trials in flight, "
              f"model-selecting by val minority-F1")
        futures = {pool.submit(_search_trial_task, t): t[1] for t in trials}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            search_log[result["trial"]] = result
            done += 1
            print(f"[{name}] [{done:>2}/{n_search}] trial {result['trial']:>2}  "
                  f"val_f1={result['val_f1']:6.3f}  ({result['seconds']:5.1f}s)")

        best = max(search_log, key=lambda r: r["val_f1"])
        best_params, best_val_f1 = best["params"], best["val_f1"]
        print(f"[{name}] search wall time: {time.time() - t_search0:.1f}s "
              f"(sum of trial times: {sum(r['seconds'] for r in search_log):.1f}s)")
        print(f"[{name}] best params (val_f1={best_val_f1:.3f}): {best_params}")

        print(f"[{name}] retraining best config on {len(seeds)} seeds (in parallel), "
              f"evaluating on test set")
        seed_tasks = [
            (name, seed, best_params, threads_per_trial, out_dir / f"model_seed{seed}.{ext}", ext)
            for seed in seeds
        ]
        seed_futures = [pool.submit(_seed_eval_task, t) for t in seed_tasks]
        per_seed_metrics_by_seed = {}
        for fut in as_completed(seed_futures):
            r = fut.result()
            per_seed_metrics_by_seed[r["seed"]] = r["metrics"]
            m = r["metrics"]
            print(f"[{name}] seed={r['seed']}  precision={m['precision']:.2f}  "
                  f"recall={m['recall']:.2f}  f1={m['f1']:.2f}")

    per_seed_metrics = [per_seed_metrics_by_seed[s] for s in seeds]
    summary = {}
    for k in ["precision", "recall", "f1"]:
        vals = np.array([m[k] for m in per_seed_metrics])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    print(f"[{name}] TEST minority-class F1 = {summary['f1']['mean']:.2f} +/- {summary['f1']['std']:.2f}  "
          f"(precision={summary['precision']['mean']:.2f}+/-{summary['precision']['std']:.2f}, "
          f"recall={summary['recall']['mean']:.2f}+/-{summary['recall']['std']:.2f})")

    results = {
        "model_family": name,
        "dataset": dataset,
        "best_params": best_params,
        "best_val_f1": best_val_f1,
        "search_log": search_log,
        "per_seed_test_metrics": per_seed_metrics,
        "test_summary": summary,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{name}] results -> {results_path}")
    return results


def main():
    args = parse_args()
    dataset_dir = Path(args.processed_dir) / args.dataset
    results_dir = Path(args.results_dir)

    meta = json.load(open(dataset_dir / "metadata.json"))
    feature_cols = meta["feature_columns"]
    label_col = meta["label_column"]
    print(f"[main] dataset={args.dataset}  #features={len(feature_cols)}  label={label_col}")

    cpu_count = os.cpu_count() or 8
    families = ["lightgbm", "xgboost"] if args.model == "both" else [args.model]
    all_results = {}
    for name in families:
        n_tasks = max(args.n_search, len(args.seeds))
        max_workers = args.max_workers or max(1, min(n_tasks, cpu_count // args.threads_per_trial))
        all_results[name] = run_model_family(
            name, args.dataset, dataset_dir, feature_cols, label_col,
            args.n_search, args.seeds, results_dir,
            args.threads_per_trial, max_workers,
        )

    print(f"\n{'=' * 70}\nSUMMARY ({args.dataset}, minority-class F1 on test)\n{'=' * 70}")
    for name, res in all_results.items():
        s = res["test_summary"]
        print(f"  {name:10s} F1={s['f1']['mean']:6.2f} +/- {s['f1']['std']:4.2f}   "
              f"P={s['precision']['mean']:6.2f}  R={s['recall']['mean']:6.2f}")


if __name__ == "__main__":
    main()
