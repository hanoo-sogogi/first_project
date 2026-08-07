# IBM AML Benchmark Reproduction — Handoff

Reproduction of Altman et al., *"Realistic Synthetic Financial Transactions for
Anti-Money Laundering Models"* (NeurIPS 2023, arXiv:2306.16424), using the
Kaggle **IBM Transactions for Anti-Money Laundering (AML)** dataset
(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`).

Scope on this project: **data preprocessing + modeling** (GBT and GNN pipelines).
Production/serving infra is out of scope.

Raw Kaggle CSVs (`HI-Small_Trans.csv`, `LI-Small_Trans.csv`,
`HI-Small_accounts.csv`, `LI-Small_accounts.csv`, ...) are **not** included here —
already available in the target environment. Re-download via `kagglehub` if needed.

---

## 1. Two independent pipelines

| | GBT + Graph Feature Preprocessor | GNN (Multi-GNN) |
|---|---|---|
| Location | `scripts/preprocess_gbt_gfp.py`, `scripts/train_gbt.py` | `multi_gnn/` (patched clone of `github.com/IBM/Multi-GNN`) |
| Account IDs | Used to build the graph for GFP, then **dropped** from the final feature matrix | Kept as the graph structure itself (integer node IDs) |
| Graph signal | 99 hand-engineered features from IBM `snapml` GraphFeaturePreprocessor (vertex stats, scatter-gather, temporal cycles, ≤10-hop simple cycles) | Learned via message passing (GIN / GIN+EU / PNA) |
| Split | Row-order 60/20/20 (already timestamp-sorted) | Day-based 60/20/20 (auto-searched day boundaries) |
| Env | `.venv` → `requirements-gbt.txt` | `venv-gnn` → `requirements-gnn.txt` |

### Recreating the environments
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-gbt.txt
python3 -m venv venv-gnn && venv-gnn/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
# then match PyG wheels to whatever torch version actually installs, e.g.:
venv-gnn/bin/pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-<VERSION>+cu124.html
venv-gnn/bin/pip install -r requirements-gnn.txt
```
`requirements-gnn.txt` was frozen from a torch 2.6.0+cu124 environment — if the new
box's CUDA driver differs, resolve torch/PyG versions fresh rather than forcing
these exact pins (driver just needs to be >= the wheel's CUDA runtime).

---

## 2. GBT + GFP — what it does, results

`preprocess_gbt_gfp.py`:
1. Loads raw `<dataset>_Trans.csv`, sorts by timestamp (GFP requires this).
2. Builds a unified (Bank, Account) → integer vertex-ID space shared between
   source/target.
3. Encodes categoricals (Payment Format, Payment/Receiving Currency), derives
   `amount_diff`, `currency_mismatch`, `hour_of_day`, `day_of_week` (9 base features).
4. Runs `snapml.GraphFeaturePreprocessor` — vertex stats (fan/degree/ratio/avg/
   sum/var/skew/kurtosis on `amount_paid`, 1-day window), scatter-gather (6h
   window), temporal cycles (1-day window), length-constrained simple cycles
   (≤10 hops, 1-day window) → 99 features.
5. Drops account/edge IDs, splits 60/20/20 by row order, writes Parquet +
   `metadata.json`.

`train_gbt.py`: random search over LightGBM/XGBoost hyperparameters (ranges
from the paper's Table 10), then 4-seed retrain of the best config, reporting
mean±std minority-class F1/precision/recall on test.

**Deviations from the paper** (both driven by the container's 27GB cgroup
memory limit — check `cat /sys/fs/cgroup/memory.max`, not `free -h`, which
reports host memory and is misleading in a container):
- 15 random-search trials instead of the paper's successive-halving over up to
  ~1000 configs.
- `num_leaves` capped at 1024, `max_depth` at 10 (paper: 16384 / 15). Running
  4 parallel workers with the paper's uncapped ranges pushed memory to 92%
  and nearly OOM'd.

**Results (HI-Small, minority-class F1 %, 4-seed mean±std):**

| Model | Ours | Paper |
|---|---|---|
| LightGBM | 54.26 ± 0.28 (P 93.38 / R 38.24) | 62.86 ± 0.25 |
| XGBoost | 55.55 ± 0.44 (P 87.87 / R 40.61) | 63.23 ± 0.17 |

Artifacts: `results/HI-Small/{lightgbm,xgboost}/results.json` (full search log +
best params) and `model_seed{0-3}.{txt,json}`.

---

## 3. GNN (Multi-GNN) — what it does, results, and the 3 upstream bugs patched

Cloned `github.com/IBM/Multi-GNN` (the paper authors' own reference
implementation) and patched **3 real bugs** found while running it:

1. **`util.py`** — `--unique_name` was `action='store_true'` (only ever `True`/
   `False`), so every model would collide on the same checkpoint filename.
   Changed to `type=str, default="model"` so each model family gets its own
   `checkpoint_<name>.tar`.
2. **`util.py`** — `inference.py` reads `args.avg_tps`, which was never defined
   in the parser → `AttributeError` on every `--inference` run. Added
   `parser.add_argument("--avg_tps", action='store_true', ...)`.
3. **`train_util.py`** — `inference.py` calls `evaluate_homo(..., precrec=True)`
   but the function didn't accept that kwarg → `TypeError`. Added `precrec`
   support to both `evaluate_homo` and `evaluate_hetero`, returning
   `(f1, precision, recall)` when set.
4. **`inference.py`** — computed `te_f1, te_prec, te_rec` but never printed/
   logged them. Added a `logging.info(...)` line.

### Data prep
`multi_gnn/format_kaggle_files.py <path/to/HI-Small_Trans.csv>` (needs the
`datatable` package — installs fine on Python 3.12) → `formatted_transactions.csv`
with columns `EdgeID, from_id, to_id, Timestamp(relative seconds), Amount Sent,
Sent Currency, Amount Received, Received Currency, Payment Format, Is Laundering`.
Place it at `<aml_data>/<dataset_name>/formatted_transactions.csv` and point
`data_config.json`'s `aml_data` path at the parent dir (currently reset to
placeholder paths — **edit `multi_gnn/data_config.json` before running**).

Node features are a constant placeholder (`1` for every node) — the model is
purely topology + edge-feature driven (inductive, no learned per-node lookup
table). Edge features used: `Timestamp, Amount Received, Received Currency,
Payment Format` only (Amount Sent / Payment Currency unused — that's the
original repo's choice, not ours).

Day-based split (not row-order): searches day boundaries minimizing deviation
from 60/20/20 of cumulative daily transaction counts. HI-Small: train days
0-5 (64.02%), val days 6-7 (19.01%), test days 8-17 (16.97%). Val graph =
train+val edges (val-period scored only); test graph = everything (test-period
scored only).

`z_norm` (data_util.py) is applied **independently per split** using each
split's own mean/std (not train stats propagated to val/test) — worth knowing
if you touch normalization.

### Training pattern used
Paper doesn't state epoch count explicitly in the text we had access to; the
official repo's `--n_epochs` default is **100**, which we treated as the
target. Approach: run `--n_epochs 15` first, check the Val-F1 trend, and if
still improving, continue via `--finetune --n_epochs <remaining> --unique_name
<same name>` (loads `checkpoint_<name>.tar`, saves to
`checkpoint_<name>_finetuned.tar`). Best checkpoint is always selected by
**Validation F1**, saved automatically whenever val F1 improves on the
in-process best (a bad epoch afterward can't clobber the saved best).

**GPU note**: PNA alone uses ~29.5GB / 32GB (V100) — it cannot run concurrently
with any GIN-family run. Observed a real `CUDA out of memory` crash trying
GIN+EU + PNA at once. Run them sequentially.

### Results (HI-Small, single seed unless noted; paper numbers are 4-seed avg)

| Model | Status | Best epoch | Test F1 (ours) | Paper F1 (4-seed avg) |
|---|---|---|---|---|
| GIN | **Done, 100/100** | 91 | **32.29** | 28.70 ± 1.13 |
| GIN+EU (`--emlps`) | **Stopped at 83/100** (see below) | 51 (of the continuation run) | **46.38** (Val F1 0.4897) | 47.73 ± 7.86 |
| PNA | **Paused at 15/100** | 15 (still improving when stopped) | **51.99** (Val F1 0.4819) | 56.77 ± 2.41 |

**GIN+EU training collapse**: epochs 81-83 (of the 85-epoch continuation run)
had Train/Val/Test F1 = exactly 0.0000 for 3 consecutive epochs — the run
didn't crash (no exception, GPU still active), it just diverged into predicting
the majority class only. No recovery observed by epoch 83. Decision: stop and
keep the epoch-51 checkpoint as final (`checkpoints/gnn/checkpoint_gin_emlps_finetuned.tar`).
If picking this back up, consider resuming from that checkpoint with a **lower
learning rate** rather than the original config — the collapse looks like an
optimizer divergence, not a data issue.

**PNA next step**: resume from `checkpoints/gnn/checkpoint_pna.tar` with
`--finetune --n_epochs 85 --unique_name pna --save_model` — was still setting
new Val-F1 records at epoch 15 when paused, so it very likely benefits from
completing the full 100.

**4-seed averaging**: not done for any GNN model (compute/time cost) — our GNN
numbers are single-seed, unlike the paper's 4-seed mean±std and unlike our own
GBT numbers (which do use 4 seeds).

### Checkpoints (`checkpoints/gnn/`)
| File | What it is |
|---|---|
| `checkpoint_gin.tar` | GIN, phase-1 best (15 epochs) |
| `checkpoint_gin_HI-Small_100ep_best.tar` | GIN, final 100-epoch best (epoch 91) — **use this one** |
| `checkpoint_gin_15ep_backup.tar` | GIN, redundant backup of phase-1 |
| `checkpoint_gin_LI-Small_finetuned.tar` | GIN, HI→LI-Small 5-epoch fine-tune result |
| `checkpoint_gin_emlps.tar` | GIN+EU, phase-1 best (13 epochs) |
| `checkpoint_gin_emlps_finetuned.tar` | GIN+EU, final best (epoch 51 of continuation) — **use this one** |
| `checkpoint_pna.tar` | PNA, 15-epoch checkpoint — **resume training from this** |

---

## 4. Cross-dataset transfer: HI-Small → LI-Small (GIN only)

- **Zero-shot** (HI-Small GIN checkpoint evaluated directly on LI-Small,
  no fine-tuning): F1 = 0.00 / P = 0.00 / R = 0.00. This matches the paper's
  own documented finding (Table 3: PNA zero-shot on LI-Small = 0.00 ± 0.00) —
  a model tuned for HI's class balance and decision boundary just doesn't fire
  on LI's sparser, differently-distributed laundering patterns.
- **5-epoch fine-tune** from the HI-Small checkpoint: best (by val F1, epoch 5)
  Test F1 = 10.28%. Paper's illustrative reference (different model — PNA + 5ep
  fine-tune) = 27.38 ± 1.03. Not directly comparable model-to-model, but shows
  fine-tuning recovers *some* signal from zero.

---

## 5. Data quality checks done (both investigated, neither fixed — negligible impact)

1. **Exact duplicate transactions**: HI-Small has 9 duplicate-row groups (18
   rows), LI-Small has 8 groups (16 rows) — all near-zero-value Bitcoin
   transactions, all `Is Laundering=0`. Duplicate pairs share identical
   timestamps so they always land in the same train/val/test split — **no
   leakage**. ~0.0003% of the data; left as-is.
2. **Day-span mismatch**: paper's Table 4 states HI-Small spans "10 days," but
   the actual data spans 18 days (Sep 1–18) with a long sparse tail after day
   10 (transaction counts drop from ~200-450K/day to 11-400/day). The
   day-based split logic in `data_loading.py` computes proportions from the
   *actual* distribution, so this doesn't break anything — just a documentation
   discrepancy in the paper worth knowing about.

## 6. Known unaddressed issue — deferred by design, not an oversight

**Currency is not normalized/FX-converted.** Amounts stay in native currency
units — e.g. Bitcoin transactions average ~20.8 (raw BTC), Rupee/Yen/Ruble
transactions average in the tens of millions (raw units). Currency itself is
a meaningful predictor on its own (HI-Small: Saudi Riyal laundering rate
0.42% vs Bitcoin 0.038%, ~11x difference), so both pipelines pass currency as
a categorical feature alongside the raw amount, without FX-adjusting the
amount itself.

- **GBT**: low risk — trees can condition amount-thresholds on the currency
  category jointly, so the scale mismatch mostly gets absorbed.
- **GNN**: higher risk — `z_norm` uses one global mean/std across all
  currencies mixed together, which likely compresses the signal for
  low-magnitude currencies (e.g. Bitcoin) toward zero.

**Decision**: treat current results as the "no currency normalization"
baseline and do currency normalization as a **separate follow-up experiment**
(touches both `preprocess_gbt_gfp.py`'s GFP vertex-stats input and
`multi_gnn/data_util.py`'s `z_norm`) rather than reworking runs that were
already hours into training. Not started yet.

---

## 7. Suggested next steps, in rough priority order

1. Resume PNA: 15 → 100 epochs (`checkpoint_pna.tar`, was still improving).
2. Decide on GIN+EU: accept epoch-51 as final, or resume from that checkpoint
   with a lower LR to try to safely reach 100.
3. Currency normalization follow-up experiment (Section 6).
4. 4-seed averaging for whichever GNN models matter most for the final
   writeup (paper reports 4-seed mean±std; we don't yet, for GNN).
5. If useful: a hybrid pipeline that feeds the GFP-engineered graph features
   (99 of them) into the GNN as *additional* edge/node features, combining
   "GNN learns structure" with "hand-engineered graph stats" — not something
   the paper does, but a natural extension a teammate suggested and that
   the GBT+GFP results (already the strongest pipeline here) motivate.

## 8. Team-facing results summary

A shareable HTML progress report (same numbers as this doc, plus charts) was
published as a Claude Artifact during the original work session — ask
whoever ran that session for the link if you want the visual version; it
isn't portable into this repo as a live page.
