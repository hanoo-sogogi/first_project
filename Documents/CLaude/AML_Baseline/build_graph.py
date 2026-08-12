"""
Build PyG graph snapshots (train / val / test) for the HI-Small AML dataset,
following the paper's evaluation protocol (Section 4, "Data Split"):

  - train graph:  edges from the train period only
  - val graph:    edges from train+val periods (train edges give message-
                   passing context; only VAL edges are scored)
  - test graph:   all edges (train+val+test context; only TEST edges scored)

Node set is the global account vocabulary (same accounts appear across
splits). Node features are simple in/out-degree stats computed from each
snapshot's own edges (structural only, no label leakage). Edge features are
transaction-level only (amount, format, time, self/same-bank flags) - no
hand-engineered graph aggregates, since the GNN is expected to learn
structure via message passing itself (this is the point of comparison
against the GBT + hand-crafted graph features baseline).
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"C:\Users\aica_\Documents\CLaude")
OUT_DIR = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
DATASET = "HI-Small"

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log("Loading transactions...")
df = pd.read_csv(DATA_DIR / f"{DATASET}_Trans.csv")
df.columns = [
    "Timestamp", "FromBankID", "FromAccount", "ToBankID", "ToAccount",
    "AmountReceived", "ReceivingCurrency", "AmountPaid", "PaymentCurrency",
    "PaymentFormat", "IsLaundering",
]
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
df = df.sort_values("Timestamp").reset_index(drop=True)
df["FromNode"] = df["FromBankID"].astype(str) + "_" + df["FromAccount"]
df["ToNode"] = df["ToBankID"].astype(str) + "_" + df["ToAccount"]

# --- USD normalization (same FX-BFS approach as the GBT baseline) ---
fx = df[df["ReceivingCurrency"] != df["PaymentCurrency"]].copy()
fx["rate_recv_per_paid"] = fx["AmountReceived"] / fx["AmountPaid"]
pair_rate = fx.groupby(["PaymentCurrency", "ReceivingCurrency"])["rate_recv_per_paid"].median().to_dict()
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
df["LogAmountUSD"] = np.log1p(df["AmountPaid"] * df["PaymentCurrency"].map(usd_per))
log("FX normalization done")

# --- transaction-level (edge) features ---
df["IsSelfLoop"] = (df["FromNode"] == df["ToNode"]).astype(np.float32)
df["IsSameBank"] = (df["FromBankID"] == df["ToBankID"]).astype(np.float32)
df["Hour"] = df["Timestamp"].dt.hour.astype(np.float32) / 23.0
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype(np.float32) / 6.0

fmt_codes = df["PaymentFormat"].astype("category")
FORMAT_CATEGORIES = list(fmt_codes.cat.categories)
df["PaymentFormatCode"] = fmt_codes.cat.codes.astype(np.int64)

# --- global node vocabulary ---
log("Building global node vocabulary...")
all_nodes = pd.Index(pd.concat([df["FromNode"], df["ToNode"]]).unique())
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
df["FromIdx"] = df["FromNode"].map(node_to_idx).astype(np.int64)
df["ToIdx"] = df["ToNode"].map(node_to_idx).astype(np.int64)
n_nodes = len(all_nodes)
log(f"{n_nodes:,} unique accounts across all splits")

# --- temporal split boundaries (same 60/20/20 as the GBT baseline) ---
n = len(df)
t1_idx, t2_idx = int(n * 0.6), int(n * 0.8)
t1, t2 = df["Timestamp"].iloc[t1_idx], df["Timestamp"].iloc[t2_idx]
train_mask = df["Timestamp"] < t1
val_mask = (df["Timestamp"] >= t1) & (df["Timestamp"] < t2)
test_mask = df["Timestamp"] >= t2

EDGE_FEATURE_COLS = ["LogAmountUSD", "IsSelfLoop", "IsSameBank", "Hour", "DayOfWeek"]


def build_snapshot(context_mask, score_mask, name):
    """context_mask selects edges usable for message passing; score_mask
    (subset of context_mask) selects which of those edges are labeled/scored."""
    ctx = df.loc[context_mask]
    score_local_mask = score_mask.loc[context_mask].to_numpy()

    edge_index = torch.tensor(
        np.stack([ctx["FromIdx"].to_numpy(), ctx["ToIdx"].to_numpy()]), dtype=torch.long
    )
    edge_attr = torch.tensor(ctx[EDGE_FEATURE_COLS].to_numpy(), dtype=torch.float32)
    edge_fmt = torch.tensor(ctx["PaymentFormatCode"].to_numpy(), dtype=torch.long)
    edge_y = torch.tensor(ctx["IsLaundering"].to_numpy(), dtype=torch.float32)
    score_mask_t = torch.tensor(score_local_mask, dtype=torch.bool)

    # Node features: degree + amount + counterparty-diversity stats within
    # this snapshot (structural only - no labels involved). Richer than bare
    # degree so the GNN has a useful starting point instead of having to
    # rediscover basic account statistics from scratch via message passing;
    # the GNN's job is then to layer *relational* signal (neighborhood
    # composition, multi-hop patterns) on top of this.
    from_idx = ctx["FromIdx"].to_numpy()
    to_idx = ctx["ToIdx"].to_numpy()
    amt = ctx["LogAmountUSD"].to_numpy()

    out_deg = np.zeros(n_nodes, dtype=np.float64)
    in_deg = np.zeros(n_nodes, dtype=np.float64)
    out_amt_sum = np.zeros(n_nodes, dtype=np.float64)
    in_amt_sum = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(out_deg, from_idx, 1.0)
    np.add.at(in_deg, to_idx, 1.0)
    np.add.at(out_amt_sum, from_idx, amt)
    np.add.at(in_amt_sum, to_idx, amt)
    out_avg_amt = np.divide(out_amt_sum, out_deg, out=np.zeros_like(out_amt_sum), where=out_deg > 0)
    in_avg_amt = np.divide(in_amt_sum, in_deg, out=np.zeros_like(in_amt_sum), where=in_deg > 0)

    out_cp = ctx.groupby("FromIdx")["ToIdx"].nunique()
    in_cp = ctx.groupby("ToIdx")["FromIdx"].nunique()
    out_unique_cp = np.zeros(n_nodes, dtype=np.float64)
    in_unique_cp = np.zeros(n_nodes, dtype=np.float64)
    out_unique_cp[out_cp.index.to_numpy()] = out_cp.to_numpy()
    in_unique_cp[in_cp.index.to_numpy()] = in_cp.to_numpy()

    x = torch.tensor(
        np.stack([
            np.log1p(out_deg), np.log1p(in_deg),
            out_avg_amt, in_avg_amt,
            np.log1p(out_unique_cp), np.log1p(in_unique_cp),
        ], axis=1), dtype=torch.float32
    )

    log(f"{name}: {edge_index.shape[1]:,} context edges, "
        f"{int(score_mask_t.sum()):,} scored edges "
        f"({int(edge_y[score_mask_t].sum())} positive)")

    torch.save(
        {
            "edge_index": edge_index, "edge_attr": edge_attr, "edge_fmt": edge_fmt,
            "y": edge_y, "score_mask": score_mask_t, "x": x, "n_nodes": n_nodes,
        },
        OUT_DIR / f"graph_{name}.pt",
    )


log("Building train snapshot (context=train, score=train)...")
build_snapshot(train_mask, train_mask, "train")

log("Building val snapshot (context=train+val, score=val)...")
build_snapshot(train_mask | val_mask, val_mask, "val")

log("Building test snapshot (context=all, score=test)...")
build_snapshot(train_mask | val_mask | test_mask, test_mask, "test")

meta = {"n_nodes": n_nodes, "format_categories": FORMAT_CATEGORIES,
        "edge_feature_cols": EDGE_FEATURE_COLS}
torch.save(meta, OUT_DIR / "graph_meta.pt")
log(f"Done. Total runtime: {time.time() - t0:.1f}s")
