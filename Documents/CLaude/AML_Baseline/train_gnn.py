"""
GIN+EU-style edge-classification GNN for HI-Small laundering detection.

Architecture follows the GNN family benchmarked in the paper (Section 4):
message-passing over the transaction graph (GINEConv, edge-feature-aware),
with a final edge readout combining the two endpoint node embeddings and the
edge's own features - i.e. the model must learn structural signal (degree,
neighborhood patterns, fan-in/out, repeat counterparties) purely through
message passing, unlike the GBT baseline which was handed those as explicit
features.

Graph snapshots (train / val / test) are the ones written by build_graph.py.
"""
import json
import os
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch_geometric.nn import GINEConv

OUT_DIR = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
SEEDS = [1, 2, 3]
HIDDEN = 48
FMT_DIM = 6
EPOCHS = 120
LR = 0.003
GRAD_CLIP = 1.0
NEG_SAMPLES_PER_EPOCH = 50_000
PATIENCE = 25

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


meta = torch.load(OUT_DIR / "graph_meta.pt", weights_only=False)
n_formats = len(meta["format_categories"])
log(f"Loaded meta: {meta['n_nodes']:,} nodes, {n_formats} payment formats")

graphs = {}
for split in ("train", "val", "test"):
    g = torch.load(OUT_DIR / f"graph_{split}.pt", weights_only=False)
    graphs[split] = g
    log(f"{split}: x={tuple(g['x'].shape)}, edges={g['edge_index'].shape[1]:,}, "
        f"scored={int(g['score_mask'].sum()):,}, pos={int(g['y'][g['score_mask']].sum())}")


class EdgeFeatureEncoder(nn.Module):
    def __init__(self, n_formats, fmt_dim):
        super().__init__()
        self.fmt_embed = nn.Embedding(n_formats, fmt_dim)
        self.out_dim = 5 + fmt_dim  # LogAmountUSD, IsSelfLoop, IsSameBank, Hour, DayOfWeek

    def forward(self, edge_attr, edge_fmt):
        return torch.cat([edge_attr, self.fmt_embed(edge_fmt)], dim=1)


class GINEUModel(nn.Module):
    """2-layer GINEConv encoder + edge-update-style readout head."""

    def __init__(self, n_formats, hidden=HIDDEN, fmt_dim=FMT_DIM):
        super().__init__()
        self.edge_enc = EdgeFeatureEncoder(n_formats, fmt_dim)
        edim = self.edge_enc.out_dim

        self.node_enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())

        def mlp():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.conv1 = GINEConv(mlp(), edge_dim=edim)
        self.conv2 = GINEConv(mlp(), edge_dim=edim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.readout = nn.Sequential(
            nn.Linear(hidden * 2 + edim, hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index, edge_attr, edge_fmt):
        eattr = self.edge_enc(edge_attr, edge_fmt)
        h0 = self.node_enc(x)
        h1 = F.relu(self.norm1(self.conv1(h0, edge_index, eattr)))
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index, eattr))) + h1
        return h2, eattr

    def edge_logits(self, h, eattr, edge_index, score_mask):
        src, dst = edge_index[:, score_mask]
        feat = torch.cat([h[src], h[dst], eattr[score_mask]], dim=1)
        return self.readout(feat).squeeze(-1)


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
    log(f"{name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} PR-AUC={pr_auc:.4f} "
        f"P@500={p_at_k.get(500, float('nan')):.3f}")
    return result


all_results = []
best_val_curve_by_seed = {}

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    log(f"=== Seed {seed} ===")

    model = GINEUModel(n_formats)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    g_train = graphs["train"]
    g_val = graphs["val"]
    y_train_full = g_train["y"]  # train score_mask is all-True (context == score)
    pos_idx = torch.where(y_train_full == 1)[0]
    neg_idx_all = torch.where(y_train_full == 0)[0]
    log(f"  train pool: {len(pos_idx)} positive, {len(neg_idx_all)} negative edges "
        f"(sampling {NEG_SAMPLES_PER_EPOCH} negatives/epoch for the loss - full graph "
        f"is still used for message passing, only the LOSS is balanced)")

    rng = torch.Generator().manual_seed(seed)

    best_val_prauc = -1.0
    best_state = None
    epochs_no_improve = 0
    val_curve = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        h, eattr = model.encode(g_train["x"], g_train["edge_index"], g_train["edge_attr"], g_train["edge_fmt"])

        neg_sample = neg_idx_all[torch.randperm(len(neg_idx_all), generator=rng)[:NEG_SAMPLES_PER_EPOCH]]
        loss_idx = torch.cat([pos_idx, neg_sample])
        loss_mask = torch.zeros(g_train["edge_index"].shape[1], dtype=torch.bool)
        loss_mask[loss_idx] = True

        logits = model.edge_logits(h, eattr, g_train["edge_index"], loss_mask)
        y_batch = y_train_full[loss_mask]
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        model.eval()
        with torch.no_grad():
            h, eattr = model.encode(g_val["x"], g_val["edge_index"], g_val["edge_attr"], g_val["edge_fmt"])
            val_logits = model.edge_logits(h, eattr, g_val["edge_index"], g_val["score_mask"])
            val_scores = torch.sigmoid(val_logits).numpy()
            val_y = g_val["y"][g_val["score_mask"]].numpy()
            val_prauc = average_precision_score(val_y, val_scores)
        val_curve.append(val_prauc)
        sched.step(val_prauc)

        if epoch == 1 or epoch % 5 == 0:
            cur_lr = opt.param_groups[0]["lr"]
            log(f"  epoch {epoch:2d}: train_loss={loss.item():.4f} val_PR-AUC={val_prauc:.4f} lr={cur_lr:.5f}")

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log(f"  early stop at epoch {epoch} (best val PR-AUC={best_val_prauc:.4f})")
                break

    best_val_curve_by_seed[seed] = val_curve
    model.load_state_dict(best_state)
    model.eval()
    g_test = graphs["test"]
    with torch.no_grad():
        h, eattr = model.encode(g_test["x"], g_test["edge_index"], g_test["edge_attr"], g_test["edge_fmt"])
        test_logits = model.edge_logits(h, eattr, g_test["edge_index"], g_test["score_mask"])
        test_scores = torch.sigmoid(test_logits).numpy()
        test_y = g_test["y"][g_test["score_mask"]].numpy()

    all_results.append(evaluate(test_y, test_scores, f"gnn_seed{seed}"))

# --- aggregate ---
def summarize(results_list):
    keys = ["f1", "precision", "recall", "pr_auc", "recall_at_precision90"]
    summary = {k: {"mean": float(np.mean([r[k] for r in results_list])),
                    "min": float(np.min([r[k] for r in results_list])),
                    "max": float(np.max([r[k] for r in results_list]))} for k in keys}
    for k in (100, 500, 1000, 2000):
        vals = [r["precision_at_k"].get(k) for r in results_list if k in r["precision_at_k"]]
        if vals:
            summary[f"precision_at_{k}"] = {"mean": float(np.mean(vals))}
    return summary


final = {
    "dataset": "HI-Small", "model": "GINEConv x2 + edge-readout (GIN+EU style)",
    "hidden_dim": HIDDEN, "epochs_max": EPOCHS,
    "neg_samples_per_epoch": NEG_SAMPLES_PER_EPOCH,
    "seeds": SEEDS, "raw_results": all_results, "summary": summarize(all_results),
    "val_curve_by_seed": best_val_curve_by_seed,
}
with open(OUT_DIR / "results_gnn.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

log(f"Done. Results written to {OUT_DIR / 'results_gnn.json'}")
log(f"Total runtime: {time.time() - t0:.1f}s")
