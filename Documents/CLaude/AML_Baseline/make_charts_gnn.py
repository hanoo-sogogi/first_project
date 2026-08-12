import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
gbt = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
gnn = json.loads((OUT / "results_gnn.json").read_text(encoding="utf-8"))

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --- Chart: val PR-AUC learning curves per seed ---
fig, ax = plt.subplots(figsize=(8, 4.8))
colors = {"1": "#d9663b", "2": "#2c7f6e", "3": "#4a6fa5"}
for seed, curve in gnn["val_curve_by_seed"].items():
    ax.plot(range(1, len(curve) + 1), curve, label=f"seed {seed}", color=colors[seed], linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("검증(val) PR-AUC")
ax.set_title("GNN 학습 곡선 (시드별 val PR-AUC 추이)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "chart_gnn_learning_curve.png", dpi=150)
plt.close()

# --- Chart: PR-AUC comparison, GBT vs GNN vs paper baselines ---
paper_hi_small_pr_proxy_f1 = {
    "GIN\n(논문)": 28.70,
    "GIN+EU\n(논문)": 47.73,
    "PNA\n(논문)": 56.77,
    "GFP+LightGBM\n(논문)": 62.86,
    "GFP+XGBoost\n(논문)": 63.23,
}
ours_f1 = {
    "본 구현\nXGBoost": gbt["summary"]["xgboost"]["f1"]["mean"] * 100,
    "본 구현\nLightGBM": gbt["summary"]["lightgbm"]["f1"]["mean"] * 100,
    "본 구현\nGNN (평균)": gnn["summary"]["f1"]["mean"] * 100,
    "본 구현\nGNN (최고 시드)": gnn["summary"]["f1"]["max"] * 100,
}
labels = list(paper_hi_small_pr_proxy_f1.keys()) + list(ours_f1.keys())
values = list(paper_hi_small_pr_proxy_f1.values()) + list(ours_f1.values())
colors_bar = ["#9fb4c7"] * len(paper_hi_small_pr_proxy_f1) + ["#d9663b", "#2c7f6e", "#c9a227", "#7a4fb5"]

fig, ax = plt.subplots(figsize=(10, 5.2))
bars = ax.bar(labels, values, color=colors_bar)
ax.set_ylabel("Minority-class F1 (%)")
ax.set_title("HI-Small: GBT vs GNN vs 논문 베이스라인 (F1, threshold=0.5)")
ax.set_ylim(0, 75)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
plt.xticks(rotation=20, ha="right", fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT / "chart_gnn_vs_gbt_comparison.png", dpi=150)
plt.close()

# --- Chart: per-seed PR-AUC variance for GNN vs GBT ---
fig, ax = plt.subplots(figsize=(7.5, 4.5))
gbt_x = [r["pr_auc"] for r in gbt["raw_results"]["xgboost"]]
gbt_l = [r["pr_auc"] for r in gbt["raw_results"]["lightgbm"]]
gnn_v = [r["pr_auc"] for r in gnn["raw_results"]]
positions = [1, 2, 3]
bp = ax.boxplot([gbt_x, gbt_l, gnn_v], positions=positions, widths=0.5, patch_artist=True,
                labels=["XGBoost", "LightGBM", "GNN"])
for patch, c in zip(bp["boxes"], ["#d9663b", "#2c7f6e", "#7a4fb5"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for i, vals in enumerate([gbt_x, gbt_l, gnn_v], start=1):
    ax.scatter([i] * len(vals), vals, color="black", zorder=5, s=18)
ax.set_ylabel("Test PR-AUC")
ax.set_title("모델별 시드 간 PR-AUC 변동폭 (3-seed)")
plt.tight_layout()
plt.savefig(OUT / "chart_seed_variance.png", dpi=150)
plt.close()

print("gnn charts written")
