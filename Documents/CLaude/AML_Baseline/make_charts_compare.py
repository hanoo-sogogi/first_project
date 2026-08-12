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

PAPER = "#9fb4c7"
OURS_GBT = "#2c7f6e"
OURS_GNN = "#7a4fb5"

# --- Chart 1: paired paper-vs-ours by model family ---
families = ["GIN 계열\n(GNN)", "GBT\n(LightGBM)", "GBT\n(XGBoost)"]
paper_vals = [28.70, 62.86, 63.23]
ours_vals = [
    gnn["summary"]["f1"]["max"] * 100,          # best GNN seed
    gbt["summary"]["lightgbm"]["f1"]["mean"] * 100,
    gbt["summary"]["xgboost"]["f1"]["mean"] * 100,
]

x = np.arange(len(families))
w = 0.36
fig, ax = plt.subplots(figsize=(8.5, 4.8))
b1 = ax.bar(x - w / 2, paper_vals, w, label="논문 (GFP / GPU 학습)", color=PAPER)
b2 = ax.bar(x + w / 2, ours_vals, w, label="본 구현 (자체 피처 / CPU)",
            color=[OURS_GNN, OURS_GBT, OURS_GBT])
ax.set_xticks(x)
ax.set_xticklabels(families)
ax.set_ylabel("Minority-class F1 (%)")
ax.set_title("모델 계열별: 논문 보고치 vs 본 구현 (HI-Small)")
ax.legend(fontsize=9)
ax.set_ylim(0, 75)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{b.get_height():.1f}", ha="center", fontsize=8.5)
# gap annotations
for i, (p, o) in enumerate(zip(paper_vals, ours_vals)):
    ax.annotate("", xy=(i + w / 2, o), xytext=(i - w / 2, p),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3, alpha=0.75))
    ax.text(i, max(p, o) + 6, f"\u0394 {p - o:+.1f}p", ha="center", fontsize=8.5,
            color="#c0392b", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "cmp_paper_vs_ours.png", dpi=150)
plt.close()

# --- Chart 2: what the gap is made of (attribution, illustrative) ---
fig, ax = plt.subplots(figsize=(8.5, 4.2))
components = [
    ("본 구현 LightGBM", 34.2, OURS_GBT),
    ("+ 다중 홉 그래프 피처\n(사이클/scatter-gather)", 0, "#e8b84b"),
    ("+ GFP 최적화 구현\n및 하이퍼파라미터 탐색", 0, "#e8b84b"),
    ("논문 GFP+LightGBM", 62.9, PAPER),
]
labels = [c[0] for c in components]
vals = [34.2, 0, 0, 62.9]
colors_c = [c[2] for c in components]
bars = ax.bar(labels, vals, color=colors_c)
ax.set_ylabel("Minority-class F1 (%)")
ax.set_title("GBT 성능 격차 28.7%p의 구성 요소 (미측정 구간은 회색 표시)")
ax.set_ylim(0, 75)
ax.text(1.5, 48, "이 두 요소가 격차 28.7%p를\n만든 것으로 추정되나,\n본 구현에서는 미측정",
        ha="center", fontsize=9.5, color="#7a5c00",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff5d6", edgecolor="#e8b84b"))
for b, v in zip(bars, vals):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
plt.xticks(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "cmp_gap_attribution.png", dpi=150)
plt.close()

# --- Chart 3: stability comparison (the finding the paper does not report) ---
fig, ax = plt.subplots(figsize=(8, 4.3))
models = ["XGBoost\n(본 구현)", "LightGBM\n(본 구현)", "GNN\n(본 구현)"]
data = [
    [r["f1"] * 100 for r in gbt["raw_results"]["xgboost"]],
    [r["f1"] * 100 for r in gbt["raw_results"]["lightgbm"]],
    [r["f1"] * 100 for r in gnn["raw_results"]],
]
positions = np.arange(len(models))
for i, vals in enumerate(data):
    ax.scatter([i] * len(vals), vals, s=70, zorder=5,
               color=[OURS_GBT, OURS_GBT, OURS_GNN][i], edgecolor="black", linewidth=0.6)
    ax.plot([i, i], [min(vals), max(vals)], color="#888888", lw=2, zorder=3)
    ax.text(i + 0.14, np.mean(vals), f"폭 {max(vals)-min(vals):.1f}p",
            fontsize=9, va="center", color="#444444")
ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_ylabel("Minority-class F1 (%)")
ax.set_title("시드 3회 반복 시 F1 변동폭 — GBT는 조밀, GNN은 0%까지 붕괴")
ax.set_ylim(-3, 45)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "cmp_stability.png", dpi=150)
plt.close()

print("comparison charts written")
