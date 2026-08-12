import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --- Chart 1: PR-AUC comparison against paper baselines (HI-Small, Table 2) ---
paper_hi_small_f1 = {
    "GIN": 28.70,
    "GIN+EU": 47.73,
    "PNA": 56.77,
    "GFP+LightGBM\n(paper, HI-Small)": 62.86,
    "GFP+XGBoost\n(paper, HI-Small)": 63.23,
}
ours_f1 = {
    "Ours: XGBoost": results["summary"]["xgboost"]["f1"]["mean"] * 100,
    "Ours: LightGBM": results["summary"]["lightgbm"]["f1"]["mean"] * 100,
}

labels = list(paper_hi_small_f1.keys()) + list(ours_f1.keys())
values = list(paper_hi_small_f1.values()) + list(ours_f1.values())
colors = ["#9fb4c7"] * len(paper_hi_small_f1) + ["#d9663b", "#2c7f6e"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, values, color=colors)
ax.set_ylabel("Minority-class F1 (%)")
ax.set_title("HI-Small: 본 구현 vs 논문 보고 베이스라인 (F1, threshold=0.5)")
ax.set_ylim(0, 75)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "chart_f1_comparison.png", dpi=150)
plt.close()

# --- Chart 2: Precision@K ---
ks = [100, 500, 1000, 2000]
xgb_p = [results["summary"]["xgboost"][f"precision_at_{k}"]["mean"] for k in ks]
lgb_p = [results["summary"]["lightgbm"][f"precision_at_{k}"]["mean"] for k in ks]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(ks))
w = 0.35
ax.bar(x - w / 2, xgb_p, w, label="XGBoost", color="#d9663b")
ax.bar(x + w / 2, lgb_p, w, label="LightGBM", color="#2c7f6e")
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in ks])
ax.set_ylabel("Precision@K")
ax.set_title("알람 상위 K건 기준 정밀도 (3-seed 평균, test 1,015,882건)")
ax.legend()
ax.set_ylim(0, 1)
for i, v in enumerate(xgb_p):
    ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate(lgb_p):
    ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "chart_precision_at_k.png", dpi=150)
plt.close()

# --- Chart 3: XGBoost feature importance ---
fi = results["feature_importance"]["xgboost"]
fi_sorted = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
names = [k for k, _ in fi_sorted]
vals = [v for _, v in fi_sorted]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(names[::-1], vals[::-1], color="#d9663b")
ax.set_xlabel("XGBoost feature importance (gain-normalized)")
ax.set_title("피처 중요도 (XGBoost, seed 1)")
plt.tight_layout()
plt.savefig(OUT / "chart_feature_importance.png", dpi=150)
plt.close()

print("charts written")
