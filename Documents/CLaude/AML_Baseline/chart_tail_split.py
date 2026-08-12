import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
R = json.loads((OUT / "results_tail_split.json").read_text(encoding="utf-8"))

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#8a8a86"

subsets = ["combined", "normal", "tail"]
subset_labels = ["전체 test\n(원래 보고치)", "정상 구간만\n(9/8~9/10)", "꼬리 구간만\n(9/11~9/18)"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, metric, title in zip(axes, ["f1", "pr_auc"], ["Minority-class F1", "PR-AUC"]):
    xgb_vals = [R["summary"]["xgboost"][s][metric]["mean"] for s in subsets]
    lgb_vals = [R["summary"]["lightgbm"][s][metric]["mean"] for s in subsets]
    x = np.arange(len(subsets))
    w = 0.35
    b1 = ax.bar(x - w/2, xgb_vals, w, label="XGBoost", color=BLUE)
    b2 = ax.bar(x + w/2, lgb_vals, w, label="LightGBM", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(subset_labels, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + max(xgb_vals+lgb_vals)*0.02,
                     f"{v:.3f}" if metric == "pr_auc" else f"{v*100:.1f}%",
                     ha="center", fontsize=8.5)
    ax.set_ylim(0, max(xgb_vals + lgb_vals) * 1.25)

axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("HI-Small GBT 재평가: 전체 test vs 정상/꼬리 구간 분리", fontsize=13, y=1.02)

# annotate class balance shift
axes[1].annotate("이 구간은 양성 비율 59%\n(655/1,108) — 정상 test의\n0.11%와는 다른 분포",
                  xy=(1.85, 0.82),
                  xytext=(0.15, 0.55), fontsize=8.5, color="#c0392b",
                  arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

plt.tight_layout()
plt.savefig(OUT / "chart_tail_split.png", dpi=150, bbox_inches="tight")
print("chart written")
