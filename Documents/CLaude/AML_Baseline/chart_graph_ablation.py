import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = __import__("pathlib").Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
v2 = json.loads((OUT / "results_v2.json").read_text(encoding="utf-8"))
v3 = json.loads((OUT / "results_v3_nograph.json").read_text(encoding="utf-8"))

BLUE = "#2a78d6"
ORANGE = "#eb6834"

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
models = ["XGBoost", "LightGBM"]
x = np.arange(len(models))
w = 0.35

for ax, metric, label in zip(axes, ["f1", "pr_auc"], ["Minority-class F1", "PR-AUC"]):
    with_graph = [v2["summary"]["xgboost"][metric]["mean"], v2["summary"]["lightgbm"][metric]["mean"]]
    no_graph = [v3["summary"]["xgboost"][metric]["mean"], v3["summary"]["lightgbm"][metric]["mean"]]
    b1 = ax.bar(x - w/2, with_graph, w, label="그래프 피처 포함(v2)", color=BLUE)
    b2 = ax.bar(x + w/2, no_graph, w, label="그래프 피처 제외(v3)", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(label, fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fmt = (lambda v: f"{v*100:.1f}%") if metric == "f1" else (lambda v: f"{v:.3f}")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(with_graph+no_graph)*0.02,
                     fmt(b.get_height()), ha="center", fontsize=9)
    ax.set_ylim(0, max(with_graph+no_graph)*1.25)

axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("그래프 집계 피처 유/무 — XGBoost는 도움, LightGBM은 오히려 방해", fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig(OUT / "chart_graph_ablation.png", dpi=150, bbox_inches="tight")
print("chart written")
