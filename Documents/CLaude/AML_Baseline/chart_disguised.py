import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#8a8a86"

other = pd.read_csv(r"C:\Users\aica_\AppData\Local\Temp\claude\C--Users-aica-\0f755edb-4aeb-4e08-9c7c-667d266fa8c0\scratchpad\hi_small_other_laundering.csv")
trans = pd.read_csv(r"C:\Users\aica_\Documents\CLaude\HI-Small_Trans.csv")
trans.columns = ["Timestamp","FromBankID","FromAccount","ToBankID","ToAccount",
                  "AmountReceived","ReceivingCurrency","AmountPaid","PaymentCurrency",
                  "PaymentFormat","IsLaundering"]
other["FromNode"] = other["FromBankID"].astype(str) + "_" + other["FromAccount"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

# --- panel 1: payment format over/under representation ---
overall_fmt = trans["PaymentFormat"].value_counts(normalize=True)
other_fmt = other["PaymentFormat"].value_counts(normalize=True)
formats = ["ACH", "Cheque", "Credit Card", "Cash", "Bitcoin", "Wire", "Reinvestment"]
ratio = [(other_fmt.get(f, 0) / overall_fmt.get(f, 1e-9)) for f in formats]
colors = [ORANGE if r > 1 else BLUE if r > 0 else MUTED for r in ratio]
y = np.arange(len(formats))
axes[0].barh(y, ratio, color=colors)
axes[0].axvline(1, color="#444", linewidth=1, linestyle="--")
axes[0].set_yticks(y)
axes[0].set_yticklabels(formats)
axes[0].set_xlabel("위장 이상거래 비중 / 전체 거래 비중 (배율)")
axes[0].set_title("결제수단별 위장 이상거래 쏠림 정도", fontsize=11.5)
axes[0].spines[["top", "right"]].set_visible(False)
for yi, r in zip(y, ratio):
    axes[0].text(r + 0.1, yi, f"{r:.2f}x", va="center", fontsize=8.5)
axes[0].invert_yaxis()

# --- panel 2: sender concentration (cumulative share by top senders) ---
from_counts = other["FromNode"].value_counts().sort_values(ascending=False).reset_index(drop=True)
cum_share = from_counts.cumsum() / from_counts.sum()
n_senders = len(from_counts)
x_share = (np.arange(1, n_senders + 1)) / n_senders * 100
axes[1].plot(x_share, cum_share * 100, color=BLUE, linewidth=2)
axes[1].plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--")
axes[1].set_xlabel("송금계좌 순위 (누적, %)")
axes[1].set_ylabel("위장 이상거래 누적 비중 (%)")
axes[1].set_title("송금계좌 집중도 (상위 소수 계좌가 다수 발신)", fontsize=11.5)
axes[1].spines[["top", "right"]].set_visible(False)
top2_share = from_counts.iloc[:2].sum() / from_counts.sum() * 100
axes[1].annotate(f"상위 2개 계좌가\n전체의 {top2_share:.0f}%\n(243건+158건)",
                  xy=(2/n_senders*100, top2_share), xytext=(25, 45),
                  fontsize=9, color="#c0392b",
                  arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)

fig.suptitle("HI-Small \"위장(기타)\" 이상거래 1,968건의 특성", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\aica_\Documents\CLaude\AML_Baseline\chart_disguised.png", dpi=150, bbox_inches="tight")
print("chart written")
