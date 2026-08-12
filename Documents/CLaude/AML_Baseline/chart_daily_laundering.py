import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv(
    r"C:\Users\aica_\AppData\Local\Temp\claude\C--Users-aica-\0f755edb-4aeb-4e08-9c7c-667d266fa8c0\scratchpad\daily_laundering.csv",
    parse_dates=["Date"],
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#8a8a86"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

tail_start = pd.Timestamp("2022-09-11")

# --- top: daily transaction volume ---
colors1 = [MUTED if d >= tail_start else BLUE for d in daily["Date"]]
ax1.bar(daily["Date"], daily["total"], color=colors1, width=0.8)
ax1.set_ylabel("일별 거래 건수")
ax1.set_title("HI-Small: 날짜별 거래량 vs 이상거래(라벨링) 비율", fontsize=13, loc="left")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#e5e5e3", linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)

# annotate the two regimes
ax1.annotate("정상 구간\n(9/1~9/10)\n~5.08M건 (99.98%)",
             xy=(pd.Timestamp("2022-09-05"), 900000), fontsize=9, color=BLUE,
             ha="center")
ax1.annotate("꼬리 구간\n(9/11~9/18)\n1,108건 (0.02%)",
             xy=(pd.Timestamp("2022-09-14"), 200000), fontsize=9, color=MUTED,
             ha="center")

# --- bottom: laundering rate ---
colors2 = [MUTED if d >= tail_start else ORANGE for d in daily["Date"]]
ax2.bar(daily["Date"], daily["rate_pct"], color=colors2, width=0.8)
ax2.set_ylabel("이상거래 비율 (%)")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color="#e5e5e3", linewidth=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

ax2.axvline(pd.Timestamp("2022-09-10T12:00:00"), color="#c0392b", linestyle="--", linewidth=1)
ax2.annotate("9/10 이후 거래량 급감,\n이상거래 비율만 56~73%로 급등",
             xy=(pd.Timestamp("2022-09-14"), 45), fontsize=9, color="#c0392b", ha="center")

for d, r, l in zip(daily["Date"], daily["rate_pct"], daily["laundering"]):
    if d >= tail_start:
        ax2.text(d, r + 2, f"{r:.0f}%", ha="center", fontsize=7.5, color="#444")

plt.tight_layout()
plt.savefig(r"C:\Users\aica_\Documents\CLaude\AML_Baseline\chart_daily_laundering.png", dpi=150)
print("chart written")
