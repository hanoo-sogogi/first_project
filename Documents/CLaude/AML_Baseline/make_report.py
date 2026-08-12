# -*- coding: utf-8 -*-
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline")
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
G = json.loads((OUT / "results_gnn.json").read_text(encoding="utf-8"))

pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBold", r"C:\Windows\Fonts\malgunbd.ttf"))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("KTitle", fontName="MalgunBold", fontSize=20, leading=26,
                           spaceAfter=6, textColor=colors.HexColor("#1a1a1a")))
styles.add(ParagraphStyle("KSubtitle", fontName="Malgun", fontSize=11, leading=16,
                           textColor=colors.HexColor("#555555"), spaceAfter=18))
styles.add(ParagraphStyle("KH1", fontName="MalgunBold", fontSize=14, leading=20,
                           spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a1a1a")))
styles.add(ParagraphStyle("KH2", fontName="MalgunBold", fontSize=11.5, leading=16,
                           spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#333333")))
styles.add(ParagraphStyle("KBody", fontName="Malgun", fontSize=9.7, leading=15,
                           spaceAfter=6, textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle("KBullet", fontName="Malgun", fontSize=9.7, leading=15,
                           spaceAfter=3, leftIndent=12, textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle("KCaption", fontName="Malgun", fontSize=8.5, leading=12,
                           textColor=colors.HexColor("#666666"), spaceAfter=12,
                           alignment=1))
styles.add(ParagraphStyle("KCode", fontName="Malgun", fontSize=8.3, leading=12,
                           textColor=colors.HexColor("#1a1a1a"),
                           backColor=colors.HexColor("#f2f2f2"), spaceAfter=8,
                           leftIndent=8, borderPadding=6))

TABLE_HEAD_BG = colors.HexColor("#2c3e50")
TABLE_ALT_BG = colors.HexColor("#f4f6f7")


def make_table(data, col_widths=None, align_first_left=True):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "MalgunBold"),
        ("FONTNAME", (0, 1), (-1, -1), "Malgun"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.7),
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_BG]),
    ]
    if align_first_left:
        style.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return t


story = []

# ============================================================ Title page
story.append(Spacer(1, 10 * mm))
story.append(Paragraph("IBM AML (HI-Small) 자금세탁 탐지 모델", styles["KTitle"]))
story.append(Paragraph(
    "GBT(그래프 피처) vs GNN(메시지 패싱) 두 계열 모델 구현 및 평가 보고서",
    styles["KSubtitle"]))
story.append(Paragraph(
    "근거 논문: Altman et al., <i>\"Realistic Synthetic Financial Transactions for "
    "Anti-Money Laundering Models\"</i>, NeurIPS 2023 Datasets &amp; Benchmarks Track "
    "(arXiv:2306.16424) &mdash; IBM이 공개한 AMLworld 합성 데이터셋과 벤치마크 방법론",
    styles["KBody"]))
story.append(Paragraph("작성일: 2026-08-06", styles["KBody"]))
story.append(Spacer(1, 6 * mm))

summary_data = [
    ["항목", "값"],
    ["데이터셋", f"IBM AML HI-Small ({R['n_transactions']:,}건 거래, "
                 f"양성 {R['n_positive']:,}건 = {R['n_positive']/R['n_transactions']*100:.3f}%)"],
    ["모델", "XGBoost, LightGBM (GBT) + GINEConv 기반 GNN"],
    ["최고 성능", f"LightGBM 3-seed 평균 PR-AUC {R['summary']['lightgbm']['pr_auc']['mean']:.3f}, "
                 f"F1 {R['summary']['lightgbm']['f1']['mean']*100:.1f}%"],
    ["운영 관점", f"알람 상위 500건 기준 정밀도 "
                 f"{R['summary']['lightgbm']['precision_at_500']['mean']*100:.0f}%"],
    ["추가 구현: GNN", f"GINEConv 기반 edge-classification, 3-seed PR-AUC "
                     f"{G['summary']['pr_auc']['min']:.3f}~{G['summary']['pr_auc']['max']:.3f} "
                     f"(시드 간 변동 큼, 5-3절 참조)"],
]
story.append(make_table(summary_data, col_widths=[35 * mm, 125 * mm]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "본 보고서는 IBM AML 데이터셋(HI-Small)에 대해 두 가지 다른 계열의 모델을 처음부터 "
    "구현·평가한 결과를 담고 있다: (1) 그래프 구조 기반 피처를 직접 설계한 GBT(XGBoost/"
    "LightGBM), (2) 메시지 패싱으로 구조를 스스로 학습하는 GNN(GINEConv 기반). 논문이 "
    "사용한 독점 도구인 Graph Feature Preprocessor(GFP, 사이클/scatter-gather 패턴 탐지 "
    "포함) 대신, GBT에는 leak-free 계좌 집계 피처를 자체 구현했고, GNN은 그런 손설계 "
    "피처 없이 순수 메시지 패싱만으로 구조를 학습하도록 구성했다.",
    styles["KBody"]))

story.append(PageBreak())

# ============================================================ 1. Data & problem
story.append(Paragraph("1. 데이터 및 문제 정의", styles["KH1"]))
story.append(Paragraph(
    "IBM Research가 NeurIPS 2023에서 공개한 AMLworld 합성 거래 생성기의 산출물 중 "
    "HI-Small(High Illicit ratio, Small) 데이터셋을 사용했다. 다중 에이전트 기반 가상 "
    "경제 시뮬레이션으로 생성된 데이터로, 계좌 간 송수신 거래마다 실제 자금세탁 여부의 "
    "완전한 정답 라벨(Is Laundering)이 부여되어 있다 &mdash; 실거래 데이터에서는 얻을 수 없는 특징이다.",
    styles["KBody"]))

data_table = [
    ["항목", "값"],
    ["전체 거래 수", f"{R['n_transactions']:,}건"],
    ["자금세탁 거래 수", f"{R['n_positive']:,}건 ({R['n_positive']/R['n_transactions']*100:.4f}%)"],
    ["클래스 불균형", f"약 1 : {int(R['n_transactions']/R['n_positive'])}"],
    ["학습 구간(60%)", f"{R['split']['train']:,}건 (양성 {R['split']['train_pos']:,}건)"],
    ["검증 구간(20%)", f"{R['split']['val']:,}건"],
    ["평가 구간(20%, 시간상 최신)", f"{R['split']['test']:,}건 (양성 {R['split']['test_pos']:,}건)"],
]
story.append(make_table(data_table, col_widths=[55 * mm, 105 * mm]))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph(
    "\"전부 정상\"이라고만 예측해도 accuracy는 99.90%에 달하므로, accuracy는 이 문제에서 "
    "의미가 없다. 본 보고서는 소수 클래스(자금세탁) 기준 F1 / PR-AUC / Precision@K를 "
    "주 평가지표로 사용한다.",
    styles["KBody"]))

# ============================================================ 2. Methodology
story.append(Paragraph("2. 방법론", styles["KH1"]))

story.append(Paragraph("2-1. 통화 정규화 (USD 환산)", styles["KH2"]))
story.append(Paragraph(
    "원본 데이터는 15개 통화가 혼재하며 통화별 금액 스케일 차이가 최대 100만 배에 달해, "
    "원시 금액을 그대로 사용하면 모델이 \"금액의 크기\"가 아니라 \"어떤 통화로 결제했는가\"를 "
    "학습하게 된다. 동일 거래 내 Payment Currency와 Receiving Currency가 다른 환전 거래"
    "(전체의 약 1.4%)에서 Amount Received / Amount Paid 비율의 통화쌍별 중앙값을 구하고, "
    "USD를 기준점으로 BFS 방식으로 전 통화의 USD 환산율을 도출했다. 이후 "
    "log1p(금액 × USD환산율)을 최종 금액 피처로 사용했다.",
    styles["KBody"]))

story.append(Paragraph("2-2. 피처 엔지니어링", styles["KH2"]))
feat_table = [
    ["구분", "피처", "설명"],
    ["거래 자체", "LogAmountUSD", "USD 환산 후 log1p 변환한 거래 금액"],
    ["", "PaymentFormat", "결제수단 (ACH/Wire/Cheque/Credit Card 등, 범주형)"],
    ["", "IsSelfLoop", "송금계좌 = 수취계좌 여부"],
    ["", "IsSameBank", "송금은행 = 수취은행 여부"],
    ["", "Hour / DayOfWeek", "거래 시각 / 요일"],
    ["계좌 그래프\n(학습구간만 집계)", "out_degree / in_degree", "계좌의 송금/수신 거래 건수"],
    ["", "out_unique_cp / in_unique_cp", "고유 거래상대방 수"],
    ["", "out_avg_amt / in_avg_amt", "평균 송금/수신 금액(USD, log)"],
    ["", "in_unique_banks", "수취계좌에 입금하는 고유 은행 수 (팬인 분산도)"],
    ["", "pair_prior_count", "동일 (송금계좌\u2192수취계좌) 쌍의 과거 거래 반복 횟수"],
]
story.append(make_table(feat_table, col_widths=[28 * mm, 42 * mm, 90 * mm]))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph(
    "<b>데이터 누수 방지:</b> 모든 계좌 그래프 집계 피처는 학습 구간(전체 60%) 거래만으로 "
    "계산한 뒤, 검증/평가 구간에는 이미 계산된 값을 고정 매핑으로 부여했다. 전체 기간으로 "
    "집계할 경우 미래 거래 정보가 새어 들어가 성능이 부풀려지는 것을 사전 확인했기 때문이다. "
    "환전 거래 여부(is_fx) 피처는 데이터 생성 과정의 인공물(환전 거래 중 자금세탁 0건이라는 "
    "비현실적 패턴)로 판단되어 의도적으로 제외했다.",
    styles["KBody"]))

story.append(Paragraph("2-3. 학습/평가 분할 및 클래스 불균형 대응", styles["KH2"]))
story.append(Paragraph(
    "논문과 동일하게 타임스탬프 기준 60/20/20 시간 분할을 사용했다(무작위 분할은 동일 "
    "자금세탁 시도의 거래가 학습/평가 양쪽에 섞여 성능이 부풀려지므로 사용하지 않음). "
    "학습셋은 양성 전량을 유지하고 음성만 40만 건으로 다운샘플링하여 학습 시간을 단축했다 "
    "(검증/평가셋은 전량 유지하므로 평가지표에는 영향 없음).",
    styles["KBody"]))

story.append(Paragraph("2-4. 모델 및 하이퍼파라미터", styles["KH2"]))
story.append(Paragraph(
    "XGBoost(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, "
    "colsample_bytree=0.8)와 LightGBM(n_estimators=400, num_leaves=15, "
    "min_child_samples=50, learning_rate=0.05, reg_lambda=1.0)을 각각 3개 시드로 학습했다.",
    styles["KBody"]))
story.append(Paragraph(
    "<b>구현 중 발견한 함정:</b> 학습셋을 이미 다운샘플링한 상태에서 scale_pos_weight를 "
    "추가로 적용하면(다운샘플링 후 비율 기준 약 174배) 두 모델 모두 예측 확률이 threshold "
    "0.5를 훨씬 초과하도록 밀려나 정밀도가 붕괴했다 &mdash; 특히 LightGBM은 leaf-wise 트리 "
    "성장 특성상 PR-AUC가 0.31 \u2192 0.01 수준까지 무너지는 것을 확인했다. scale_pos_weight를 "
    "제거하고 다운샘플링만으로 불균형을 보정하도록 수정한 뒤 두 모델 모두 정상화되었다 "
    "(XGBoost F1 0.13\u21920.33, LightGBM F1 0.01\u21920.34). 아래 결과는 모두 수정된 설정 기준이다.",
    styles["KBody"]))

story.append(PageBreak())

# ============================================================ 3. Results
story.append(Paragraph("3. 결과", styles["KH1"]))
story.append(Paragraph(
    f"평가 구간(시간상 최신 20%, {R['split']['test']:,}건, 양성 {R['split']['test_pos']:,}건) "
    "기준, 3개 시드 학습 결과의 평균값이다.",
    styles["KBody"]))

res_table = [
    ["모델", "F1", "Precision", "Recall", "PR-AUC", "Recall@Prec90%"],
]
for name, key in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
    s = R["summary"][key]
    res_table.append([
        name,
        f"{s['f1']['mean']*100:.1f}% [{s['f1']['min']*100:.1f}-{s['f1']['max']*100:.1f}]",
        f"{s['precision']['mean']*100:.1f}%",
        f"{s['recall']['mean']*100:.1f}%",
        f"{s['pr_auc']['mean']:.3f} [{s['pr_auc']['min']:.3f}-{s['pr_auc']['max']:.3f}]",
        f"{s['recall_at_precision90']['mean']*100:.2f}%",
    ])
story.append(make_table(res_table, col_widths=[26 * mm, 34 * mm, 24 * mm, 22 * mm, 34 * mm, 30 * mm]))
story.append(Spacer(1, 3 * mm))

pk_table = [["모델", "P@100", "P@500", "P@1000", "P@2000"]]
for name, key in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
    s = R["summary"][key]
    pk_table.append([
        name,
        f"{s['precision_at_100']['mean']*100:.0f}%",
        f"{s['precision_at_500']['mean']*100:.0f}%",
        f"{s['precision_at_1000']['mean']*100:.0f}%",
        f"{s['precision_at_2000']['mean']*100:.0f}%",
    ])
story.append(make_table(pk_table, col_widths=[30 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm]))
story.append(Paragraph(
    "Precision@K: 모델이 위험도 순으로 정렬한 상위 K건 알람 중 실제 자금세탁 거래의 비율. "
    "운영 관점(하루 몇 건의 알람을 실사할 것인가)에서 F1보다 직관적인 지표다.",
    styles["KCaption"]))

story.append(Image(str(OUT / "chart_precision_at_k.png"), width=150 * mm, height=97 * mm))

story.append(Spacer(1, 4 * mm))
story.append(Paragraph("3-1. 논문 벤치마크와의 비교 (HI-Small, Table 2)", styles["KH2"]))
story.append(Image(str(OUT / "chart_f1_comparison.png"), width=150 * mm, height=79 * mm))
story.append(Paragraph(
    "회색 막대는 논문 Table 2에 보고된 HI-Small 기준 minority-class F1이다. 본 구현(주황/초록)은 "
    "그래프 신경망 GIN(28.7%)보다는 높지만, 거래상대방 정보를 재귀적으로 전파하는 GIN+EU(47.7%), "
    "PNA(56.8%), 그리고 논문이 자체 개발한 GFP(사이클 열거, scatter-gather 서브그래프 탐지 포함) "
    "+ GBT 조합(62.9~63.2%)에는 못 미친다. 이는 본 구현이 사용한 그래프 피처가 1-hop 집계 "
    "(직접 연결된 이웃의 통계)에 그치고, 사이클·팬아웃/팬인 패턴 자체를 구조적으로 탐지하지는 "
    "않기 때문으로 해석된다.",
    styles["KBody"]))

story.append(PageBreak())

story.append(Paragraph("3-2. 피처 중요도", styles["KH2"]))
story.append(Image(str(OUT / "chart_feature_importance.png"), width=150 * mm, height=107 * mm))
story.append(Paragraph(
    "가장 예측력이 높았던 피처는 <b>pair_prior_count</b>(동일 송수신 계좌 쌍의 과거 거래 "
    "반복 횟수)로, 특정 계좌 쌍 간의 비정상적으로 잦은 반복 송금이 세탁 패턴과 강하게 연관됨을 "
    "시사한다. 그 뒤를 <b>IsSameBank</b>, <b>PaymentFormat</b>, <b>IsSelfLoop</b>, "
    "<b>out_degree</b>가 잇는다. EDA에서 확인된 대로 ACH 결제수단의 세탁률이 타 수단 대비 "
    "20배 이상 높다는 사실이 PaymentFormat의 높은 중요도로 이어진 것으로 보인다.",
    styles["KBody"]))

# ============================================================ 4. Discussion
story.append(Paragraph("4. 평가 및 해석", styles["KH1"]))

story.append(Paragraph("4-1. 현실적 성능 수준", styles["KH2"]))
story.append(Paragraph(
    f"LightGBM 기준 알람 상위 500건을 실사한다면 그중 약 "
    f"{R['summary']['lightgbm']['precision_at_500']['mean']*100:.0f}%가 실제 자금세탁 거래일 "
    f"것으로 기대된다. 반면 정밀도 90% 이상을 유지하려면 재현율이 "
    f"{R['summary']['lightgbm']['recall_at_precision90']['mean']*100:.1f}% 수준까지 떨어진다 "
    "&mdash; 즉 전체 세탁 거래의 극히 일부만 매우 높은 확신도로 잡아낼 수 있다는 뜻이다. "
    "\"정밀도 90%\" 같은 목표는 재현율·알람 건수와 함께 재정의되어야 실무적으로 의미가 있다.",
    styles["KBody"]))

story.append(Paragraph("4-2. XGBoost와 LightGBM의 차이", styles["KH2"]))
story.append(Paragraph(
    "PR-AUC(순위 품질)는 두 모델이 비슷하지만(0.290 vs 0.290), threshold 0.5에서의 "
    "정밀도-재현율 균형은 다르다: LightGBM은 더 보수적으로 양성을 예측해 정밀도가 높고"
    "(42.6% vs 32.1%) 재현율이 낮으며(28.5% vs 34.2%), P@100/P@500에서도 LightGBM이 "
    "다소 앞선다. 운영상 \"소수의 고확신 알람\"이 목표라면 LightGBM이, \"더 많은 후보를 "
    "덜 놓치는\" 것이 목표라면 XGBoost가 유리하다.",
    styles["KBody"]))

story.append(Paragraph("4-3. 한계", styles["KH2"]))
limits = [
    "<b>구조적 그래프 패턴 미탐지:</b> 사이클, scatter-gather, bipartite 등 논문이 정의한 "
    "8가지 세탁 패턴(Patterns.txt에 라벨링되어 있음)을 명시적으로 탐지하는 피처는 구현하지 "
    "않았다. 1-hop 집계만으로는 다중 홉(예: CYCLE 평균 5+ 계좌 경유) 패턴을 포착하기 어렵다.",
    "<b>단일 데이터셋:</b> HI-Small 한 가지 버전만 검증했다. 이상거래 비율이 더 낮은 "
    "LI 계열이나 Medium/Large 규모에서 동일 경향이 재현되는지는 확인되지 않았다.",
    "<b>시드 3회:</b> 양성 클래스가 학습 구간에 2,297건뿐이라 시드 간 변동이 존재한다"
    "(XGBoost F1 0.315~0.343). 결론을 확정하려면 5~10회 반복이 바람직하다.",
    "<b>Threshold 미조정:</b> 모든 F1/Precision/Recall은 기본 threshold 0.5 기준이며, "
    "논문의 Precision-Recall 곡선 보고 방식과 동일하게 threshold 자체는 별도로 최적화하지 "
    "않았다.",
]
for l in limits:
    story.append(Paragraph("&bull; " + l, styles["KBullet"]))

story.append(PageBreak())

# ============================================================ 5. GNN
story.append(Paragraph("5. 추가 구현: GNN (그래프 신경망)", styles["KH1"]))
story.append(Paragraph(
    "GBT 베이스라인과는 다른 계열의 모델로 논문이 벤치마크한 GNN(GIN/GIN+EU/PNA) 계열을 "
    "직접 구현해 비교했다. GBT가 사람이 설계한 그래프 집계 피처(out_degree, "
    "pair_prior_count 등)에 의존하는 반면, GNN은 메시지 패싱을 통해 이웃 구조 정보를 "
    "스스로 학습한다는 점이 핵심 차이다.",
    styles["KBody"]))

story.append(Paragraph("5-1. 그래프 구성 및 아키텍처", styles["KH2"]))
story.append(Paragraph(
    "논문의 데이터 분할 방식(Section 4)을 그대로 따라 train/val/test 3개의 그래프 스냅샷을 "
    "구성했다: train 그래프는 학습구간 거래만, val 그래프는 학습+검증구간 거래(검증구간 "
    "거래만 채점), test 그래프는 전체 거래(평가구간 거래만 채점)로 구성해, 이웃 노드의 "
    "메시지 패싱에는 더 넓은 문맥을 주되 손실/평가는 해당 구간 거래에만 적용했다. "
    "노드 피처는 각 스냅샷 내에서 leak-free하게 계산한 계좌별 통계(입/출 거래건수, "
    "평균 금액, 고유 거래상대방 수, 총 6차원)를, 엣지 피처는 거래 자체 속성(USD 환산 "
    "금액, 결제수단 임베딩, 자기거래/동일은행 여부, 시각·요일)을 사용했다 &mdash; GBT에서 "
    "썼던 pair_prior_count 등 손으로 설계한 그래프 집계 피처는 의도적으로 제외해, GNN이 "
    "구조 정보를 메시지 패싱만으로 얼마나 학습할 수 있는지를 관찰했다.",
    styles["KBody"]))
story.append(Paragraph(
    "모델은 2-layer GINEConv(엣지 피처를 반영하는 GIN 변형) + 잔차 연결로 노드 임베딩을 "
    "구성한 뒤, 거래(엣지) 분류를 위해 [송금계좌 임베딩, 수취계좌 임베딩, 엣지 피처]를 "
    "결합한 MLP 판독head를 두었다 &mdash; 논문의 GIN+EU(edge-update) 계열과 같은 계통이다. "
    "GPU 없이 CPU만으로 학습했으며, 전체 그래프(300만~500만 엣지)를 배치 하나로 매 epoch "
    "순전파하는 full-batch 방식을 사용했다(참고: 논문은 V100 GPU로 neighbor sampling "
    "기반 미니배치 학습을 사용했고 GIN 한 모델 학습에만 총 22,703초가 소요됐다 &mdash; "
    "본 구현은 그 정도 컴퓨팅 자원이 없어 규모를 조정했다).",
    styles["KBody"]))

story.append(Paragraph("5-2. 학습 중 발견한 문제와 조치", styles["KH2"]))
gnn_issues = [
    "<b>1차 시도(순수 degree 피처, pos_weight=7):</b> val PR-AUC가 20 epoch 동안 "
    "0.0007&rarr;0.0049 수준에 머물렀다. 전체 학습그래프의 실제 불균형(1:1,326)에 비해 "
    "pos_weight=7이 턱없이 부족해, 모델이 \"전부 음성\"에 가까운 퇴화해(degenerate) "
    "예측으로 수렴한 것으로 판단했다.",
    "<b>2차 시도(음성 다운샘플링 + 노드 피처 보강):</b> GBT에서 검증된 방식대로, 그래프 "
    "구조(메시지 패싱)는 전체 엣지를 그대로 쓰되 손실(loss) 계산 대상 엣지만 양성 전량 "
    "+ 음성 5만 건으로 다운샘플링했다. 또한 노드 입력 피처를 degree 2차원에서 금액·"
    "상대방 다양성을 포함한 6차원으로 늘렸다. 이 조치만으로 1 epoch 만에 val PR-AUC가 "
    "0.0355까지 뛰었으나(러닝레이트 0.01), 이후 진동하며 하락했다.",
    "<b>3차 시도(최종):</b> 러닝레이트를 0.003으로 낮추고 gradient clipping(1.0), "
    "ReduceLROnPlateau 스케줄러, patience 25 조기종료를 추가해 최종 결과를 얻었다.",
]
for l in gnn_issues:
    story.append(Paragraph("&bull; " + l, styles["KBullet"]))

story.append(Paragraph("5-3. 결과 &mdash; 시드 간 극심한 변동", styles["KH2"]))
gnn_res_table = [["시드", "F1", "Precision", "Recall", "PR-AUC", "P@500", "종료 방식"]]
end_reason = {1: "조기종료 (47ep)", 2: "120ep 만료 (계속 상승 중)", 3: "조기종료 (39ep)"}
for r in G["raw_results"]:
    seed = int(r["model"].replace("gnn_seed", ""))
    gnn_res_table.append([
        str(seed), f"{r['f1']*100:.1f}%", f"{r['precision']*100:.1f}%",
        f"{r['recall']*100:.1f}%", f"{r['pr_auc']:.4f}",
        f"{r['precision_at_k'].get('500', 0)*100:.1f}%", end_reason[seed],
    ])
gnn_res_table.append([
    "평균", f"{G['summary']['f1']['mean']*100:.1f}%", f"{G['summary']['precision']['mean']*100:.1f}%",
    f"{G['summary']['recall']['mean']*100:.1f}%", f"{G['summary']['pr_auc']['mean']:.4f}",
    f"{G['summary']['precision_at_500']['mean']*100:.1f}%", "-",
])
story.append(make_table(gnn_res_table, col_widths=[16 * mm, 20 * mm, 22 * mm, 20 * mm, 22 * mm, 20 * mm, 34 * mm]))
story.append(Spacer(1, 3 * mm))
story.append(Image(str(OUT / "chart_gnn_learning_curve.png"), width=150 * mm, height=94 * mm))
story.append(Paragraph(
    "시드 2는 epoch 45 부근부터 val PR-AUC가 꾸준히 상승해 120 epoch 종료 시점까지도 "
    "정체되지 않았다(최종 test PR-AUC 0.228, F1 23.6% &mdash; GBT의 F1 33~34%에 근접). "
    "반면 시드 1과 3은 epoch 15~20 부근의 낮은 국소 구간에서 벗어나지 못하고 각각 "
    "epoch 47, 39에서 조기종료됐다(test PR-AUC 0.082, 0.041 &mdash; 사실상 F1=0, "
    "threshold 0.5에서 양성을 하나도 예측하지 못함). 동일한 아키텍처·하이퍼파라미터로 "
    "난수 시드만 바꿨을 뿐인데 PR-AUC가 0.04~0.23까지 5배 이상 벌어졌다는 것 자체가 "
    "이번 GNN 구현의 가장 중요한 발견이다.",
    styles["KBody"]))

story.append(PageBreak())
story.append(Image(str(OUT / "chart_gnn_vs_gbt_comparison.png"), width=150 * mm, height=79 * mm))
story.append(Spacer(1, 3 * mm))
story.append(Image(str(OUT / "chart_seed_variance.png"), width=140 * mm, height=84 * mm))
story.append(Paragraph(
    "GBT(XGBoost/LightGBM)는 시드 간 PR-AUC가 0.27~0.31 범위로 조밀하게 모이는 반면, "
    "GNN은 0.04~0.23으로 훨씬 넓게 퍼져 있다 &mdash; 상자그림의 폭 차이가 이를 보여준다.",
    styles["KCaption"]))

story.append(Paragraph("5-4. 해석", styles["KH2"]))
story.append(Paragraph(
    "GNN이 GBT보다 근본적으로 약하다는 뜻은 아니다. 가장 좋은 시드(시드 2)는 F1 23.6%로 "
    "논문의 plain GIN(28.7%)에 근접했고, PR-AUC 0.228은 GBT(0.290)에 상당히 가깝다. "
    "문제는 <b>안정성</b>이다: 본 구현은 CPU 제약으로 (1) full-batch 학습(epoch당 "
    "gradient 업데이트 1회뿐 &mdash; 논문은 미니배치 neighbor sampling으로 epoch당 "
    "수백~수천 회 업데이트), (2) 3개 시드만 시도, (3) 120 epoch로 제한했다. 이 조건에서는 "
    "손실 곡면의 나쁜 국소 구간에 갇히는 시드가 쉽게 나타난다. 논문이 GPU와 미니배치로 "
    "훨씬 많은 gradient step을 확보한 것과 대조적이다. 즉 <b>GBT는 하이퍼파라미터/시드에 "
    "둔감하고 안정적으로 준수한 성능을 내는 반면, GNN은 동일한 컴퓨팅 예산 안에서는 "
    "\"운이 좋으면 GBT급, 운이 나쁘면 사실상 무용지물\"이라는 양극단을 오갔다</b>는 것이 "
    "이번 비교의 결론이다. 논문 자체도 plain GNN이 GFP+GBT보다 낮은 성능을 보고하지만"
    "(GIN 28.7% vs GFP+GBT 62~63%), 그 격차는 이만큼 시드에 따라 요동치지 않는다 "
    "&mdash; 이는 GPU 기반 미니배치 학습이 안정성에서도 이점을 준다는 방증으로 해석된다.",
    styles["KBody"]))

story.append(Paragraph("5-5. GNN 관련 한계 및 다음 단계", styles["KH2"]))
gnn_limits = [
    "<b>Neighbor sampling 미구현:</b> 논문처럼 미니배치 + neighbor sampling(1-hop 100개, "
    "2-hop 100개)으로 전환하면 epoch당 gradient step이 크게 늘어 시드 안정성이 개선될 "
    "가능성이 높다. 현재는 시간 제약으로 시도하지 못했다.",
    "<b>시드 3회:</b> 변동폭이 워낙 커서(F1 0~23.6%) 3개 시드로는 \"평균 성능\"이 큰 "
    "의미가 없다. 5~10개 시드로 분포 자체를 보고하는 것이 더 정직한 방법이다.",
    "<b>PNA 미구현:</b> 논문에서 가장 성능이 좋았던 GNN(HI-Small F1 56.8%)은 PNA인데, "
    "본 구현은 GIN+EU 계열만 시도했다.",
    "<b>하이브리드 미시도:</b> GBT 피처(pair_prior_count 등)를 GNN 노드/엣지 피처에 "
    "추가하거나, GNN 임베딩을 GBT 입력으로 결합하는 하이브리드는 시도하지 않았다.",
]
for l in gnn_limits:
    story.append(Paragraph("&bull; " + l, styles["KBullet"]))

story.append(PageBreak())

story.append(Spacer(1, 4 * mm))
story.append(Paragraph("6. 재현 방법", styles["KH1"]))
story.append(Paragraph(
    "C:\\Users\\aica_\\Documents\\CLaude\\AML_Baseline\\ 에 전체 코드가 있다.",
    styles["KBody"]))
story.append(Paragraph(
    "python train_baseline.py   # GBT: 전처리 + 피처 엔지니어링 + 3-seed 학습 -&gt; results.json<br/>"
    "python make_charts.py      # GBT results.json -&gt; 차트 PNG 3종<br/>"
    "python build_graph.py      # GNN: train/val/test 그래프 스냅샷(.pt) 생성<br/>"
    "python train_gnn.py        # GNN: 3-seed 학습 -&gt; results_gnn.json<br/>"
    "python make_charts_gnn.py  # GNN 비교 차트 PNG 3종<br/>"
    "python make_report.py      # 본 PDF 보고서 생성",
    styles["KCode"]))

story.append(Paragraph("7. 다음 단계 제언", styles["KH1"]))
next_steps = [
    "GNN을 neighbor-sampling 기반 미니배치 학습으로 전환해 시드 안정성 확보 (최우선)",
    "사이클/scatter-gather 등 다중 홉 그래프 패턴 피처를 GBT에 추가하여 논문의 GFP+GBT 성능"
    "(F1 63%대)과의 격차를 좁힐 수 있는지 검증",
    "LI-Small 등 다른 illicit ratio 데이터셋에서 GBT/GNN 재현성 확인",
    "GBT·GNN 각각 시드 5~10회로 확대하여 통계적 신뢰도 확보",
    "GNN 임베딩을 GBT 피처와 결합하는 하이브리드 모델 실험",
]
for s in next_steps:
    story.append(Paragraph("&bull; " + s, styles["KBullet"]))

doc = SimpleDocTemplate(
    str(OUT / "AML_HI-Small_Baseline_Report.pdf"),
    pagesize=A4,
    leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=18 * mm,
    title="IBM AML HI-Small 자금세탁 탐지 모델 - 베이스라인 보고서",
)
doc.build(story)
print("PDF written to", OUT / "AML_HI-Small_Baseline_Report.pdf")
