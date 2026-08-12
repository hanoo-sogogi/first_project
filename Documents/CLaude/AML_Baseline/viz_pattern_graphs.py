# -*- coding: utf-8 -*-
"""
Visualize real HI-Small laundering-pattern subgraphs (accounts=nodes,
transactions=edges) extracted directly from HI-Small_Patterns.txt.

This is the concrete, ground-truth version of the structures a GNN's message
passing needs to recognize -- each subplot is ONE real labeled laundering
attempt, not a synthetic mockup.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PATTERNS_PATH = Path(r"C:\Users\aica_\Documents\CLaude\HI-Small_Patterns.txt")
OUT_PATH = Path(r"C:\Users\aica_\Documents\CLaude\AML_Baseline\chart_pattern_graphs.png")

EDGE_RED = "#e34948"
NODE_BLUE = "#2a78d6"
NODE_HUB = "#eb6834"

# ---------------------------------------------------------------------------
# 1. Parse Patterns.txt into individual attempts (grouped, not pooled)
# ---------------------------------------------------------------------------
attempts = []
current = None
with open(PATTERNS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)", line)
            current = {"pattern": m.group(1) if m else "UNKNOWN", "edges": []}
            continue
        if line.startswith("END LAUNDERING ATTEMPT"):
            if current is not None:
                attempts.append(current)
            current = None
            continue
        if current is not None and "," in line:
            parts = line.split(",")
            if len(parts) == 11:
                ts, fb, fa, tb, ta, amt_r, cur_r, amt_p, cur_p, fmt, lbl = parts
                current["edges"].append((f"{fb}_{fa}", f"{tb}_{ta}"))

print(f"Parsed {len(attempts)} attempts")

# ---------------------------------------------------------------------------
# 2. Pick one representative (median-sized) attempt per pattern type
# ---------------------------------------------------------------------------
PATTERN_ORDER = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER",
                  "CYCLE", "RANDOM", "BIPARTITE", "STACK"]

chosen = {}
for p in PATTERN_ORDER:
    cands = [a for a in attempts if a["pattern"] == p]
    cands_sorted = sorted(cands, key=lambda a: len(a["edges"]))
    if p == "BIPARTITE":
        # every BIPARTITE instance in this dataset is a disjoint union of
        # 1:1 pairs (n_nodes == 2 * n_edges, verified across all 49
        # instances) - pick a small-medium one so the "batch of one-time
        # transfers" structure is legible rather than trying to find a
        # converging fan shape that does not exist in the real data.
        chosen[p] = [a for a in cands_sorted if len(a["edges"]) == 6][0]
    elif p == "STACK":
        # pick the one instance that deviates from the pure disjoint-chain
        # ratio (nodes != 1.5 * edges), i.e. has actual node-sharing in the
        # middle layer, so the "extra bipartite layer" structure is visible.
        chosen[p] = [a for a in cands_sorted if len(a["edges"]) == 10
                      and len(set(sum(a["edges"], ()))) == 16][0]
    else:
        chosen[p] = cands_sorted[len(cands_sorted) // 2]
    print(f"{p}: {len(cands)} instances available, chose one with "
          f"{len(chosen[p]['edges'])} edges / "
          f"{len(set(sum(chosen[p]['edges'], ())))} nodes")

# ---------------------------------------------------------------------------
# 3. Draw
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
axes = axes.flatten()

for ax, pname in zip(axes, PATTERN_ORDER):
    a = chosen[pname]
    G = nx.MultiDiGraph()
    G.add_edges_from(a["edges"])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # relabel nodes to short synthetic labels for legibility
    mapping = {n: f"A{i+1}" for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # layout choice per pattern shape
    if pname == "CYCLE":
        pos = nx.circular_layout(G)
    elif pname in ("FAN-OUT", "FAN-IN"):
        # put the hub (max total degree) in the center
        deg = dict(G.degree())
        hub = max(deg, key=deg.get)
        others = [nnode for nnode in G.nodes() if nnode != hub]
        pos = nx.shell_layout(G, nlist=[[hub], others])
    elif pname in ("BIPARTITE", "STACK"):
        # layered layout: layer = shortest hop-distance from any source
        # (in-degree 0) node, via BFS on the simple (non-multi) digraph
        DG = nx.DiGraph(G)
        sources = [nnode for nnode in DG.nodes() if DG.in_degree(nnode) == 0]
        depth = {}
        for s in sources:
            for nnode, d in nx.single_source_shortest_path_length(DG, s).items():
                depth[nnode] = min(depth.get(nnode, d), d)
        for nnode in DG.nodes():
            depth.setdefault(nnode, 0)
        nx.set_node_attributes(G, depth, "layer")
        pos = nx.multipartite_layout(G, subset_key="layer")
    else:
        pos = nx.spring_layout(G, seed=7, k=1.1 / max(n_nodes, 1) ** 0.4)

    deg = dict(G.degree())
    node_sizes = [260 + 90 * deg[nnode] for nnode in G.nodes()]
    max_deg_node = max(deg, key=deg.get)
    node_colors = [NODE_HUB if nnode == max_deg_node and deg[nnode] >= 3 else NODE_BLUE
                   for nnode in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_RED, width=1.4,
                            arrowsize=11, alpha=0.85, connectionstyle="arc3,rad=0.08",
                            node_size=node_sizes)
    if n_nodes <= 14:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color="white",
                                 font_weight="bold")

    ax.set_title(f"{pname}\n(노드 {n_nodes}개, 거래 {n_edges}건)", fontsize=11.5)
    ax.axis("off")
    if pname == "BIPARTITE":
        ax.text(0.5, -0.08, "* 계좌 공유 없는 완전 분리 1:1 쌍\n  (49개 사례 전부 동일 구조)",
                transform=ax.transAxes, ha="center", fontsize=8, color="#c0392b")

hub_patch = mpatches.Patch(color=NODE_HUB, label="허브 계좌 (구조 내 최다 연결)")
node_patch = mpatches.Patch(color=NODE_BLUE, label="일반 계좌 (노드)")
edge_patch = mpatches.Patch(color=EDGE_RED, label="이상거래 (엣지)")
fig.legend(handles=[node_patch, hub_patch, edge_patch], loc="lower center",
           ncol=3, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("HI-Small 실제 라벨링 데이터로 본 8가지 자금세탁 패턴의 그래프 구조\n"
             "(각 패턴 유형에서 대표 사례 1건씩, Patterns.txt 원본 거래로 그림)",
             fontsize=14, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print("saved:", OUT_PATH)
