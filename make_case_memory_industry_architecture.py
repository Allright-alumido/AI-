"""
make_case_memory_industry_architecture.py

Diagrams today's addition (the Case Memory reflection layer inside the
Speed Layer's Decision step) as TWO parallel tracks:
  1. What was actually built here (a single-process simulation).
  2. What the equivalent COMPONENT would be in an industry-grade real-time
     streaming deployment -- named explicitly, not implied.

This is meant to sit alongside lambda_architecture.png in the proposal:
that diagram covers Batch vs. Speed layer; this one zooms into the Speed
layer's Decision step specifically, since that's where today's work
(memory_store.CaseMemory) lives, and makes the "simulation -> real
infrastructure" mapping explicit rather than leaving it as a footnote.

Honesty is the whole point of this diagram: every industry-track box is a
NAME of a class of system this prototype's own component stands in for,
not a claim that this project runs on it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 10.5))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 12.5)
ax.axis("off")

NAVY = "#1f3a5f"
BLUE = "#2f6690"
LIGHT = "#d9e6f2"
GOLD = "#c9922a"
GREY = "#6b7280"
GREEN = "#2e7d4f"
LIGHTGREEN = "#e3f0e8"
PURPLE = "#5b3a8e"
LIGHTPURPLE = "#ece4f5"


def box(x, y, w, h, text, fc=LIGHT, ec=NAVY, fontsize=9, fontcolor="#10243e",
        weight="bold", dashed=False):
    style = "round,pad=0.28,rounding_size=0.15"
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.6,
                        edgecolor=ec, facecolor=fc,
                        linestyle=(0, (4, 3)) if dashed else "solid")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, weight=weight, linespacing=1.3)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                         linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


ax.text(7.25, 12.15, "Case Memory Layer -- Prototype vs. Industry-Grade Real-Time Architecture",
        ha="center", fontsize=14.5, weight="bold", color=NAVY)
ax.text(7.25, 11.72, "Zooming into the Speed Layer's Decision step (see lambda_architecture.png for the full Batch/Speed picture)",
        ha="center", fontsize=9.8, color=GREY, style="italic")

# ------------------------------------------------------------------ #
# Row labels
# ------------------------------------------------------------------ #
box(0.3, 10.55, 13.9, 0.55, "THIS PROTOTYPE  (single Python process, in-memory, today's actual code)",
    fc=NAVY, fontcolor="white", fontsize=10.5)
box(0.3, 0.55, 13.9, 0.55, "INDUSTRY-GRADE EQUIVALENT  (named component class, NOT what this project runs on)",
    fc=PURPLE, fontcolor="white", fontsize=10.5)

# ------------------------------------------------------------------ #
# 5 pipeline stages, top (prototype) row
# ------------------------------------------------------------------ #
col_w = 2.55
gap = 0.28
x0 = 0.45
y_top = 8.55
h_top = 1.75

stages = [
    ("(1) INGEST", "BlockBootstrapGenerator\n\nseasonal block-bootstrap\nof real history -> 1 new\nsettlement row / pair / tick", LIGHT),
    ("(2) STATE", "streaming_pipeline.PairState\n\nin-memory dict: rolling\nsales, inventory, carried\nexogenous context", LIGHT),
    ("(3) DECIDE", "inference.Recommender\n.recommend_for_row()\n\nfrozen LightGBM + PPO,\nin-process function call", "#eaf2fb"),
    ("(4) REFLECT", "memory_store.CaseMemory\n\nlinear-scan k-NN\n(np.linalg.norm loop)\nover a Python list\n\nclosed loop: logs this\ntick's action+reward back", LIGHTPURPLE),
    ("(5) SERVE", "write_snapshot()\n\nlatest_snapshot.json +\nstream_log.csv, polled by\nStreamlit autorefresh", LIGHTGREEN),
]

xs = []
for i, (title, body, fc) in enumerate(stages):
    x = x0 + i * (col_w + gap)
    xs.append(x)
    fontsize = 7.3 if i == 3 else 8.0
    box(x, y_top, col_w, h_top, f"{title}\n\n{body}", fc=fc, fontsize=fontsize)
    if i < len(stages) - 1:
        arrow(x + col_w, y_top + h_top / 2, x + col_w + gap, y_top + h_top / 2, color=NAVY)

# ------------------------------------------------------------------ #
# Industry-equivalent row (bottom), same x columns
# ------------------------------------------------------------------ #
y_bot = 1.35
h_bot = 1.75

industry = [
    ("(1) INGEST", "Kafka / Kinesis /\nPub-Sub\n\nreal event stream from\nPOS, ERP, or WMS", "#f3ede0"),
    ("(2) STATE", "Flink / Spark\nStructured Streaming\n\nstateful operators +\nRedis / RocksDB backend\n(fault-tolerant, distributed)", "#f3ede0"),
    ("(3) DECIDE", "Model Serving\n(KServe / Seldon /\nTorchServe / BentoML)\n\nbehind a Feature Store\n(e.g. Feast) for train/serve parity", "#f3ede0"),
    ("(4) REFLECT", "Vector Database /\nANN Index\n(FAISS, Milvus, pgvector)\n\nsub-linear similarity search\nat production scale", "#f3ede0"),
    ("(5) SERVE", "Push-based dashboard\n\nKafka topic -> WebSocket/\nSSE -> Grafana or a live\ndashboard (no polling)", "#f3ede0"),
]

for i, (title, body, fc) in enumerate(industry):
    x = xs[i]
    box(x, y_bot, col_w, h_bot, f"{title}\n\n{body}", fc=fc, ec=PURPLE, fontcolor="#4a2f70", fontsize=7.8, dashed=True)
    if i < len(industry) - 1:
        arrow(x + col_w, y_bot + h_bot / 2, x + col_w + gap, y_bot + h_bot / 2, color=PURPLE)

# vertical dashed "maps to" arrows between the two rows
for x in xs:
    arrow(x + col_w / 2, y_top, x + col_w / 2, y_bot + h_bot,
          color=GOLD, style="-|>", lw=1.3)
ax.text(7.25, (y_top + y_bot + h_bot) / 2 + 0.15, "maps to  (component-for-component, not deployed as such)",
        ha="center", fontsize=8.6, color="#6b4d10", style="italic")

# ------------------------------------------------------------------ #
# Honesty / scope note
# ------------------------------------------------------------------ #
box(0.3, 3.55, 13.9, 2.15,
    "What today's Case Memory layer actually is: a bounded, evidence-gated case-based memory sitting on top of the FROZEN\n"
    "PPO policy -- it logs (state, executed action, reward), retrieves the k most similar past states for the SAME\n"
    "store/product pair, and proposes a nudge (capped at +/-1 order step) only when 3+ similar cases consistently show a\n"
    "different action did better. It never touches the PPO network's weights -- this is retrieval-augmented decision\n"
    "support with online feedback, NOT online reinforcement learning.\n\n"
    "What's genuinely analogous to industry practice: the two-track separation itself (compute vs. state vs. model\n"
    "serving vs. similarity search are already separated concerns in this code, even though they run in one process\n"
    "here), and the k-NN-over-feature-vectors design -- which IS exactly what a vector database does, just without the\n"
    "ANN index needed to make it sub-linear at scale. What's different: single process vs. distributed, in-memory\n"
    "Python list vs. a real datastore, and JSON/CSV polling vs. an event-driven push.",
    fc="#fbf7ee", ec=GOLD, fontcolor="#4a3a1a", fontsize=8.3, weight="normal")

plt.tight_layout()
fig.savefig("case_memory_industry_architecture.png", dpi=180, bbox_inches="tight")
print("Saved case_memory_industry_architecture.png")
