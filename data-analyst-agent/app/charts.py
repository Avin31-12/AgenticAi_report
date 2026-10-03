"""
charts.py
---------
Builds PNG charts for the PDF from the analysis result.

Uses matplotlib's Figure object directly (not pyplot), so it is safe to call
from several web requests at the same time. The old tools.py used global
pyplot state, which is not thread-safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .analysis import fmt_num

FIG_SIZE = (8, 4.2)
RATIO = FIG_SIZE[1] / FIG_SIZE[0]
INK, TEAL, AMBER = "#12263A", "#0F8B8D", "#D98E04"


def _new(title: str):
    fig = Figure(figsize=FIG_SIZE)
    ax = fig.subplots()
    ax.set_title(title, fontsize=11, loc="left", color=INK, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return str(path)


def _short(label: str, n: int = 26) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def build_charts(result: dict, out_dir) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: list[dict] = []

    t = result.get("trend")
    if t:
        fig, ax = _new(t["title"])
        labels = [p["period"] for p in t["series"]]
        vals = [p["value"] for p in t["series"]]
        ax.plot(range(len(vals)), vals, color=TEAL, linewidth=2, marker="o", markersize=3)
        ax.fill_between(range(len(vals)), vals, alpha=0.08, color=TEAL)
        idx = np.linspace(0, len(labels) - 1, min(len(labels), 8)).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax.grid(axis="y", alpha=0.25)
        charts.append({"title": t["title"], "path": _save(fig, out_dir / "trend.png")})

    for i, b in enumerate(result.get("breakdowns", [])[:2], start=1):
        fig, ax = _new(b["title"])
        items = b["items"][:8][::-1]
        ax.barh([_short(x["label"]) for x in items], [x["value"] for x in items], color=TEAL)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        charts.append({"title": b["title"], "path": _save(fig, out_dir / f"breakdown_{i}.png")})

    h = result.get("histogram")
    if h:
        fig, ax = _new(h["title"])
        counts = [b["count"] for b in h["bins"]]
        ax.bar(range(len(counts)), counts, color=AMBER, width=0.85)
        step = max(1, len(counts) // 6)
        ax.set_xticks(range(0, len(counts), step))
        ax.set_xticklabels([h["bins"][i]["label"] for i in range(0, len(counts), step)],
                           rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("Rows", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        charts.append({"title": h["title"], "path": _save(fig, out_dir / "distribution.png")})

    s = result.get("scatter")
    if s:
        fig, ax = _new(f"{s['title']} (r = {s['r']:.2f})")
        ax.scatter([p["x"] for p in s["points"]], [p["y"] for p in s["points"]],
                   s=12, alpha=0.6, color=INK)
        ax.set_xlabel(s["x"], fontsize=8)
        ax.set_ylabel(s["y"], fontsize=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax.grid(alpha=0.2)
        charts.append({"title": s["title"], "path": _save(fig, out_dir / "relationship.png")})

    return charts
