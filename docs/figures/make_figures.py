"""Render the report's figures as SVG, with no plotting dependency.

The numbers are copied from `user-sim eval` / `user-sim llm-eval` runs on ml-1m (see
docs/report.md). Rerun after changing them:

    uv run python docs/figures/make_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).parent
STYLE = """<style>
  text { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; fill: #556170; }
  .ink { fill: #141B24; } .title { fill: #141B24; font-size: 13px; font-weight: 600; }
  .grid { stroke: #DAE0E7; } .axis { stroke: #556170; } .ref { stroke: #556170; stroke-dasharray: 4 4; }
  .bg { fill: #FFFFFF; } .base { fill: #9BA6B3; } .llm { fill: #2346C0; } .leak { fill: #C27F0E; }
  .chance { fill: #E7EAEE; stroke: #DAE0E7; } .ci { stroke: #141B24; stroke-width: 1.6; opacity: .6; }
  .line-base { stroke: #9BA6B3; } .line-llm { stroke: #2346C0; } .hollow { fill: #FFFFFF; stroke: #2346C0; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    text { fill: #9AA6B4; } .ink, .title { fill: #E6EAF0; } .grid { stroke: #273140; }
    .axis, .ref { stroke: #9AA6B4; } .bg { fill: #151C25; } .base { fill: #5E6A78; } .llm { fill: #8EA5FF; }
    .leak { fill: #F0B13C; } .chance { fill: #222B36; stroke: #273140; } .ci { stroke: #E6EAF0; }
    .line-base { stroke: #5E6A78; } .line-llm { stroke: #8EA5FF; } .hollow { fill: #151C25; stroke: #8EA5FF; }
  }
</style>"""


def svg(width: int, height: int, body: list[str], label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{label}">{STYLE}<rect class="bg" width="{width}" height="{height}" rx="6"/>'
        + "".join(body)
        + "</svg>\n"
    )


def text(x, y, s, cls="", anchor="start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{s}</text>'


def bars(rows, x0, y0, width, max_v=0.6, ticks=(0, 0.2, 0.4, 0.6), chance=None, label_w=150, row_h=28, title=None):
    """Horizontal bars. rows: (label, value, kind, lo, hi). Returns (elements, bottom y)."""
    out = []
    if title:
        out.append(text(x0, y0, title, "title"))
        y0 += 16
    left, right = x0 + label_w, x0 + width - 56
    scale = lambda v: left + v / max_v * (right - left)
    top, bottom = y0, y0 + row_h * len(rows)
    for t in ticks:
        out.append(f'<line class="grid" x1="{scale(t):.1f}" x2="{scale(t):.1f}" y1="{top}" y2="{bottom}"/>')
        out.append(text(scale(t), bottom + 16, f"{t:.0%}", anchor="middle"))
    if chance is not None:
        out.append(f'<line class="ref" x1="{scale(chance):.1f}" x2="{scale(chance):.1f}" y1="{top - 4}" y2="{bottom + 2}"/>')
    for i, (name, v, kind, lo, hi) in enumerate(rows):
        y = top + i * row_h
        out.append(text(x0, y + row_h / 2 + 4, name, "ink"))
        out.append(f'<rect class="{kind}" x="{left}" y="{y + 6}" width="{scale(v) - left:.1f}" height="{row_h - 12}" rx="2"/>')
        if lo is not None and hi > lo:
            ym = y + row_h / 2
            out.append(f'<path class="ci" fill="none" d="M{scale(lo):.1f} {ym}H{scale(hi):.1f}M{scale(lo):.1f} {ym - 5}V{ym + 5}M{scale(hi):.1f} {ym - 5}V{ym + 5}"/>')
        out.append(text(x0 + width, y + row_h / 2 + 4, f"{v:.1%}", "ink", "end"))
    return out, bottom + 22


def fig_leak() -> str:
    before = [("random", 0.235, "chance"), ("popularity", 0.280, "base"), ("item-kNN", 0.287, "base"),
              ("newest film", 0.487, "leak"), ("LLM · last 20", 0.489, "llm")]
    after = [("random", 0.207, "chance"), ("popularity", 0.370, "base"), ("item-kNN", 0.410, "base"),
             ("newest film", 0.200, "leak"), ("LLM · last 20", 0.451, "llm")]
    a, h = bars([(*r, None, None) for r in before], 20, 28, 400, chance=0.2, label_w=118,
                title="Before: decoys were older, popular films")
    b, _ = bars([(*r, None, None) for r in after], 460, 28, 400, chance=0.2, label_w=118,
                title="After: decoys from the same release year")
    return svg(880, int(h) + 12, a + b, "hit@1 before and after matching decoys on release year")


def fig_results() -> str:
    rows = [
        ("random", 0.207, "chance", 0.160, 0.253), ("newest film", 0.200, "chance", None, None),
        ("popularity", 0.370, "base", 0.317, 0.423), ("item-kNN", 0.410, "base", 0.353, 0.467),
        ("LLM · last 20", 0.451, "llm", 0.400, 0.504), ("LLM · memo + tools", 0.474, "llm", 0.419, 0.526),
        ("LLM · memo", 0.487, "llm", 0.431, 0.541), ("LLM · full history", 0.513, "llm", 0.460, 0.567),
    ]
    body, h = bars(rows, 20, 24, 720, chance=0.2, label_w=160, row_h=30)
    body.append(text(20, h + 10, "hit@1 on 300 test decisions · whiskers: 95% interval (bootstrap over users) · dashed: chance"))
    return svg(760, int(h) + 24, body, "hit@1 with 95% intervals for every predictor")


def fig_history() -> str:
    W, H, top, bottom, y0, y1 = 880, 320, 44, 262, 0.30, 0.55
    y = lambda v: bottom - (v - y0) / (y1 - y0) * (bottom - top)
    panels = [
        (70, 410, "Item-kNN (validation)", "base", [("last 10", 0.389), ("last 20", 0.377), ("last 50", 0.372), ("last 100", 0.367)]),
        (510, 850, "LLM (test)", "llm", [("last 20", 0.451), ("all (median 163)", 0.513)]),
    ]
    out = []
    for x0, x1, title, kind, pts in panels:
        out.append(text(x0, 24, title, "title"))
        for v in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
            out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            out.append(text(x0 - 8, y(v) + 4, f"{v:.0%}", anchor="end"))
        step = (x1 - x0) / len(pts)
        xs = [x0 + step * (i + 0.5) for i in range(len(pts))]
        out.append(f'<polyline class="line-{kind}" fill="none" stroke-width="3" points="{" ".join(f"{x:.1f},{y(v):.1f}" for x, (_, v) in zip(xs, pts))}"/>')
        for x, (name, v) in zip(xs, pts):
            out.append(f'<circle class="{kind}" cx="{x:.1f}" cy="{y(v):.1f}" r="6"/>')
            out.append(text(x, y(v) - 13, f"{v:.1%}", "ink", "middle"))
            out.append(text(x, bottom + 20, name, anchor="middle"))
        out.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{bottom}" y2="{bottom}"/>')
        out.append(text(x0, H - 12, "history the predictor uses →"))
    return svg(W, H, out, "kNN gets worse with more history; the LLM gets better")


def fig_cost() -> str:
    W, H, L, R, T, B = 880, 400, 70, 850, 24, 340
    xa, xb, ya, yb = 200, 5000, 0.35, 0.54
    x = lambda v: L + (math.log(v) - math.log(xa)) / (math.log(xb) - math.log(xa)) * (R - L)
    y = lambda v: B - (v - ya) / (yb - ya) * (B - T)
    out = []
    for v in (0.36, 0.40, 0.44, 0.48, 0.52):
        out.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(text(L - 8, y(v) + 4, f"{v:.0%}", anchor="end"))
    for v in (250, 500, 1000, 2000, 4000):
        out.append(f'<line class="axis" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{B}" y2="{B + 5}"/>')
        out.append(text(x(v), B + 20, f"{v:,}", anchor="middle"))
    out.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}"/>')
    out.append(text((L + R) / 2, H - 12, "tokens read per decision (log scale)", anchor="middle"))
    for name, v in (("item-kNN 41.0% · no LLM", 0.410), ("popularity 37.0% · no LLM", 0.370)):
        out.append(f'<line class="ref" x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(text(R, y(v) - 7, name, anchor="end"))
    for a, b, v in ((351, 3448, 0.487), (617, 3714, 0.474)):
        out.append(f'<line class="line-llm" stroke-dasharray="2 4" x1="{x(a) + 8:.1f}" x2="{x(b) - 8:.1f}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
    points = [
        ("last 20", 419, 0.451, 12, 18, "start", False), ("memo", 351, 0.487, -12, 5, "end", False),
        ("memo + tools", 617, 0.474, 12, 18, "start", False), ("full history", 3002, 0.513, -12, -10, "end", False),
        ("memo, incl. writing it", 3448, 0.487, -12, -10, "end", True),
        ("tools, incl. writing", 3714, 0.474, -12, 24, "end", True),
    ]
    for name, v, h, dx, dy, anchor, hollow in points:
        out.append(f'<circle class="{"hollow" if hollow else "llm"}" cx="{x(v):.1f}" cy="{y(h):.1f}" r="7"/>')
        out.append(text(x(v) + dx, y(h) + dy, f"{name} · {h:.1%}", "ink", anchor))
    return svg(W, H, out, "hit@1 against tokens per decision for the four LLM contexts")


def fig_tools() -> str:
    groups = [
        ("Agent 1 (~9 lookups)", 0.500, 0.489), ("Agent 2 (~237, popularity)", 0.531, 0.483),
        ("Agent 3 (~63, recent ratings)", 0.511, 0.517), ("Agents 4–5 (~27)", 0.447, 0.440),
    ]
    rows = []
    for name, memo, tools in groups:
        rows += [(name, memo, "base", None, None), ("", tools, "llm", None, None)]
    body, h = bars(rows, 20, 24, 720, label_w=220, row_h=24)
    body.append(text(20, h + 10, "hit@1 on the same decisions · grey: memo only · blue: memo + tools"))
    return svg(760, int(h) + 24, body, "memo vs memo plus tools by answering agent")


if __name__ == "__main__":
    for name, fig in [("leak", fig_leak), ("results", fig_results), ("history", fig_history),
                      ("cost", fig_cost), ("tools", fig_tools)]:
        (OUT / f"{name}.svg").write_text(fig())
        print(f"wrote {OUT / name}.svg")
