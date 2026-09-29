"""Streamlit dashboard for the claim 2.2 entity-chain analysis.
    uv run streamlit run analysis/claim2_2/dashboard.py

A background job (Task 7) is still filling in `out/<article_id>/` for most of
the 140 coded articles, and even a finished article can be missing one of the
three per-view files if the job wrote them out of order. Every read from
`out/` therefore goes through `_read_csv`/`_read_json`, which return `None`
instead of raising, and every section that depends on one of those tables
checks for `None` and shows an explanatory line instead of a traceback (A8.5).
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.claim2_2 import common, entities, figures, graph  # noqa: E402
from analysis.claim2_2.metrics import SUFFICIENT_N, z_against_null  # noqa: E402

# Statement labels and colours come from claim 1 so the two analyses read the same: the claim 1
# script imports `boilerplate` flat, hence the extra path entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claim1"))
from analysis.claim1.per_article_subcategory_stacked_bars_by_comment_count import (  # noqa: E402
    _TAG_CLEANUP, FINEGRAIN_PLOT_ORDER, SUBCAT_TO_GROUP)

# The network charts draw every link as a sampled curve, which passes Altair's 5,000-row default.
alt.data_transformers.disable_max_rows()

CAT_COLOR = {"VO": "#3b6fb6", "PK": "#c67c2b", "INF": "#579c5a", "redundant": "#9aa0a6"}
SUBCAT_COLOR = dict(FINEGRAIN_PLOT_ORDER)
GROUPS = ("Visual Observation", "Inference", "Prior Knowledge")
NOT_CODED = "not coded"  # tags claim 1 drops: Meta / Paratext, unprompted, instruction_repeat, ...
NOT_CODED_COLOR = "#c4c7cc"
# Entity network: VO | INF | PK columns (inference in the middle, where observation and prior
# knowledge meet), each coloured by a mid swatch of its claim 1 ramp.
NET_COLS = [("VO", "Visual observation"), ("INF", "Inference"), ("PK", "Prior knowledge")]
NET_COLOR = {"VO": SUBCAT_COLOR["Discrete data observations"],
             "INF": SUBCAT_COLOR["Hypothetical arguments"],
             "PK": SUBCAT_COLOR["Background knowledge"]}
VIEWS = ("all", "minus_echo", "minus_boilerplate")
VIEW_CAPTION = (
    "Readers were asked how the chart relates to them and their community and to invent a "
    "headline, so some entity mentions are the survey talking rather than the reader. "
    "`minus_echo` drops mentions of entity groups flagged as echoing the prompt; "
    "`minus_boilerplate` drops boilerplate sentences outright."
)
PER_VIEW_TABLES = ("signatures", "lengths", "components_dist", "bridging",
                    "sparsity", "paths_to_insight", "null_chains", "comment_graphs", "null_graphs")
BASE_TABLES = ("mentions", "edges", "chains", "components")
# Article-level statement-graph measures (graph.graph_summary), in display order.
GRAPH_METRICS = [
    ("mean_density", "density", "share of possible statement pairs that share an entity (mean over comments)"),
    ("isolated_rate", "isolated statements", "share of statements linked to nothing"),
    ("mean_cycle_rank", "cycle rank", "edges − nodes + components: 0 for a chain or tree, higher when points are tied back together"),
    ("share_cyclic", "comments with a cycle", "share of comments whose graph is not a tree"),
    ("mean_max_clique", "largest clique", "size of the biggest group of statements all about a common entity (mean over comments)"),
    ("articulation_rate", "bridge statements", "share of statements whose removal splits the comment's graph"),
    ("articulation_rate_VO", "bridge rate · VO", "share of VO statements that are bridges"),
    ("articulation_rate_INF", "bridge rate · INF", "share of INF statements that are bridges"),
    ("articulation_rate_PK", "bridge rate · PK", "share of PK statements that are bridges"),
    ("mean_degree_VO", "degree · VO", "mean number of statements a VO statement links to"),
    ("mean_degree_INF", "degree · INF", "mean number of statements an INF statement links to"),
    ("mean_degree_PK", "degree · PK", "mean number of statements a PK statement links to"),
    ("cross_edge_share", "cross-category links", "share of links (among VO/PK/INF) that join different categories"),
]
HEADLINE_FIELDS = [
    ("n_chains", "n chains", "{:.0f}"),
    ("mean_chain_length", "mean chain length", "{:.3f}"),
    ("share_len1", "share length 1", "{:.1%}"),
    ("isolated_sentence_rate", "isolated sentence rate", "{:.1%}"),
    ("largest_component_share", "largest component share", "{:.1%}"),
]


def _read_csv(out: Path, name: str) -> pd.DataFrame | None:
    p = out / f"{name}.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data
def load(article_id: str, view: str):
    out = common.out_dir(article_id)
    tables = {n: _read_csv(out, n) for n in BASE_TABLES}
    suffix = "" if view == "all" else f"__{view}"
    for n in PER_VIEW_TABLES:
        tables[n] = _read_csv(out, f"{n}{suffix}")
    sents = common.sentences(article_id)
    sents["subcat"] = [_TAG_CLEANUP.get(t, t) if isinstance(t, str) else "unknown" for t in sents["tag"]]
    sents["group"] = sents["subcat"].map(SUBCAT_TO_GROUP).fillna(NOT_CODED)
    sents["color"] = sents["subcat"].map(SUBCAT_COLOR).fillna(NOT_CODED_COLOR)
    tables["sentences"] = sents
    tables["groups"] = pd.DataFrame(entities.load(article_id))
    tables["summary"] = _read_json(out / "summary.json")
    return tables


@st.cache_data
def load_cross_article():
    return {
        "by_variant": _read_csv(common.OUT_DIR, "cross_article_by_variant"),
        "summary": _read_csv(common.OUT_DIR, "cross_article_summary"),
    }


TYPE_COLOR = {"entity": "#fde7b0", "data_variable": "#cfe3f7", "value": "#d9efd3", "event": "#f3d4e6"}
PAGE_SIZE = 10
TOKEN_CSS = """<style>
.stmt{margin:0 0 .55rem 0;line-height:1.9}
.stmt.off{opacity:.45}
.sid{font-family:monospace;font-size:.8rem;color:#6b7280;margin-right:.35rem}
.cat{display:inline-block;padding:0 .45rem;border-radius:.6rem;color:#fff;font-size:.72rem;
     font-weight:600;margin-right:.3rem;line-height:1.5}
.grp{font-size:.72rem;color:#6b7280;margin-right:.3rem;text-transform:uppercase;letter-spacing:.03em}
.legend{line-height:2;margin-bottom:.4rem}
.legend .grp{display:inline-block;min-width:8.5rem}
.tok{border-radius:.25rem;padding:0 .15rem;color:#111}
.tok sub{font-size:.62rem;color:#374151;margin-left:.1rem}
.tok.echo{background:transparent!important;border-bottom:2px dotted #9aa0a6;color:inherit}
.tok.echo sub{color:#9aa0a6}
</style>"""


def token_html(text: str, rows: list, entity_meta: dict, dim_ids: set[str]) -> str:
    """Sentence text with each dictionary token wrapped and labelled with its entity id."""
    out, last = [], 0
    for r in sorted(rows, key=lambda r: (int(r.span_start), -int(r.span_end))):
        a, b = int(r.span_start), int(r.span_end)
        if a < last:  # overlapping match; the earlier, longer one already covers it
            continue
        canon, etype = entity_meta.get(r.entity_id, (r.entity_id, r.type))
        cls = "tok echo" if r.entity_id in dim_ids else "tok"
        out.append(html.escape(text[last:a]))
        out.append(f'<span class="{cls}" style="background:{TYPE_COLOR.get(etype, "#eee")}" '
                   f'title="{html.escape(str(canon))} ({etype})">{html.escape(text[a:b])}'
                   f'<sub>{html.escape(str(r.entity_id))}</sub></span>')
        last = b
    out.append(html.escape(text[last:]))
    return "".join(out)


def _ink(hex_color: str) -> str:
    """Dark text on claim 1's light swatches, white on its dark ones."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#111" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#fff"


def stmt_badge(subcat: str, group: str) -> str:
    """Claim 1 subcategory pill, preceded by its coarse group."""
    color = SUBCAT_COLOR.get(subcat, NOT_CODED_COLOR)
    return (f'<span class="grp">{html.escape(group)}</span>'
            f'<span class="cat" style="background:{color};color:{_ink(color)}">{html.escape(subcat)}</span>')


def legend_html() -> str:
    parts = []
    for grp in GROUPS:
        subs = [n for n, _ in FINEGRAIN_PLOT_ORDER if SUBCAT_TO_GROUP[n] == grp]
        parts.append(f'<span class="grp">{grp}</span>' + "".join(
            f'<span class="cat" style="background:{SUBCAT_COLOR[n]};color:{_ink(SUBCAT_COLOR[n])}">{n}</span>'
            for n in subs))
    parts.append(f'<span class="grp">{NOT_CODED}</span><span class="cat" style="background:{NOT_CODED_COLOR};'
                 f'color:#111">Meta / Paratext, unprompted, …</span>')
    return '<div class="legend">' + "<br>".join(parts) + "</div>"


def view_edges(edges: pd.DataFrame, sents: pd.DataFrame, view: str, echo_ids: set[str]) -> pd.DataFrame:
    """Apply the selected view to the committed (all-view) edge table, the same way the metrics
    job does: minus_echo drops echo entities from each link, minus_boilerplate drops links that
    touch a boilerplate sentence."""
    e = edges.copy()
    e["shared"] = e["shared_entities"].fillna("").astype(str).str.split("|")
    if view == "minus_echo":
        e["shared"] = e["shared"].map(lambda xs: [x for x in xs if x and x not in echo_ids])
        e = e[e["shared"].map(len) > 0]
    elif view == "minus_boilerplate":
        bp = set(zip(sents.loc[sents["boilerplate"].astype(bool), "comment_index"],
                     sents.loc[sents["boilerplate"].astype(bool), "sent_id"]))
        keep = [(c, a) not in bp and (c, b) not in bp
                for c, a, b in zip(e["comment_index"], e["sent_a"], e["sent_b"])]
        e = e[keep]
    return e


def bridge_statements(sents: pd.DataFrame, edges: pd.DataFrame) -> set[int]:
    """Articulation points of one comment's statement graph (same rule as graph.py)."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(int(x) for x in sents["sent_id"])
    g.add_edges_from((int(a), int(b)) for a, b in zip(edges["sent_a"], edges["sent_b"]))
    return set(nx.articulation_points(g)) if g.number_of_edges() else set()


def arc_graph(sents: pd.DataFrame, edges: pd.DataFrame, bridges: frozenset = frozenset()) -> alt.Chart:
    """Statements on a line in reading order; a semicircle joins every pair sharing an entity.
    Bridge statements (articulation points) get a dark ring."""
    import math
    pos = {int(s): i for i, s in enumerate(sents["sent_id"])}
    nodes = pd.DataFrame({
        "x": range(len(sents)), "y": 0.0,
        "label": [f"S{s}" for s in sents["sent_id"]],
        "subcategory": sents["subcat"].values, "group": sents["group"].values,
        "color": sents["color"].values,
        "bridge": [int(x) in bridges for x in sents["sent_id"]],
        "text": [t if len(t) < 90 else t[:87] + "…" for t in sents["text"].astype(str)]})
    pts = []
    for k, r in enumerate(edges.itertuples()):
        a, b = sorted((pos[int(r.sent_a)], pos[int(r.sent_b)]))
        mid, rad = (a + b) / 2, (b - a) / 2
        link = f"S{r.sent_a} – S{r.sent_b}"
        for j in range(21):
            t = math.pi * j / 20
            pts.append({"edge": k, "x": mid - rad * math.cos(t), "y": rad * math.sin(t),
                        "order": j, "link": link, "shared": ", ".join(r.shared), "n": len(r.shared)})
    top = max([p["y"] for p in pts], default=0.5) + 0.35
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.6, max(len(sents) - 0.4, 0.6)]))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.45, top]))
    layers = []
    if pts:
        layers.append(alt.Chart(pd.DataFrame(pts)).mark_line(color="#6b7280", opacity=0.55).encode(
            x=xs, y=ys, detail="edge:N", order="order:Q",
            strokeWidth=alt.StrokeWidth("n:Q", scale=alt.Scale(range=[1.2, 4]), legend=None),
            tooltip=["link:N", alt.Tooltip("shared:N", title="shared entities")]))
    base = alt.Chart(nodes)
    layers.append(base.mark_circle(size=260, opacity=1).encode(
        x=xs, y=ys,
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["label:N", "group:N", "subcategory:N", "text:N"]))
    if bridges:
        layers.append(alt.Chart(nodes[nodes["bridge"]]).mark_circle(
            size=560, filled=False, stroke="#111827", strokeWidth=2).encode(x=xs, y=ys))
    layers.append(base.mark_text(dy=18, fontSize=11).encode(x=xs, y=ys, text="label:N"))
    return alt.layer(*layers).properties(height=max(120, min(320, int(70 + 40 * top))))


def network_graph(sents: pd.DataFrame, edges: pd.DataFrame, bridges: frozenset = frozenset()) -> alt.Chart:
    """The comment's statement graph as a node-link drawing: a force layout (fixed seed, so it
    does not move between reruns) pulls linked statements together, so cliques, cycles and
    bridges are visible. Isolated statements sit in a row along the bottom."""
    import networkx as nx
    ids = [int(x) for x in sents["sent_id"]]
    g = nx.Graph()
    g.add_nodes_from(ids)
    for r in edges.itertuples():
        g.add_edge(int(r.sent_a), int(r.sent_b), n=len(r.shared), shared=", ".join(r.shared))
    linked = [n for n in ids if g.degree(n) > 0]
    isolated = [n for n in ids if g.degree(n) == 0]
    # Lay out each connected component on its own (a joint layout pushes components far apart
    # and collapses small ones to a point), scale it by its size, and set them side by side.
    pos, x_off = {}, 0.0
    comps = sorted(nx.connected_components(g.subgraph(linked)), key=lambda c: (-len(c), min(c)))
    for comp in comps:
        sub = g.subgraph(comp)
        raw = (nx.spring_layout(sub, seed=0, iterations=200) if len(comp) > 2
               else {n: (float(i), 0.0) for i, n in enumerate(sorted(comp))})
        xs0, ys0 = [v[0] for v in raw.values()], [v[1] for v in raw.values()]
        span = max(max(xs0) - min(xs0), max(ys0) - min(ys0)) or 1.0
        size = 0.6 + 0.35 * len(comp) ** 0.5
        cx, cy = (max(xs0) + min(xs0)) / 2, (max(ys0) + min(ys0)) / 2
        for n, (x, y) in raw.items():
            pos[n] = (x_off + size / 2 + (x - cx) / span * size, (y - cy) / span * size)
        x_off += size + 0.5
    width = max(x_off - 0.5, 1.0)
    bottom = min((y for _, y in pos.values()), default=0) - 0.7
    for i, n in enumerate(isolated):
        pos[n] = (width * (i + 0.5) / len(isolated), bottom)
    info = sents.set_index(sents["sent_id"].astype(int))
    nodes = pd.DataFrame({
        "x": [pos[n][0] for n in ids], "y": [pos[n][1] for n in ids],
        "label": [f"S{n}" for n in ids], "group": [info.at[n, "group"] for n in ids],
        "subcategory": [info.at[n, "subcat"] for n in ids], "color": [info.at[n, "color"] for n in ids],
        "bridge": [n in bridges for n in ids],
        "text": [(t if len(t) < 90 else t[:87] + "…") for t in (str(info.at[n, "text"]) for n in ids)]})
    lines = []
    for k, (a, b, d) in enumerate(g.edges(data=True)):
        for n in (a, b):
            lines.append({"edge": k, "x": pos[n][0], "y": pos[n][1], "n": d["n"],
                          "link": f"S{a} – S{b}", "shared": d["shared"]})
    xs_ = [p[0] for p in pos.values()] or [0]
    ys_ = [p[1] for p in pos.values()] or [0]
    pad = 0.25
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[min(xs_) - pad, max(xs_) + pad]))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[min(ys_) - pad, max(ys_) + pad]))
    layers = []
    if lines:
        layers.append(alt.Chart(pd.DataFrame(lines)).mark_line(color="#6b7280", opacity=0.55).encode(
            x=xs, y=ys, detail="edge:N",
            strokeWidth=alt.StrokeWidth("n:Q", scale=alt.Scale(range=[1.2, 4]), legend=None),
            tooltip=["link:N", alt.Tooltip("shared:N", title="shared entities")]))
    base = alt.Chart(nodes)
    layers.append(base.mark_circle(size=300, opacity=1).encode(
        x=xs, y=ys, color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["label:N", "group:N", "subcategory:N", "text:N"]))
    if bridges:
        layers.append(alt.Chart(nodes[nodes["bridge"]]).mark_circle(
            size=620, filled=False, stroke="#111827", strokeWidth=2).encode(x=xs, y=ys))
    layers.append(base.mark_text(dy=-15, fontSize=11).encode(x=xs, y=ys, text="label:N"))
    return alt.layer(*layers).properties(height=280)


@st.cache_data
def entity_network_data(article_id: str, view: str, echo_ids: frozenset):
    """graph.entity_links under the selected view: minus_echo drops echo entities' mentions and
    the links they alone created, minus_boilerplate drops boilerplate statements."""
    T = load(article_id, view)
    sents, m = T["sentences"], T["mentions"]
    if m is None or T["edges"] is None:
        return None
    if view == "minus_echo":
        m = m[~m["entity_id"].isin(echo_ids)]
    elif view == "minus_boilerplate":
        sents = sents[~sents["boilerplate"].astype(bool)]
    e = view_edges(T["edges"], T["sentences"], view, set(echo_ids))
    return graph.entity_links(sents, graph.mention_sets(m), e)


def _bezier(x0, y0, x1, y1, n=9):
    """Curve for one link: straight between adjacent columns, bent below the row line when it
    skips the middle column, bulging outward when it stays inside one column."""
    if x0 == x1:
        side = -1 if x0 == 0 else 1
        cx, cy = x0 + side * min(0.45, 0.15 + 0.05 * abs(y1 - y0)), (y0 + y1) / 2
    elif abs(x1 - x0) == 2:
        cx, cy = 1, (y0 + y1) / 2 + 0.9  # dips ~half a row, clear of the inference node
    else:
        return [(x0, y0, 0), (x1, y1, 1)]
    out = []
    for j in range(n):
        t = j / (n - 1)
        out.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
                    (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1, j))
    return out


def entity_network_chart(nodes: pd.DataFrame, links: pd.DataFrame, entity_order: list[str]) -> alt.LayerChart:
    col_x = {c: i for i, (c, _) in enumerate(NET_COLS)}
    row = {e: i for i, e in enumerate(entity_order)}
    nodes = nodes.assign(x=nodes["category"].map(col_x), y=nodes["entity_id"].map(row))
    xy = dict(zip(nodes["node"], zip(nodes["x"], nodes["y"])))
    n_com = dict(zip(nodes["node"], nodes["n_comments"]))
    pts = []
    for k, r in enumerate(links.itertuples()):
        (x0, y0), (x1, y1) = xy[r.a], xy[r.b]
        same = r.a.rsplit("·", 1)[0] == r.b.rsplit("·", 1)[0]
        fwd = f"P({r.b} | {r.a}) = {r.p_b_given_a:.2f}  ({r.n_connected}/{n_com[r.a]} comments)"
        rev = f"P({r.a} | {r.b}) = {r.p_a_given_b:.2f}  ({r.n_connected}/{n_com[r.b]} comments)"
        # one copy per endpoint, so clicking either node picks the link out
        for end in (r.a, r.b):
            for x, y, j in _bezier(x0, y0, x1, y1):
                pts.append({"edge": f"{k}-{end}", "node": end, "x": x, "y": y, "order": j, "p": r.p,
                            "kind": "same entity" if same else "different entities",
                            "link": f"{r.a} — {r.b}", "forward": fwd, "reverse": rev})
    n_rows = len(entity_order)
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.25, 2.35]))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[n_rows - 0.4, -1.1]))
    sel = alt.selection_point(fields=["node"], on="click", clear="dblclick", empty=True)
    layers = []
    if pts:
        layers.append(alt.Chart(pd.DataFrame(pts)).mark_line().encode(
            x=xs, y=ys, detail="edge:N", order="order:Q",
            strokeWidth=alt.StrokeWidth("p:Q", scale=alt.Scale(domain=[0, 1], range=[0.3, 7]),
                                        title="P(connected)"),
            color=alt.Color("kind:N", title="link", scale=alt.Scale(
                domain=["same entity", "different entities"], range=["#374151", "#a3a8b0"])),
            opacity=alt.condition(sel, alt.value(0.4), alt.value(0.03)),
            tooltip=["link:N", alt.Tooltip("forward:N", title=" "),
                     alt.Tooltip("reverse:N", title="  ")]))
    base = alt.Chart(nodes)
    layers.append(base.mark_circle(opacity=1, stroke="white", strokeWidth=1).encode(
        x=xs, y=ys,
        size=alt.Size("n_comments:Q", scale=alt.Scale(range=[30, 700]), title="comments"),
        color=alt.Color("category:N", legend=None,
                        scale=alt.Scale(domain=list(NET_COLOR), range=list(NET_COLOR.values()))),
        tooltip=["node:N", "n_comments:Q"]).add_params(sel))
    labels = pd.DataFrame({"x": -0.18, "y": range(n_rows), "entity": entity_order})
    layers.append(alt.Chart(labels).mark_text(align="right", fontSize=11).encode(x=xs, y=ys, text="entity:N"))
    heads = pd.DataFrame({"x": range(len(NET_COLS)), "y": -0.9, "h": [h for _, h in NET_COLS],
                          "category": [c for c, _ in NET_COLS]})
    layers.append(alt.Chart(heads).mark_text(fontSize=12, fontWeight="bold").encode(
        x=xs, y=ys, text="h:N",
        color=alt.Color("category:N", legend=None,
                        scale=alt.Scale(domain=list(NET_COLOR), range=list(NET_COLOR.values())))))
    return alt.layer(*layers).properties(height=26 * n_rows + 70).resolve_scale(color="independent")


@st.cache_data
def directed_network_data(article_id: str, view: str, echo_ids: frozenset):
    """graph.directed_entity_links under the selected view (same filtering as entity_network_data)."""
    T = load(article_id, view)
    sents, m = T["sentences"], T["mentions"]
    if m is None or T["edges"] is None:
        return None
    if view == "minus_echo":
        m = m[~m["entity_id"].isin(echo_ids)]
    elif view == "minus_boilerplate":
        sents = sents[~sents["boilerplate"].astype(bool)]
    e = view_edges(T["edges"], T["sentences"], view, set(echo_ids))
    return graph.directed_entity_links(sents, graph.mention_sets(m), e)


def _directed_curve(x0, y0, x1, y1, n=10):
    """Quadratic curve bent to the left of travel, so a -> b and b -> a take opposite sides; links
    that skip the middle column bend further to clear it."""
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    bend = 0.55 if abs(dx) == 2 else (0.35 if dx == 0 else 0.18)
    cx, cy = (x0 + x1) / 2 - dy / norm * bend, (y0 + y1) / 2 + dx / norm * bend
    return [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
             (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1, t)
            for t in (j / (n - 1) for j in range(n))]


def directed_network_chart(nodes: pd.DataFrame, links: pd.DataFrame, entity_order: list[str]) -> alt.LayerChart:
    """Same VO | INF | PK column layout as entity_network_chart. Each link a -> b is a tapered band,
    wide where it leaves a and thin where it reaches b, coloured by a's category."""
    col_x = {c: i for i, (c, _) in enumerate(NET_COLS)}
    row = {e: i for i, e in enumerate(entity_order)}
    nodes = nodes.assign(x=nodes["category"].map(col_x), y=nodes["entity_id"].map(row))
    xy = dict(zip(nodes["node"], zip(nodes["x"], nodes["y"])))
    pts = []
    for k, r in enumerate(links.itertuples()):
        (x0, y0), (x1, y1) = xy[r.src], xy[r.dst]
        fwd = f"P({r.dst} after {r.src}) = {r.p:.2f}  ({r.n_forward} comments)"
        rev = f"P({r.src} after {r.dst}) = {r.p_reverse:.2f}  ({r.n_reverse} comments)"
        for end in (r.src, r.dst):
            for x, y, t in _directed_curve(x0, y0, x1, y1):
                pts.append({"edge": f"{k}-{end}", "node": end, "x": x, "y": y, "t": t,
                            "w": r.p * (1 - 0.95 * t), "src_cat": r.src.rsplit("·", 1)[1],
                            "link": f"{r.src} → {r.dst}", "forward": fwd, "reverse": rev,
                            "share": f"{r.forward_share:.0%} of this pair's orderings go this way"})
    n_rows = len(entity_order)
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.25, 2.35]))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[n_rows - 0.4, -1.1]))
    cscale = alt.Scale(domain=list(NET_COLOR), range=list(NET_COLOR.values()))
    sel = alt.selection_point(fields=["node"], on="click", clear="dblclick", empty=True)
    layers = []
    if pts:
        layers.append(alt.Chart(pd.DataFrame(pts)).mark_trail().encode(
            x=xs, y=ys, detail="edge:N", order="t:Q",
            size=alt.Size("w:Q", scale=alt.Scale(domain=[0, 1], range=[0.3, 20]), legend=None),
            color=alt.Color("src_cat:N", scale=cscale, title="from"),
            opacity=alt.condition(sel, alt.value(0.45), alt.value(0.03)),
            tooltip=["link:N", alt.Tooltip("forward:N", title=" "), alt.Tooltip("reverse:N", title="  "),
                     alt.Tooltip("share:N", title="   ")]))
    layers.append(alt.Chart(nodes).mark_circle(opacity=1, stroke="white", strokeWidth=1).encode(
        x=xs, y=ys,
        size=alt.Size("n_comments:Q", scale=alt.Scale(range=[30, 700]), title="comments"),
        color=alt.Color("category:N", legend=None, scale=cscale),
        tooltip=["node:N", "n_comments:Q"]).add_params(sel))
    labels = pd.DataFrame({"x": -0.18, "y": range(n_rows), "entity": entity_order})
    layers.append(alt.Chart(labels).mark_text(align="right", fontSize=11).encode(x=xs, y=ys, text="entity:N"))
    heads = pd.DataFrame({"x": range(len(NET_COLS)), "y": -0.9, "h": [h for _, h in NET_COLS],
                          "category": [c for c, _ in NET_COLS]})
    layers.append(alt.Chart(heads).mark_text(fontSize=12, fontWeight="bold").encode(
        x=xs, y=ys, text="h:N", color=alt.Color("category:N", legend=None, scale=cscale)))
    return alt.layer(*layers).properties(height=26 * n_rows + 70).resolve_scale(size="independent")


FLOW_POS = {"INF": (0.0, 0.0), "VO": (-1.0, 1.55), "PK": (1.0, 1.55)}  # triangle, inference on top


def entity_flow_chart(links: pd.DataFrame, n_label: int = 3) -> alt.LayerChart:
    """One node per category; one curved edge per entity between each pair of categories, as a
    fan of parallel curves. `links` holds same-entity, cross-category rows of entity_links."""
    pts, labels = [], []
    for (ca, cb), grp in links.groupby(["cat_a", "cat_b"]):
        (x0, y0), (x1, y1) = FLOW_POS[ca], FLOW_POS[cb]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5
        px, py = -dy / norm, dx / norm  # unit normal to the pair's axis
        grp = grp.sort_values("p", ascending=False).reset_index(drop=True)
        n = len(grp)
        # strongest edge in the middle, weaker ones alternating outward
        slots = [0] + [s for k in range(1, n) for s in (k, -k)][: n - 1]
        spread = min(0.1, 1.0 / max(n, 1))
        for r, slot in zip(grp.itertuples(), slots):
            off = slot * spread * 2  # control-point offset; the curve peaks at half of it
            cx, cy = mx + px * off, my + py * off
            for j in range(13):
                t = j / 12
                pts.append({"x": (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
                            "y": (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1,
                            "order": j, "edge": f"{r.entity_id}|{ca}|{cb}", "entity": r.entity_id,
                            "p": r.p, "pair": f"{ca} — {cb}", "forward": r.forward, "reverse": r.reverse})
            if r.Index < n_label:
                labels.append({"x": mx + px * off / 2, "y": my + py * off / 2, "entity": r.entity_id})
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.6, 1.6]))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[2.2, -0.45]))
    sel = alt.selection_point(fields=["entity"], on="click", clear="dblclick", empty=True)
    layers = []
    if pts:
        layers.append(alt.Chart(pd.DataFrame(pts)).mark_line().encode(
            x=xs, y=ys, detail="edge:N", order="order:Q",
            strokeWidth=alt.StrokeWidth("p:Q", scale=alt.Scale(domain=[0, 1], range=[0.3, 9]),
                                        title="P(connected)"),
            color=alt.Color("p:Q", legend=None,
                            scale=alt.Scale(domain=[0, 1], range=["#e5e7eb", "#111827"])),
            opacity=alt.condition(sel, alt.value(0.8), alt.value(0.06)),
            tooltip=["entity:N", "pair:N", alt.Tooltip("forward:N", title=" "),
                     alt.Tooltip("reverse:N", title="  ")]).add_params(sel))
    if labels:
        layers.append(alt.Chart(pd.DataFrame(labels)).mark_text(
                fontSize=11, fontWeight="bold", color="#b91c1c").encode(
            x=xs, y=ys, text="entity:N"))
    nodes = pd.DataFrame([{"x": x, "y": y, "category": c, "name": dict(NET_COLS)[c]}
                          for c, (x, y) in FLOW_POS.items()])
    cscale = alt.Scale(domain=list(NET_COLOR), range=list(NET_COLOR.values()))
    layers.append(alt.Chart(nodes).mark_circle(size=2600, opacity=1).encode(
        x=xs, y=ys, color=alt.Color("category:N", legend=None, scale=cscale)))
    layers.append(alt.Chart(nodes).mark_text(fontSize=13, fontWeight="bold", color="white").encode(
        x=xs, y=ys, text="category:N"))
    layers.append(alt.Chart(nodes).mark_text(fontSize=12, dy=-42).encode(
        x=xs, y=ys, text="name:N", color=alt.Color("category:N", legend=None, scale=cscale)))
    return alt.layer(*layers).properties(height=520)


def missing(name: str, extra: str = "") -> None:
    st.info(f"`{name}` is not available yet for this article/view "
             "(the background metrics job hasn't written it, or it doesn't apply). " + extra)


def headline_from_tables(lens: pd.DataFrame | None, nd: pd.DataFrame | None,
                          summary: dict | None, view: str) -> dict:
    """Best-effort headline stats for one view, preferring the cheap tables that are
    written early over summary.json (written last, per A8.5's timing note)."""
    out: dict = {}
    if summary is not None and view in summary.get("views", {}):
        out.update(summary["views"][view])
        out["n_null"] = summary.get("n_null")
        return out
    if lens is not None and len(lens):
        out["mean_chain_length"] = float((lens["length"] * lens["share"]).sum())
        out["n_chains"] = int((lens["n_chains"]).sum())
    if nd is not None and "mean_chain_length" in out and len(nd):
        z, p = z_against_null(out["mean_chain_length"], nd["mean_length"])
        out["null"] = {"mean_length_z": z, "mean_length_p": p}
        out["n_null"] = len(nd)
    return out


st.set_page_config(page_title="Claim 2.2 — entity chains", layout="wide")

_articles = common.coded_articles()
# No prompt_variant() in format_func: it loads every article's sentence table, which blanked the
# page for ~20 s on first load. The selected article's variant is shown below instead.
aid = st.sidebar.selectbox("Article", _articles)
view = st.sidebar.radio("View", VIEWS, format_func=lambda v: v.replace("_", " "))
st.sidebar.caption(VIEW_CAPTION)

T = load(aid, view)
groups_df = T["groups"]
sents_all = T["sentences"]

# A8.2 -- article context, computed directly from the committed dictionary and the sentence
# table so it works even when out/<article_id>/ is empty.
n_groups = len(groups_df)
n_echo = int(groups_df["prompt_echo"].fillna(False).astype(bool).sum()) if "prompt_echo" in groups_df.columns else 0
boilerplate_share = float(sents_all["boilerplate"].astype(bool).mean()) if len(sents_all) else 0.0
cat_counts = sents_all["category"].value_counts()
sufficiency = {cat: int(cat_counts.get(cat, 0)) >= SUFFICIENT_N for cat in common.CATEGORIES}
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{aid}** · {common.prompt_variant(aid)}")
st.sidebar.write(f"{n_groups} entity groups, {n_echo} flagged `prompt_echo`")
st.sidebar.write(f"boilerplate share: {boilerplate_share:.1%}")
st.sidebar.write("sufficient (n ≥ 30): " +
                  ", ".join(f"{cat} {'✓' if ok else '✗'}" for cat, ok in sufficiency.items()))

s = T["summary"]
st.title(f"Claim 2.2 — entity chains · article {aid}")

n_comments = int(sents_all["comment_index"].nunique()) if len(sents_all) else 0
n_sentences = len(sents_all)
st.caption(f"{n_comments} comments · {n_sentences} statements · {n_groups} entity groups")

st.subheader("Entity network")
_echo_ids = frozenset(groups_df.loc[groups_df["prompt_echo"].fillna(False).astype(bool), "id"]
                      if "prompt_echo" in groups_df.columns else [])
_net = entity_network_data(aid, view, _echo_ids)
if _net is None:
    missing("mentions.csv / edges.csv", "so the entity network can't be drawn.")
else:
    net_nodes, net_links = _net
    ent_rank = (net_nodes.groupby("entity_id")["n_comments"].sum().sort_values(ascending=False))
    n1, n2, n3 = st.columns(3)
    top_n = n1.slider("Entities (most frequent first)", 1, len(ent_rank), min(20, len(ent_rank)))
    min_p = n2.slider("Min probability", 0.0, 1.0, 0.4, 0.05)
    min_n = n3.number_input("Min comments where connected", 1, 10_000, 5)
    keep_ents = list(ent_rank.index[:top_n])
    shown_nodes = net_nodes[net_nodes["entity_id"].isin(keep_ents)]
    keep_nodes = set(shown_nodes["node"])
    shown_links = net_links[net_links["a"].isin(keep_nodes) & net_links["b"].isin(keep_nodes)
                            & (net_links["p"] >= min_p) & (net_links["n_connected"] >= min_n)]
    st.caption(
        "Each node is an entity in one high-level category (sized by how many comments mention it "
        "there). Two nodes are connected in a comment when they are mentioned in the same statement "
        "or in two statements that share an entity. Edge width is P(connected | node appears): the "
        "share of comments containing one node in which it connects to the other, taking the larger "
        "direction; hover an edge for both. Dark edges join the same entity across categories. "
        f"Click a node to isolate its links, double-click to reset. Showing {len(shown_links)} of "
        f"{len(net_links)} links (view: {view.replace('_', ' ')}).")
    st.altair_chart(entity_network_chart(shown_nodes, shown_links, keep_ents), use_container_width=True)
    with st.expander("Link table"):
        st.dataframe(shown_links.sort_values("p", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("Entities across categories")
    flow = shown_links.assign(ent_a=shown_links["a"].str.rsplit("·", n=1).str[0],
                              cat_a=shown_links["a"].str.rsplit("·", n=1).str[1],
                              ent_b=shown_links["b"].str.rsplit("·", n=1).str[0],
                              cat_b=shown_links["b"].str.rsplit("·", n=1).str[1])
    flow = flow[(flow["ent_a"] == flow["ent_b"]) & (flow["cat_a"] != flow["cat_b"])].copy()
    flow["entity_id"] = flow["ent_a"]
    # orient every pair the same way (VO/INF/PK order) so each category pair is one fan
    rank_c = {c: i for i, (c, _) in enumerate(NET_COLS)}
    swap = flow["cat_a"].map(rank_c) > flow["cat_b"].map(rank_c)
    flow.loc[swap, ["a", "b", "cat_a", "cat_b", "p_b_given_a", "p_a_given_b"]] = \
        flow.loc[swap, ["b", "a", "cat_b", "cat_a", "p_a_given_b", "p_b_given_a"]].values
    n_com = dict(zip(net_nodes["node"], net_nodes["n_comments"]))
    flow["forward"] = [f"P({b} | {a}) = {p:.2f}  ({k}/{n_com[a]} comments)"
                       for a, b, p, k in zip(flow["a"], flow["b"], flow["p_b_given_a"], flow["n_connected"])]
    flow["reverse"] = [f"P({a} | {b}) = {p:.2f}  ({k}/{n_com[b]} comments)"
                       for a, b, p, k in zip(flow["a"], flow["b"], flow["p_a_given_b"], flow["n_connected"])]
    st.caption(
        "One node per category. Each edge is one entity: it joins two categories when that entity, "
        "mentioned in one, is connected to the same entity mentioned in the other within a comment. "
        "Width is the same P(connected) as above (the dark edges there); the strongest edge of each "
        "fan sits in the middle, darker edges are more probable, and the top three per pair are labelled. Hover for both directions; "
        "click an edge to pick that entity out across all three pairs. Uses the filters above: "
        f"{len(flow)} entity edges.")
    if len(flow):
        st.altair_chart(entity_flow_chart(flow), use_container_width=True)
    else:
        st.info("No entity links across categories at these filter settings.")

    st.subheader("Directed entity network (reading order)")
    _dnet = directed_network_data(aid, view, _echo_ids)
    if _dnet is None:
        missing("mentions.csv / edges.csv", "so the directed network can't be drawn.")
    else:
        d_nodes, d_links = _dnet
        d1, d2, d3, d4 = st.columns(4)
        d_min_p = d1.slider("Min P(b after a)", 0.0, 1.0, 0.25, 0.05, key="d_min_p")
        d_min_n = d2.number_input("Min comments a → b", 1, 10_000, 5, key="d_min_n")
        d_dominant = d3.checkbox("Dominant direction only", value=True, key="d_dom",
                                 help="Of a → b and b → a, keep only the one seen in more comments.")
        d_cross = d4.checkbox("Across categories only", value=True, key="d_cross")
        dl = d_links[d_links["src"].isin(keep_nodes) & d_links["dst"].isin(keep_nodes)
                     & (d_links["p"] >= d_min_p) & (d_links["n_forward"] >= d_min_n)]
        if d_dominant:
            dl = dl[(dl["n_forward"] > dl["n_reverse"])
                    | ((dl["n_forward"] == dl["n_reverse"]) & (dl["src"] < dl["dst"]))]
        if d_cross:
            dl = dl[dl["src"].str.rsplit("·", n=1).str[1] != dl["dst"].str.rsplit("·", n=1).str[1]]
        st.caption(
            "Same nodes and entity filter as the entity network. An edge a → b means that, within a "
            "comment, a statement mentioning a is linked (shares an entity) with a later statement "
            "mentioning b. Width at the start is P(b after a): the share of comments containing a in "
            "which this happens, estimated across all the article's comments. Edges taper toward "
            "their target and take the colour of their source. Direction is the order readers wrote "
            "in, not an inferred dependency; mentions in the same statement have no order and are "
            f"left out. Showing {len(dl)} of {len(d_links)} directed links (view: {view.replace('_', ' ')}).")
        # which way readers go between each pair of categories, over all directed links (unfiltered)
        _sc = d_links["src"].str.rsplit("·", n=1).str[1]
        _dc = d_links["dst"].str.rsplit("·", n=1).str[1]
        _pair = d_links.assign(sc=_sc, dc=_dc).groupby(["sc", "dc"])["n_forward"].sum()
        _dirs = []
        for a, b in [("VO", "INF"), ("INF", "PK"), ("VO", "PK")]:
            fwd, back = int(_pair.get((a, b), 0)), int(_pair.get((b, a), 0))
            if fwd + back:
                _dirs.append(f"{a} → {b} {fwd / (fwd + back):.0%} of the time (vs {b} → {a})")
        if _dirs:
            st.caption("Across all directed links, before filtering: " + "; ".join(_dirs) + ". "
                       "Counts are entity-pair links, so a comment naming many entities counts many times.")
        if len(dl):
            st.altair_chart(directed_network_chart(d_nodes[d_nodes["node"].isin(keep_nodes)], dl, keep_ents),
                            use_container_width=True)
        else:
            st.info("No directed links at these filter settings.")
        with st.expander("Directed link table"):
            st.dataframe(dl.sort_values("p", ascending=False), hide_index=True, use_container_width=True)

with st.expander("Article-level metrics", expanded=False):
    if s is None:
        st.warning("summary.json has not been written for this article yet — headline metrics "
                   "below are computed live from whichever tables already exist; some may show "
                   "'not available yet'.")

    # Top-line metrics for the currently selected view.
    lens_now, nd_now = T["lengths"], T["null_chains"]
    head_now = headline_from_tables(lens_now, nd_now, s, view)
    cols = st.columns(6)
    coverage_overall = s["coverage"]["overall"] if s is not None else None
    for col, (label, val) in zip(cols, [
            ("comments", n_comments), ("sentences", n_sentences),
            ("coverage", f"{coverage_overall:.0%}" if coverage_overall is not None else "n/a"),
            ("entities", n_groups),
            ("chains", head_now.get("n_chains", "n/a")),
            ("mean length", f"{head_now['mean_chain_length']:.2f}" if "mean_chain_length" in head_now else "n/a")]):
        col.metric(label, val)

    # A8.2 -- per-category coverage, suppressed in favor of "insufficient" when a category has
    # fewer than SUFFICIENT_N sentences (sufficiency itself is computed directly above, not from
    # summary.json, so this degrades correctly even before the article's metrics are written).
    st.markdown("**Coverage by category**")
    cov_cols = st.columns(len(common.CATEGORIES))
    cov = s["coverage"] if s is not None else {}
    for col, cat in zip(cov_cols, common.CATEGORIES):
        if not sufficiency.get(cat, False):
            col.metric(cat, f"insufficient (n < {SUFFICIENT_N})")
        elif cat in cov:
            col.metric(cat, f"{cov[cat]:.0%}")
        else:
            col.metric(cat, "n/a")

    # A8.1 -- the three views side by side, from summary.json's `views` object.
    st.markdown("**Headline numbers by view**")
    if s is None:
        missing("summary.json", "so the per-view comparison below can't be shown.")
    else:
        n_null = s.get("n_null")
        rows = []
        for field, label, fmt in HEADLINE_FIELDS:
            row = {"metric": label}
            for v in VIEWS:
                vd = s.get("views", {}).get(v)
                row[v] = fmt.format(vd[field]) if vd and field in vd else "n/a"
            rows.append(row)
        null_row_z = {"metric": f"null mean-length z (n_null={n_null})"}
        null_row_p = {"metric": "null mean-length p"}
        for v in VIEWS:
            vd = s.get("views", {}).get(v, {}).get("null", {})
            null_row_z[v] = f"{vd['mean_length_z']:.2f}" if "mean_length_z" in vd else "n/a"
            null_row_p[v] = f"{vd['mean_length_p']:.3f}" if "mean_length_p" in vd else "n/a"
        rows += [null_row_z, null_row_p]
        st.dataframe(pd.DataFrame(rows).set_index("metric"), use_container_width=True)

tabs = st.tabs(["Comments", "Statement graphs", "Token dictionary", "Paths", "Entities", "Sparsity", "Figures"])
st.markdown(TOKEN_CSS, unsafe_allow_html=True)

echo_ids = (set(groups_df.loc[groups_df["prompt_echo"].fillna(False).astype(bool), "id"])
            if "prompt_echo" in groups_df.columns else set())
entity_meta = {r.id: (r.canonical, r.type) for r in groups_df.itertuples()}
M = T["mentions"]
mentions_by_sent = ({k: list(g.itertuples()) for k, g in M.groupby(["comment_index", "sent_id"])}
                    if M is not None else {})

with tabs[0]:
    if not len(sents_all):
        missing("sentences")
    else:
        E = view_edges(T["edges"], sents_all, view, echo_ids) if T["edges"] is not None else None
        edges_by_comment = {c: g for c, g in E.groupby("comment_index")} if E is not None else {}
        if E is None:
            missing("edges.csv", "statements are shown without the entity graph.")
        st.caption(
            "Each statement is labelled with its claim 1 category and subcategory (legend below; "
            "graph nodes use the same colours). Dictionary "
            "tokens are highlighted by entity type (" +
            " ".join(f'<span class="tok" style="background:{c}">{t}</span>' for t, c in TYPE_COLOR.items()) +
            ") with the entity id in subscript. Beside each comment is its graph: a line joins two "
            "statements that mention the same entity (hover it for the shared entities). The network "
            "layout pulls linked statements together, with unlinked ones along the bottom; the arc "
            "layout keeps reading order. A dark ring marks a bridge "
            "statement (removing it splits the comment's graph). "
            f"View: <b>{view.replace('_', ' ')}</b>" +
            (" — echo tokens are dotted and don't create links." if view == "minus_echo" else
             " — boilerplate statements are faded and don't create links." if view == "minus_boilerplate" else "."),
            unsafe_allow_html=True)
        st.markdown(legend_html(), unsafe_allow_html=True)

        per_comment = sents_all.groupby("comment_index")
        graph_rows = ({int(r.comment_index): r for r in T["comment_graphs"].itertuples()}
                      if T["comment_graphs"] is not None else {})
        stats = pd.DataFrame({
            "n_statements": per_comment.size(),
            "n_links": pd.Series({c: len(g) for c, g in edges_by_comment.items()}, dtype=int)}).fillna(0)
        f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
        query = f1.text_input("Search comment text / entity id", "")
        min_n = f2.number_input("Min statements", 1, int(stats["n_statements"].max()), 1)
        order = f3.selectbox("Sort", ["reading order", "most links", "most statements"])
        layout = f4.selectbox("Graph", ["network", "reading-order arcs"])
        ids = stats[stats["n_statements"] >= min_n]
        if query.strip():
            q = query.strip().lower()
            hit_text = set(sents_all.loc[sents_all["text"].astype(str).str.lower().str.contains(q, regex=False),
                                         "comment_index"])
            hit_ent = set(M.loc[M["entity_id"].str.lower() == q, "comment_index"]) if M is not None else set()
            ids = ids[ids.index.isin(hit_text | hit_ent)]
        if order == "most links":
            ids = ids.sort_values(["n_links", "n_statements"], ascending=False, kind="stable")
        elif order == "most statements":
            ids = ids.sort_values(["n_statements", "n_links"], ascending=False, kind="stable")
        comment_ids = list(ids.index)
        n_pages = max(1, -(-len(comment_ids) // PAGE_SIZE))
        page = st.number_input(f"Page (of {n_pages}; {len(comment_ids)} comments)", 1, n_pages, 1)
        for ci in comment_ids[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
            sents = per_comment.get_group(ci)
            ce = edges_by_comment.get(ci, pd.DataFrame(columns=["sent_a", "sent_b", "shared"]))
            with st.container(border=True):
                bridges = bridge_statements(sents, ce) if len(ce) else set()
                gm = graph_rows.get(ci)
                extra = ""
                if gm is not None:
                    extra = (f" · cycle rank {int(gm.cycle_rank)} · largest clique {int(gm.max_clique)}"
                             f" · {int(gm.n_components)} component{'s' if gm.n_components != 1 else ''}")
                if bridges:
                    cats = dict(zip(sents["sent_id"].astype(int), sents["category"]))
                    extra += " · bridges: " + ", ".join(f"S{b} ({cats[b]})" for b in sorted(bridges))
                st.markdown(f"**Comment {ci}** · {len(sents)} statements · {len(ce)} links{extra}")
                left, right = st.columns([3, 2])
                lines = []
                for r in sents.itertuples():
                    off = view == "minus_boilerplate" and bool(r.boilerplate)
                    dim = echo_ids if view == "minus_echo" else set()
                    body = token_html(str(r.text), mentions_by_sent.get((ci, r.sent_id), []), entity_meta, dim)
                    lines.append(f'<div class="stmt{" off" if off else ""}"><span class="sid">S{r.sent_id}</span>'
                                 f'{stmt_badge(r.subcat, r.group)}'
                                 f'{"<span class=grp>· boilerplate</span>" if r.boilerplate else ""}<br>{body}</div>')
                left.markdown("".join(lines), unsafe_allow_html=True)
                if E is not None:
                    draw = network_graph if layout == "network" else arc_graph
                    right.altair_chart(draw(sents, ce, frozenset(bridges)), use_container_width=True)

with tabs[1]:
    st.caption(
        "Each comment is an undirected graph: statements are nodes, and two statements are joined "
        "when they share an entity. These are article-level summaries of those graphs, each "
        "compared with the within-comment permutation null (entity mentions reshuffled across a "
        "comment's statements, keeping each statement's mention count). Because every statement "
        "about the same entity is joined to every other, cliques and cycles are common by "
        "construction — the null, not the raw value, is the comparison that matters. "
        f"View: {view.replace('_', ' ')}.")
    gsum = s.get("views", {}).get(view, {}).get("graph") if s is not None else None
    if not gsum:
        missing("statement-graph metrics in summary.json",
                "re-run `metrics.write` for this article to add them.")
    else:
        rows = []
        for key, label, desc in GRAPH_METRICS:
            v = gsum.get(key)
            if v is None:
                continue
            rows.append({"measure": label, "observed": v["observed"], "null mean": v["null_mean"],
                         "z": v["z"], "p": v["p"], "meaning": desc})
        st.dataframe(pd.DataFrame(rows).style.format(
            {"observed": "{:.3f}", "null mean": "{:.3f}", "z": "{:+.2f}", "p": "{:.3f}"}),
            hide_index=True, use_container_width=True)
        st.caption(f"n_null = {s.get('n_null')} draws. z > 0: higher than chance.")
        ng = T["null_graphs"]
        if ng is not None:
            labels = {k: l for k, l, _ in GRAPH_METRICS if k in ng.columns}
            pick = st.selectbox("Null distribution for", list(labels), format_func=labels.get)
            obs = gsum[pick]["observed"]
            hist = alt.Chart(ng).mark_bar(color="#a3a8b0").encode(
                x=alt.X(f"{pick}:Q", bin=alt.Bin(maxbins=40), title=labels[pick]),
                y=alt.Y("count():Q", title="null draws"))
            rule = alt.Chart(pd.DataFrame({"x": [obs]})).mark_rule(color="#b91c1c", strokeWidth=2).encode(
                x="x:Q", tooltip=[alt.Tooltip("x:Q", title="observed", format=".3f")])
            st.altair_chart((hist + rule).properties(height=220), use_container_width=True)
        pc = T["comment_graphs"]
        if pc is not None and len(pc):
            st.markdown("**Per-comment distribution**")
            col = st.selectbox("Measure", ["cycle_rank", "max_clique", "density", "n_components",
                                           "n_articulation", "n_isolated"])
            st.altair_chart(alt.Chart(pc).mark_bar(color="#4b5563").encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count():Q", title="comments")
            ).properties(height=200), use_container_width=True)

with tabs[2]:
    g = groups_df.copy()
    types = st.multiselect("Type", sorted(g["type"].unique()), default=sorted(g["type"].unique()))
    g = g[g["type"].isin(types)]
    if M is None:
        missing("mentions.csv", "showing the dictionary without match counts.")
        g["surface_forms"] = g["surface_forms"].map(lambda f: ", ".join(f))
        st.dataframe(g, use_container_width=True, hide_index=True)
    else:
        counts = M.groupby("entity_id").agg(
            n_mentions=("key", "size"), n_statements=("key", "nunique"),
            n_comments=("comment_index", "nunique")).reset_index()
        form_counts = M.assign(form=M["surface"].astype(str).str.lower()).groupby(["entity_id", "form"]).size()
        def forms_with_counts(row):
            return ", ".join(f"{f} ({int(form_counts.get((row.id, f.lower()), 0))})" for f in row.surface_forms)
        g["surface_forms (matches)"] = [forms_with_counts(r) for r in g.itertuples()]
        tbl = g.merge(counts, left_on="id", right_on="entity_id", how="left")
        for c in ("n_mentions", "n_statements", "n_comments"):
            tbl[c] = tbl[c].fillna(0).astype(int)
        cols_show = ["id", "canonical", "type", "surface_forms (matches)", "n_mentions", "n_statements", "n_comments"]
        if "prompt_echo" in tbl.columns:
            tbl["prompt_echo"] = tbl["prompt_echo"].fillna(False).astype(bool)
            cols_show.insert(3, "prompt_echo")
        if "notes" in tbl.columns:
            cols_show.append("notes")
        tbl = tbl.sort_values("n_mentions", ascending=False)
        st.caption(f"{len(tbl)} entity groups · {int(tbl['n_mentions'].sum())} token matches. "
                   "Counts in parentheses are how often each surface form matched in this article.")
        st.dataframe(tbl[cols_show], use_container_width=True, hide_index=True)
        if len(tbl):
            pick = st.selectbox("Show statements mentioning", tbl["id"],
                                format_func=lambda i: f"{i} — {entity_meta.get(i, (i,))[0]}")
            sel = M[M["entity_id"] == pick][["comment_index", "sent_id"]].drop_duplicates()
            st.caption(f"{len(sel)} statements" + (" (showing first 200)" if len(sel) > 200 else ""))
            by_key = sents_all.set_index(["comment_index", "sent_id"])
            lines = []
            for ci, sid in sel.head(200).itertuples(index=False):
                r = by_key.loc[(ci, sid)]
                body = token_html(str(r["text"]), mentions_by_sent.get((ci, sid), []), entity_meta, set())
                lines.append(f'<div class="stmt"><span class="sid">C{ci}·S{sid}</span>'
                             f'{stmt_badge(r["subcat"], r["group"])} {body}</div>')
            st.markdown("".join(lines), unsafe_allow_html=True)

with tabs[3]:
    lens, nd = T["lengths"], T["null_chains"]
    if lens is None:
        missing("lengths.csv")
    else:
        obs = alt.Chart(lens).mark_bar().encode(x="length:O", y="share:Q").properties(height=240)
        st.altair_chart(obs, use_container_width=True)
        h = head_now
        if nd is None:
            missing("null_chains.csv", "showing observed mean length only.")
            st.caption(f"observed mean length {h.get('mean_chain_length', float('nan')):.2f}")
        else:
            n_null = h.get("n_null", len(nd))
            null_stats = h.get("null", {})
            if "mean_length_z" in null_stats:
                st.caption(f"observed mean length {h['mean_chain_length']:.2f}; "
                           f"null mean {nd['mean_length'].mean():.2f} (n_null={n_null}); "
                           f"z {null_stats['mean_length_z']:.2f}, p {null_stats['mean_length_p']:.3f}")
            else:
                st.caption(f"observed mean length {h.get('mean_chain_length', float('nan')):.2f}; "
                           f"null mean {nd['mean_length'].mean():.2f} (n_null={n_null})")
    sig = T["signatures"]
    if sig is None:
        missing("signatures.csv")
    else:
        st.dataframe(sig, hide_index=True, use_container_width=True)
        if len(sig):
            pick = st.selectbox("Comments producing signature", sig["signature"].head(50))
            if T["chains"] is None:
                missing("chains.csv")
            else:
                st.dataframe(T["chains"][T["chains"]["signature"] == pick][
                    ["comment_index", "sent_ids", "length", "entities"]], hide_index=True, use_container_width=True)
    st.markdown("**Many paths to the same insight**")
    if T["paths_to_insight"] is None:
        missing("paths_to_insight.csv")
    else:
        st.dataframe(T["paths_to_insight"], hide_index=True, use_container_width=True)

with tabs[4]:
    br = T["bridging"]
    if br is None:
        missing("bridging.csv")
    else:
        n_bridge = int(br["bridges"].sum())
        n_never = int((~br["bridges"]).sum())
        st.caption(f"{n_bridge} entities bridge at least two sentences; "
                   f"{n_never} appear but never bridge. (view: {view})")
        only = st.radio("Show", ["bridging", "never bridging", "all"], horizontal=True)
        view_sel = br if only == "all" else br[br["bridges"] == (only == "bridging")]
        st.dataframe(view_sel, hide_index=True, use_container_width=True)

with tabs[5]:
    cross = load_cross_article()
    st.markdown("**Prompt-variant comparison (headline)**")
    by_variant = cross["by_variant"]
    if by_variant is None:
        missing("cross_article_by_variant.csv", "run `write_cross_article` once the per-article "
                "job finishes to get the community_asked / community_not_asked contrast.")
    else:
        for field, lo, hi, title in [("mentions_per_sentence", "mps_lo", "mps_hi", "mentions per sentence"),
                                     ("distinct_per_sentence", "dps_lo", "dps_hi", "distinct entities per sentence")]:
            bars = alt.Chart(by_variant).mark_bar().encode(
                x=alt.X("category:N", title=None), y=alt.Y(f"{field}:Q", title=title),
                color=alt.Color("category:N", scale=alt.Scale(domain=list(CAT_COLOR), range=list(CAT_COLOR.values()))))
            err = alt.Chart(by_variant).mark_rule().encode(
                x="category:N", y=f"{lo}:Q", y2=f"{hi}:Q")
            chart = alt.layer(bars, err).facet(column=alt.Column("prompt_variant:N", title="prompt variant"))
            st.altair_chart(chart)
        st.dataframe(by_variant, hide_index=True, use_container_width=True)

    st.markdown("**Per-article spread**")
    cs = cross["summary"]
    if cs is None:
        missing("cross_article_summary.csv")
    else:
        xcol, ycol = "coverage", f"mean_chain_length__{view}"
        if ycol not in cs.columns:
            st.caption(f"`{ycol}` not present in cross_article_summary.csv yet; showing available columns.")
            st.dataframe(cs, hide_index=True, use_container_width=True)
        else:
            pts = alt.Chart(cs).mark_point(size=90, filled=True).encode(
                x=alt.X(f"{xcol}:Q", title="coverage"),
                y=alt.Y(f"{ycol}:Q", title=f"mean chain length ({view})"),
                color=alt.Color("prompt_variant:N"),
                tooltip=["article_id", "prompt_variant", xcol, ycol, f"n_chains__{view}"])
            st.altair_chart(pts, use_container_width=True)
            st.dataframe(cs, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Loading figure tables…")
def load_figures():
    return figures.read()


with tabs[6]:
    FT = load_figures()
    if FT is None:
        missing("out/figures/*.csv", "run `uv run python -m analysis.claim2_2.figures` to build them.")
    else:
        scope = st.radio("Scope", ["all articles", f"article {aid}"], horizontal=True, key="fig_scope")
        one = scope != "all articles"

        def pick(k):
            return FT[k][FT[k]["article_id"] == aid] if one else FT[k]

        st.caption("View `all`. A path is a reading-order chain of two or more statements that each share an "
                   "entity with the next; each step is carried by the shared entity. GoFish versions and the "
                   "CSVs behind every chart are in `out/figures/`.")

        st.markdown("#### 1 · Common paths")
        if pick("steps").empty:
            st.caption("No multi-statement paths in this article.")
        else:
            st.altair_chart(figures.flow_chart(pick("steps")), use_container_width=False)
            st.altair_chart(figures.path_types_chart(pick("types")), use_container_width=False)

        st.markdown("#### 2 · What paths look like")
        ex = FT["exemplars"]
        if one:
            ex = ex[ex["article_id"] == aid]
        if ex.empty:
            st.caption("No exemplar paths were drawn from this article; switch scope to all articles.")
        else:
            st.altair_chart(figures.exemplar_chart(ex), use_container_width=False)

        st.markdown("#### 3 · How long are paths?")
        if one:
            st.altair_chart(figures.length_chart(pick("lengths"), f"Article {aid}: chain lengths vs null"))
        else:
            st.altair_chart(alt.hconcat(figures.length_chart(FT["lengths_pooled"]),
                                        figures.length_excess_heatmap(FT["lengths"])))

        st.markdown("#### 4 · Which entities connect statements?")
        ents = pick("entities")
        if one:
            st.altair_chart(figures.entity_chart(ents, label_top=12))
            st.dataframe(ents.sort_values("n_sentences", ascending=False)[
                ["canonical", "type", "prompt_echo", "n_sentences", "n_linked", "link_rate",
                 "n_VO", "n_INF", "n_PK"]], hide_index=True, use_container_width=True)
        else:
            st.altair_chart(alt.hconcat(figures.entity_chart(ents), figures.entity_type_chart(ents))
                            .resolve_scale(color="independent"))

        st.markdown("#### 5 · Are entities sparser in VO, INF or PK?")
        sp = FT["sparsity"]
        if one:
            st.dataframe(sp[sp["article_id"] == aid], hide_index=True)
        st.altair_chart(alt.hconcat(
            figures.sparsity_chart(sp),
            figures.sparsity_chart(sp, "dictionary_share", "share of the dictionary used"),
            figures.sparsity_summary_chart(sp)))

        st.markdown("#### 7 · How rich is the path structure?")
        st.altair_chart(figures.richness_chart(FT["richness"], highlight=aid if one else None))
        if one:
            st.dataframe(FT["richness"][FT["richness"]["article_id"] == aid].T.astype(str), use_container_width=True)
        else:
            st.altair_chart(figures.transition_multiples(FT["transitions"], FT["richness"]))
