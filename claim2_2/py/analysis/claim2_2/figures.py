"""Data and Altair charts for the claim 2.2 figure panels, built from the saved per-article tables.

    uv run python -m analysis.claim2_2.figures [N_JOBS]   # data -> out/figures/*.csv, then charts

Everything here reads `out/<id>/{chains,edges,mentions}.csv`, the dictionaries and the cross-article
summary; nothing re-tokenizes. View is `all` throughout. A *path* is a reading-order chain
(`graph.chains`) of length >= 2: consecutive statements that share an entity. Each step of a path
is carried by the entities the two statements share, so a path reads
`VO -[data_variable]- INF -[event]- PK`: statements alternate with the kind of entity linking them.

Panels (one CSV each, one row per article-level cell so the dashboard can filter by article):
  1 path_steps.csv      step k -> k+1 transitions by category and linking entity type (flow chart)
    path_types.csv      category sequences of paths and their counts
  2 path_exemplars.csv  two real paths for each of the most common path types (gallery)
  3 path_lengths.csv    observed chain-length distribution and its within-comment null, per article
    path_lengths_null_pooled.csv  pooled over articles, draw by draw
  4 entity_links.csv    per entity: sentences mentioning it and the share linked through it
  5 entity_sparsity.csv per category: mentions per sentence and distinct entities used
  7 richness.csv        per article: path-type count and entropy, cycle rank and its null z
    transitions.csv     adjacent category transitions per article (fingerprint small multiples)
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from multiprocessing import Pool

import altair as alt
import numpy as np
import pandas as pd

from analysis.claim2_2 import common, graph
from analysis.claim2_2.metrics import permute_msets

FIG_DIR = common.OUT_DIR / "figures"
N_NULL = 100
RAREFY_N, RAREFY_DRAWS = 100, 50  # path-type entropy at a common sample size (richness panel)
MAX_STEP = 4                     # flow columns: statements 1..MAX_STEP + 1; later steps are not drawn
LEN_BINS = ["1", "2", "3", "4+"]
OTHER = "other"                  # redundant / uncoded statements
CATS = ["VO", "INF", "PK", OTHER]
CAT_NAME = {"VO": "Visual observation", "INF": "Inference", "PK": "Prior knowledge", OTHER: "Other"}
# claim 1 swatches (dashboard NET_COLOR) plus grey for everything outside the three categories
CAT_COLOR = {"VO": "#4A7FBA", "INF": "#2D6A4F", "PK": "#D68910", OTHER: "#b4b8bd"}
TYPES = list(common.ENTITY_TYPES)
TYPE_NAME = {"entity": "world entity", "data_variable": "chart variable", "value": "chart value",
             "event": "event"}
TYPE_COLOR = {"entity": "#c9a227", "data_variable": "#3f7fbf", "value": "#5aa469", "event": "#b5569b"}
ARROW = "→"


def _cat(c: str) -> str:
    return c if c in common.CATEGORIES else OTHER


def _len_bin(n: int) -> str:
    return str(n) if n < 4 else "4+"


def _cscale(domain=CATS) -> alt.Scale:
    return alt.Scale(domain=[CAT_NAME[c] for c in domain], range=[CAT_COLOR[c] for c in domain])


def _tscale() -> alt.Scale:
    return alt.Scale(domain=[TYPE_NAME[t] for t in TYPES], range=[TYPE_COLOR[t] for t in TYPES])


# ---------------------------------------------------------------------------
# Per-article data
# ---------------------------------------------------------------------------

def _load(aid: str):
    d = common.OUT_DIR / aid
    ch = pd.read_csv(d / "chains.csv")
    ed = pd.read_csv(d / "edges.csv")
    m = pd.read_csv(d / "mentions.csv")
    groups = json.loads((common.DICT_DIR / f"{aid}_entities.json").read_text())
    return ch, ed, m, groups


def _sentence_frame(ch: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the reading-order sentence table (comment, order, sent_id, category) from chains.csv,
    which lists every statement once, in the order `graph.chains` walked them."""
    rows = []
    for ci, sub in ch.sort_values(["comment_index", "chain_id"]).groupby("comment_index"):
        pos = 0
        for r in sub.itertuples():
            for s, c in zip(str(r.sent_ids).split(graph.JOIN), str(r.signature_raw).split("-")):
                rows.append({"comment_index": int(ci), "order": pos, "sent_id": int(s), "category": c})
                pos += 1
    return pd.DataFrame(rows)


def article_tables(aid: str, n_null: int = N_NULL, seed: int = 0) -> dict[str, pd.DataFrame]:
    ch, ed, m, groups = _load(aid)
    meta = {g["id"]: g for g in groups}
    sents = _sentence_frame(ch)
    cat = {(r.comment_index, r.sent_id): r.category for r in sents.itertuples()}
    msets = graph.mention_sets(m)
    etype = {g["id"]: g["type"] for g in groups}
    out: dict[str, pd.DataFrame] = {}

    # 1 -- steps and path types, over reading-order chains of length >= 2
    steps: Counter = Counter()
    types: Counter = Counter()
    cands = []
    for r in ch[ch["length"] >= 2].itertuples():
        ids = [int(s) for s in str(r.sent_ids).split(graph.JOIN)]
        cs = [_cat(cat[(r.comment_index, s)]) for s in ids]
        types[(ARROW.join(cs), len(ids))] += 1
        links = []
        for k in range(len(ids) - 1):
            shared = sorted(msets.get((r.comment_index, ids[k]), set()) & msets.get((r.comment_index, ids[k + 1]), set()))
            links.append(shared)
            if k + 1 > MAX_STEP:        # the flow chart stops at statement MAX_STEP + 1
                continue
            w = 1 / len(shared)
            for e in shared:
                steps[(k + 1, cs[k], cs[k + 1], etype.get(e, "entity"))] += w
        if 2 <= len(ids) <= 4 and OTHER not in cs:
            echo = any(meta.get(e, {}).get("prompt_echo") for sh in links for e in sh)
            cands.append({"article_id": aid, "comment_index": int(r.comment_index), "path": ARROW.join(cs),
                          "sent_ids": r.sent_ids, "echo": bool(echo),
                          "links": json.dumps([[meta.get(e, {}).get("canonical", e) for e in sh] for sh in links])})
    out["steps"] = pd.DataFrame([{"article_id": aid, "step": s, "from_cat": a, "to_cat": b, "link_type": t,
                                  "weight": w} for (s, a, b, t), w in steps.items()])
    out["types"] = pd.DataFrame([{"article_id": aid, "path": p, "length": n, "n": k}
                                 for (p, n), k in types.items()])
    out["cands"] = pd.DataFrame(cands)

    # 3 -- length distribution, observed vs within-comment null (same null as metrics.null_draws)
    def hist(lengths) -> dict[str, int]:
        c = Counter(_len_bin(int(x)) for x in lengths)
        return {b: c.get(b, 0) for b in LEN_BINS}

    obs = hist(ch["length"])
    rng = np.random.default_rng(seed)
    draws = []
    for d in range(n_null):
        h = hist(graph.chains(sents, permute_msets(msets, sents, rng))["length"])
        draws += [{"article_id": aid, "draw": d, "bin": b, "count": h[b]} for b in LEN_BINS]
    dr = pd.DataFrame(draws)
    tot = sum(obs.values())
    # each draw is normalised by its own chain count: the null breaks more links, so it has more chains
    null_share = dr.pivot(index="draw", columns="bin", values="count")
    null_share = null_share.div(null_share.sum(axis=1), axis=0)
    rows = []
    for b in LEN_BINS:
        nb = null_share[b]
        rows.append({"article_id": aid, "bin": b, "n_obs": obs[b], "share_obs": obs[b] / tot,
                     "share_null": nb.mean(), "null_lo": nb.quantile(0.025), "null_hi": nb.quantile(0.975)})
    out["lengths"] = pd.DataFrame(rows)
    out["null_draws"] = dr

    # 4 -- entities: sentences mentioning each one, and how many of those it links to another sentence
    mentioned: dict[str, set] = {}
    for r in m.drop_duplicates(["comment_index", "sent_id", "entity_id"]).itertuples():
        mentioned.setdefault(r.entity_id, set()).add((r.comment_index, r.sent_id))
    linked: dict[str, set] = {}
    for r in ed.itertuples():
        for e in str(r.shared_entities).split(graph.JOIN):
            linked.setdefault(e, set()).update({(r.comment_index, r.sent_a), (r.comment_index, r.sent_b)})
    ent_rows = []
    for e, ss in mentioned.items():
        g = meta.get(e, {})
        by = Counter(_cat(cat.get(s, OTHER)) for s in ss)
        ent_rows.append({"article_id": aid, "entity_id": e, "canonical": g.get("canonical", e),
                         "type": g.get("type", "entity"), "prompt_echo": bool(g.get("prompt_echo", False)),
                         "n_sentences": len(ss), "n_linked": len(ss & linked.get(e, set())),
                         "n_comments": len({c for c, _ in ss}),
                         **{f"n_{c}": by.get(c, 0) for c in common.CATEGORIES}})
    ent = pd.DataFrame(ent_rows)
    ent["link_rate"] = ent["n_linked"] / ent["n_sentences"]
    out["entities"] = ent

    # 5 -- sparsity: mentions per sentence and vocabulary breadth, per category
    sp = []
    per_sent = {k: len(v) for k, v in msets.items()}
    for c in common.CATEGORIES:
        keys = [k for k, v in cat.items() if v == c]
        if not keys:
            continue
        vocab = set().union(*(msets.get(k, set()) for k in keys))
        sp.append({"article_id": aid, "category": c, "n_sentences": len(keys),
                   "mentions_per_sentence": float(np.mean([per_sent.get(k, 0) for k in keys])),
                   "share_with_entity": float(np.mean([per_sent.get(k, 0) > 0 for k in keys])),
                   "distinct_entities": len(vocab), "dictionary_share": len(vocab) / len(groups)})
    out["sparsity"] = pd.DataFrame(sp)

    # 7 -- richness: how many distinct path types, how evenly used; transitions for the fingerprint
    counts = np.array([k for k in types.values()], dtype=float)
    p = counts / counts.sum() if counts.sum() else counts
    ent_bits = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    n_paths = int(counts.sum())
    trans = Counter()
    for r in ch[ch["length"] >= 2].itertuples():
        cs = [_cat(cat[(r.comment_index, int(s))]) for s in str(r.sent_ids).split(graph.JOIN)]
        for a, b in zip(cs, cs[1:]):
            trans[(a, b)] += 1
    out["transitions"] = pd.DataFrame([{"article_id": aid, "from_cat": a, "to_cat": b, "n": k}
                                       for (a, b), k in trans.items()])
    # entropy of a sample of path types grows with the sample, so articles are also compared at a
    # common size: mean entropy over RAREFY_DRAWS random subsamples of RAREFY_N paths
    labels = np.repeat(np.arange(len(counts)), counts.astype(int))
    rare = float("nan")
    if len(labels) >= RAREFY_N:
        rr = np.random.default_rng(seed)
        hs = []
        for _ in range(RAREFY_DRAWS):
            q = np.bincount(rr.choice(labels, RAREFY_N, replace=False)) / RAREFY_N
            q = q[q > 0]
            hs.append(float(-(q * np.log2(q)).sum()))
        rare = float(np.mean(hs))
    n_comments = sents["comment_index"].nunique()
    out["richness"] = pd.DataFrame([{
        "article_id": aid, "n_comments": n_comments, "n_statements": len(sents),
        "statements_per_comment": len(sents) / n_comments, "n_paths": n_paths,
        "paths_per_comment": n_paths / n_comments, "n_path_types": len(types),
        "path_entropy": ent_bits, f"path_entropy_at_{RAREFY_N}": rare,
        # entropy of a path-type sample grows with its size; the normalised form divides by log2(types)
        "path_evenness": ent_bits / math.log2(len(types)) if len(types) > 1 else float("nan"),
        "mean_length": float(ch["length"].mean()), "share_len1": float((ch["length"] == 1).mean())}])
    return out


def _one(aid: str):
    try:
        return aid, article_tables(aid)
    except FileNotFoundError as e:
        return aid, e


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _exemplars(cands: pd.DataFrame, types: pd.DataFrame, k_types: int = 6, per: int = 2,
               max_chars: int = 150, seed: int = 0) -> pd.DataFrame:
    """Two real paths for each of the k most common multi-statement path types (no 'other'), from
    different articles, avoiding prompt-echo links and long sentences so the gallery reads."""
    ranked = (types[~types["path"].str.contains(OTHER)].groupby("path")["n"].sum()
              .sort_values(ascending=False).index.tolist())
    mixed = [p for p in ranked if len(set(p.split(ARROW))) > 1]
    same = [p for p in ranked if len(set(p.split(ARROW))) == 1]
    # interaction is the point, so most exemplars cross categories; two show the common same-category runs
    top = same[:2] + mixed[:k_types - 2]
    rng = np.random.default_rng(seed)
    n_links = cands["links"].map(lambda j: max(len(x) for x in json.loads(j)))
    pool = cands[cands["path"].isin(top) & ~cands["echo"] & (n_links <= 2)].copy()
    pool = pool.iloc[rng.permutation(len(pool))]
    texts: dict[str, pd.DataFrame] = {}
    rows, ex_id = [], 0
    for path in top:
        used_articles: set = set()
        for r in pool[pool["path"] == path].itertuples():
            if len(used_articles) == per:
                break
            if r.article_id in used_articles:
                continue
            if r.article_id not in texts:
                texts[r.article_id] = common.sentences(r.article_id).set_index(["comment_index", "sent_id"])
            tx = texts[r.article_id]
            ids = [int(s) for s in str(r.sent_ids).split(graph.JOIN)]
            sent_text = [str(tx.loc[(r.comment_index, s), "text"]) for s in ids]
            if any(len(t) > max_chars for t in sent_text):
                continue
            links = json.loads(r.links)
            for pos, (s, t) in enumerate(zip(ids, sent_text)):
                rows.append({"exemplar": ex_id, "path": path, "article_id": r.article_id,
                             "comment_index": r.comment_index, "position": pos, "sent_id": s,
                             "category": path.split(ARROW)[pos], "text": t,
                             "link_to_next": ", ".join(links[pos]) if pos < len(links) else ""})
            used_articles.add(r.article_id)
            ex_id += 1
    return pd.DataFrame(rows)


def build(jobs: int = 8, articles=None) -> dict[str, pd.DataFrame]:
    articles = articles or [a for a in common.coded_articles() if (common.OUT_DIR / a / "chains.csv").exists()]
    articles = sorted(articles, key=lambda a: -(common.OUT_DIR / a / "chains.csv").stat().st_size)
    parts: dict[str, list] = {}
    with Pool(jobs) as pool:
        for aid, res in pool.imap_unordered(_one, articles):
            if isinstance(res, Exception):
                print(f"skip {aid}: {res}", file=sys.stderr)
                continue
            for k, df in res.items():
                parts.setdefault(k, []).append(df)
    t = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}

    cx = pd.read_csv(common.OUT_DIR / "cross_article_summary.csv", dtype={"article_id": str})
    keep = {"prompt_variant": "prompt_variant", "graph_mean_cycle_rank__all": "mean_cycle_rank",
            "graph_mean_cycle_rank_z__all": "cycle_rank_z", "graph_mean_density_z__all": "density_z"}
    t["richness"] = t["richness"].merge(cx[["article_id", *keep]].rename(columns=keep), on="article_id", how="left")
    variant = dict(zip(cx["article_id"], cx["prompt_variant"]))
    for k in ("lengths", "sparsity", "entities"):
        t[k]["prompt_variant"] = t[k]["article_id"].map(variant)

    pooled = t["null_draws"].groupby(["draw", "bin"])["count"].sum().unstack()
    pooled = pooled.div(pooled.sum(axis=1), axis=0)
    obs = t["lengths"].groupby("bin")["n_obs"].sum()
    obs = obs / obs.sum()
    t["lengths_pooled"] = pd.DataFrame({
        "bin": LEN_BINS, "share_obs": [obs[b] for b in LEN_BINS],
        "share_null": [pooled[b].mean() for b in LEN_BINS],
        "null_lo": [pooled[b].quantile(0.025) for b in LEN_BINS],
        "null_hi": [pooled[b].quantile(0.975) for b in LEN_BINS]})
    t["exemplars"] = _exemplars(t["cands"], t["types"])
    return t


FILES = {"steps": "path_steps", "types": "path_types", "exemplars": "path_exemplars",
         "lengths": "path_lengths", "lengths_pooled": "path_lengths_null_pooled",
         "entities": "entity_links", "sparsity": "entity_sparsity", "richness": "richness",
         "transitions": "transitions"}


def write(t: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for k, name in FILES.items():
        t[k].to_csv(FIG_DIR / f"{name}.csv", index=False)


def read() -> dict[str, pd.DataFrame] | None:
    try:
        return {k: pd.read_csv(FIG_DIR / f"{name}.csv", dtype={"article_id": str}) for k, name in FILES.items()}
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Charts (Altair). Each takes the tables above, pooled or filtered to one article.
# ---------------------------------------------------------------------------

def _bezier(x0, y0, x1, y1, n=16):
    t = np.linspace(0, 1, n)
    mx = (x0 + x1) / 2
    xs = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * mx + 3 * (1 - t) * t ** 2 * mx + t ** 3 * x1
    ys = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t ** 2 * y1 + t ** 3 * y1
    return xs, ys


def flow_layout(steps: pd.DataFrame, gap: float = 0.04):
    """Sankey layout of path steps: one column per statement position, one node per category,
    one band per (from, to, linking type). Heights are shares of the column's flow."""
    s = steps.groupby(["step", "from_cat", "to_cat", "link_type"], as_index=False)["weight"].sum()
    cols = sorted(s["step"].unique())
    order = {c: i for i, c in enumerate(CATS)}
    tidx = {t: i for i, t in enumerate(TYPES)}
    col_total = {k: s.loc[s["step"] == k, "weight"].sum() for k in cols}
    scale = 1.0 / max(col_total.values())
    nodes, bands = [], []
    # node extents at column position x: x=k is the source side of step k, x=k+1 the target side
    ext: dict[tuple[int, str], list] = {}
    for x in range(int(min(cols)), int(max(cols)) + 2):
        out_w = s[s["step"] == x].groupby("from_cat")["weight"].sum()
        in_w = s[s["step"] == x - 1].groupby("to_cat")["weight"].sum()
        y = 0.0
        for c in CATS:
            h = max(out_w.get(c, 0), in_w.get(c, 0)) * scale
            if h <= 0:
                continue
            ext[(x, c)] = [y, y, y]                  # top, next outgoing slot, next incoming slot
            nodes.append({"x": x, "category": CAT_NAME[c], "y0": y, "y1": y + h,
                          "flow": max(out_w.get(c, 0), in_w.get(c, 0))})
            y += h + gap
    s = s.assign(fo=s["from_cat"].map(order), to=s["to_cat"].map(order), lt=s["link_type"].map(tidx))
    band_id = 0
    for r in s.sort_values(["step", "fo", "to", "lt"]).itertuples():
        h = r.weight * scale
        src, dst = ext[(r.step, r.from_cat)], ext[(r.step + 1, r.to_cat)]
        ya, yb = src[1] + h / 2, dst[2] + h / 2
        src[1] += h
        dst[2] += h
        xs, ys = _bezier(r.step + 0.04, ya, r.step + 1 - 0.04, yb)
        for i, (xx, yy) in enumerate(zip(xs, ys)):
            bands.append({"band": band_id, "i": i, "x": xx, "y": yy - h / 2, "y2": yy + h / 2, "h": h,
                          "from": CAT_NAME[r.from_cat], "to": CAT_NAME[r.to_cat],
                          "link": TYPE_NAME.get(r.link_type, r.link_type), "weight": r.weight,
                          "step": f"{int(r.step)}{ARROW}{int(r.step) + 1}"})
        band_id += 1
    return pd.DataFrame(nodes), pd.DataFrame(bands)


def flow_chart(steps: pd.DataFrame, width: int = 760, height: int = 380) -> alt.LayerChart:
    nodes, bands = flow_layout(steps)
    ymax = nodes["y1"].max()
    y = alt.Y("y:Q", scale=alt.Scale(domain=[ymax, 0]), axis=None)
    x = alt.X("x:Q", axis=alt.Axis(title="statement position in the path (paths cut at 5)", tickMinStep=1),
              scale=alt.Scale(domain=[nodes["x"].min() - 0.15, nodes["x"].max() + 0.15]))
    band = alt.Chart(bands).mark_area(opacity=0.55, interpolate="linear").encode(
        x=x, y=alt.Y("y:Q", stack=None, scale=alt.Scale(domain=[ymax, 0]), axis=None), y2="y2:Q",
        detail="band:N",
        color=alt.Color("link:N", scale=_tscale(), title="step carried by"),
        tooltip=["step:N", "from:N", "to:N", "link:N", alt.Tooltip("weight:Q", format=",.0f", title="paths")])
    node = alt.Chart(nodes).mark_rect(width=12).encode(
        x=x, y=alt.Y("y0:Q", scale=alt.Scale(domain=[ymax, 0]), axis=None), y2="y1:Q",
        color=alt.Color("category:N", scale=_cscale(), title="statement"),
        tooltip=["category:N", alt.Tooltip("flow:Q", format=",.0f")])
    lab = alt.Chart(nodes[nodes["x"] == nodes["x"].min()]).mark_text(align="right", dx=-10, fontSize=11).encode(
        x=x, y=alt.Y("mid:Q", scale=alt.Scale(domain=[ymax, 0]), axis=None), text="category:N"
    ).transform_calculate(mid="(datum.y0 + datum.y1) / 2")
    return alt.layer(band, node, lab).resolve_scale(color="independent").properties(
        width=width, height=height, title="Common paths: how reading-order chains move between categories")


def path_types_chart(types: pd.DataFrame, top: int = 15) -> alt.Chart:
    g = types.groupby(["path", "length"], as_index=False).agg(n=("n", "sum"), n_articles=("article_id", "nunique"))
    g["share"] = g["n"] / g["n"].sum()
    g = g.sort_values("n", ascending=False).head(top)
    return alt.Chart(g).mark_bar(color="#6b7a8f").encode(
        y=alt.Y("path:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("share:Q", axis=alt.Axis(format="%"), title="share of paths (length ≥ 2)"),
        tooltip=["path", "length", alt.Tooltip("n:Q", format=","), "n_articles",
                 alt.Tooltip("share:Q", format=".1%")]
    ).properties(width=360, height=16 * len(g), title=f"Top {len(g)} path types")


def exemplar_chart(ex: pd.DataFrame, wrap: int = 34) -> alt.VConcatChart:
    import textwrap
    ex = ex.copy()
    ex["link_to_next"] = ex["link_to_next"].fillna("")
    ex["cat_name"] = ex["category"].map(CAT_NAME)
    ex["row"] = ex["exemplar"]
    ex["label"] = ex["text"].map(lambda t: textwrap.wrap(t, wrap)[:4])
    ex["title"] = ex["path"] + "   (article " + ex["article_id"].astype(str) + ", comment " + ex["comment_index"].astype(str) + ")"
    rows = []
    sc = alt.Scale(domain=[-0.4, 3.6])
    for e, sub in ex.groupby("exemplar"):
        sub = sub.sort_values("position")
        seg = sub.assign(x2=sub["position"] + 1, mid=sub["position"] + 0.5)[sub["link_to_next"] != ""]
        X = alt.X("position:Q", axis=None, scale=sc)
        Y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, 1]))
        sub, seg = sub.assign(y=0.8), seg.assign(y=0.8)
        line = alt.Chart(seg).mark_rule(strokeWidth=2, color="#888").encode(x=X, x2="x2:Q", y=Y)
        dots = alt.Chart(sub).mark_circle(size=260, opacity=1).encode(
            x=X, y=Y, color=alt.Color("cat_name:N", scale=_cscale(), title="statement"),
            tooltip=["cat_name", "text"])
        link = alt.Chart(seg).mark_text(dy=-11, fontSize=10, color="#7a4a00", fontStyle="italic").encode(
            x=alt.X("mid:Q", axis=None, scale=sc), y=Y, text="link_to_next:N")
        txt = alt.Chart(sub).mark_text(align="center", baseline="top", fontSize=10, lineHeight=12, dy=14).encode(
            x=X, y=Y, text="label")
        rows.append(alt.layer(line, dots, link, txt).properties(
            width=760, height=84, title=alt.TitleParams(sub["title"].iloc[0], fontSize=11, anchor="start")))
    return alt.vconcat(*rows, spacing=6).properties(title="What paths look like: real reading-order chains")


def length_chart(pooled: pd.DataFrame, title: str = "Chain lengths vs the within-comment null") -> alt.LayerChart:
    base = alt.Chart(pooled).encode(x=alt.X("bin:O", sort=LEN_BINS, title="chain length (statements)"))
    bars = base.mark_bar(color="#6b7a8f", opacity=0.85).encode(
        y=alt.Y("share_obs:Q", axis=alt.Axis(format="%"), title="share of chains"),
        tooltip=[alt.Tooltip("share_obs:Q", format=".1%", title="observed"),
                 alt.Tooltip("share_null:Q", format=".1%", title="null mean"),
                 alt.Tooltip("null_lo:Q", format=".1%"), alt.Tooltip("null_hi:Q", format=".1%")])
    rng = base.mark_rule(color="#c0392b", strokeWidth=3).encode(y="null_lo:Q", y2="null_hi:Q")
    tick = base.mark_tick(color="#c0392b", thickness=2, size=26).encode(y="share_null:Q")
    return alt.layer(bars, rng, tick).properties(width=280, height=260, title=title)


def length_excess_heatmap(lengths: pd.DataFrame) -> alt.Chart:
    d = lengths.copy()
    d["excess"] = d["share_obs"] - d["share_null"]
    mean_len = d.assign(w=d["bin"].map({"1": 1, "2": 2, "3": 3, "4+": 4.5}) * d["share_obs"]).groupby("article_id")["w"].sum()
    order = mean_len.sort_values(ascending=False).index.tolist()
    lim = float(d["excess"].abs().quantile(0.98))
    return alt.Chart(d).mark_rect().encode(
        x=alt.X("bin:O", sort=LEN_BINS, title="chain length"),
        y=alt.Y("article_id:N", sort=order, axis=None, title="articles (sorted by mean chain length)"),
        color=alt.Color("excess:Q", scale=alt.Scale(scheme="redblue", domain=[-lim, lim], reverse=True),
                        title="observed − null", legend=alt.Legend(format="+.0%")),
        tooltip=["article_id", "bin", "prompt_variant", alt.Tooltip("share_obs:Q", format=".1%"),
                 alt.Tooltip("share_null:Q", format=".1%"), alt.Tooltip("excess:Q", format="+.1%")]
    ).properties(width=160, height=380, title="Per article: excess over null")


def entity_chart(ent: pd.DataFrame, label_top: int = 0) -> alt.LayerChart:
    d = ent.copy()
    d["type_name"] = d["type"].map(TYPE_NAME)
    d["echo"] = np.where(d["prompt_echo"], "prompt echo", "reader's own")
    x = alt.X("n_sentences:Q", scale=alt.Scale(type="log", nice=False, domain=[1, float(d["n_sentences"].max())]),
              title="sentences mentioning the entity (log)")
    y = alt.Y("link_rate:Q", axis=alt.Axis(format="%"), title="share of those sentences it links to another")
    pts = alt.Chart(d).mark_point(opacity=0.45, size=24, filled=True).encode(
        x=x, y=y, color=alt.Color("type_name:N", scale=_tscale(), title="entity type"),
        shape=alt.Shape("echo:N", scale=alt.Scale(domain=["reader's own", "prompt echo"], range=["circle", "cross"]),
                        title=None),
        tooltip=["article_id", "canonical", "type_name", "n_sentences", "n_linked",
                 alt.Tooltip("link_rate:Q", format=".0%"), "echo"])
    trend = alt.Chart(d).transform_loess("n_sentences", "link_rate", groupby=["type_name"], bandwidth=0.5).mark_line(
        strokeWidth=2.5).encode(x=x, y=y, color=alt.Color("type_name:N", scale=_tscale()))
    layers = [pts, trend]
    if label_top:
        top = d.nlargest(label_top, "n_sentences")
        layers.append(alt.Chart(top).mark_text(dx=6, align="left", fontSize=10).encode(x=x, y=y, text="canonical"))
    return alt.layer(*layers).properties(width=460, height=320,
                                         title="Connectors vs dead ends: which entities link statements")


def entity_type_chart(ent: pd.DataFrame) -> alt.Chart:
    """Per article and entity type, the share of mentioning sentences that are linked (pooled)."""
    g = ent[~ent["prompt_echo"]].groupby(["article_id", "type"], as_index=False)[["n_linked", "n_sentences"]].sum()
    g = g[g["n_sentences"] >= 20]
    g["link_rate"] = g["n_linked"] / g["n_sentences"]
    g["type_name"] = g["type"].map(TYPE_NAME)
    return alt.Chart(g).mark_boxplot(extent=1.5, size=26).encode(
        x=alt.X("type_name:N", title=None, sort=[TYPE_NAME[t] for t in TYPES]),
        y=alt.Y("link_rate:Q", axis=alt.Axis(format="%"), title="linked share (per article)"),
        color=alt.Color("type_name:N", scale=_tscale(), legend=None)
    ).properties(width=240, height=320, title="By entity type (echo removed)")


def sparsity_chart(sp: pd.DataFrame, field: str = "mentions_per_sentence",
                   title: str = "mentions per sentence") -> alt.LayerChart:
    d = sp.copy()
    d["cat_name"] = d["category"].map(CAT_NAME)
    piv = d.pivot(index="article_id", columns="category", values=field)
    order = (piv["VO"] - piv["INF"]).sort_values(ascending=False).index.tolist() if {"VO", "INF"} <= set(piv) else None
    y = alt.Y("article_id:N", sort=order, axis=None, title="articles")
    rule = alt.Chart(d).mark_rule(color="#ccc").encode(
        y=y, x=alt.X(f"min({field}):Q"), x2=f"max({field}):Q")
    pts = alt.Chart(d).mark_circle(size=22, opacity=0.9).encode(
        y=y, x=alt.X(f"{field}:Q", title=title),
        color=alt.Color("cat_name:N", scale=_cscale(["VO", "INF", "PK"]), title="statement"),
        tooltip=["article_id", "cat_name", "prompt_variant", alt.Tooltip(f"{field}:Q", format=".2f"), "n_sentences"])
    return alt.layer(rule, pts).properties(width=300, height=max(200, 3 * d["article_id"].nunique()),
                                           title=f"Sparsity by category: {title}")


def sparsity_summary_chart(sp: pd.DataFrame) -> alt.Chart:
    d = sp.copy()
    d["cat_name"] = d["category"].map(CAT_NAME)
    return alt.Chart(d).mark_boxplot(size=28).encode(
        x=alt.X("cat_name:N", sort=[CAT_NAME[c] for c in ["VO", "INF", "PK"]], title=None),
        y=alt.Y("mentions_per_sentence:Q", title="mentions per sentence (per article)"),
        color=alt.Color("cat_name:N", scale=_cscale(["VO", "INF", "PK"]), legend=None),
        column=alt.Column("prompt_variant:N", title=None)
    ).properties(width=160, height=260)


def richness_chart(rich: pd.DataFrame, highlight: str | None = None) -> alt.LayerChart:
    d = rich.copy()
    d = d[d[f"path_entropy_at_{RAREFY_N}"].notna()]
    x = alt.X("paths_per_comment:Q", title="paths per comment", scale=alt.Scale(zero=False))
    y = alt.Y(f"path_entropy_at_{RAREFY_N}:Q", title=f"path-type entropy at {RAREFY_N} paths (bits)",
              scale=alt.Scale(zero=False))
    pts = alt.Chart(d).mark_circle(size=60, opacity=0.75).encode(
        x=x, y=y, color=alt.Color("prompt_variant:N", scale=alt.Scale(scheme="set2"), title="prompt"),
        size=alt.Size("n_comments:Q", scale=alt.Scale(range=[20, 400]), title="comments"),
        tooltip=["article_id", "prompt_variant", "n_comments", "n_path_types",
                 alt.Tooltip(f"path_entropy_at_{RAREFY_N}:Q", format=".2f"), alt.Tooltip("path_entropy:Q", format=".2f"),
                 alt.Tooltip("paths_per_comment:Q", format=".2f"), alt.Tooltip("cycle_rank_z:Q", format=".1f")])
    layers = [pts]
    if highlight is not None:
        layers.append(alt.Chart(d[d["article_id"] == highlight]).mark_circle(
            size=300, fill=None, stroke="black", strokeWidth=2).encode(x=x, y=y))
    return alt.layer(*layers).properties(width=420, height=320,
                                         title="Richness: how many distinct kinds of path an article's readers build")


def transition_multiples(trans: pd.DataFrame, rich: pd.DataFrame, n: int = 4) -> alt.Chart:
    key = f"path_entropy_at_{RAREFY_N}"
    r = rich[rich[key].notna()].sort_values(key)
    pick = list(r["article_id"].head(n)) + list(r["article_id"].tail(n))
    d = trans[trans["article_id"].isin(pick)].copy()
    d["share"] = d["n"] / d.groupby("article_id")["n"].transform("sum")
    d["from"] = d["from_cat"].map(CAT_NAME)
    d["to"] = d["to_cat"].map(CAT_NAME)
    ent = dict(zip(r["article_id"], r[key]))
    d["panel"] = d["article_id"].map(lambda a: f"{a}  (H={ent[a]:.1f})")
    order = [f"{a}  (H={ent[a]:.1f})" for a in pick]
    names = [CAT_NAME[c] for c in CATS]
    return alt.Chart(d).mark_rect().encode(
        x=alt.X("to:N", sort=names, title="to"), y=alt.Y("from:N", sort=names, title="from"),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(format="%")),
        tooltip=["article_id", "from", "to", "n", alt.Tooltip("share:Q", format=".1%")],
        facet=alt.Facet("panel:N", sort=order, columns=n, title=f"Least rich (top row) vs most rich (bottom row) articles")
    ).properties(width=110, height=110)


def all_charts(t: dict[str, pd.DataFrame]) -> dict[str, alt.TopLevelMixin]:
    return {
        "1_path_flow": flow_chart(t["steps"]),
        "1_path_types": path_types_chart(t["types"]),
        "2_path_exemplars": exemplar_chart(t["exemplars"]),
        "3_path_lengths": alt.hconcat(length_chart(t["lengths_pooled"]), length_excess_heatmap(t["lengths"])),
        "4_entity_links": alt.hconcat(entity_chart(t["entities"]), entity_type_chart(t["entities"])
                                      ).resolve_scale(color="independent"),
        "5_entity_sparsity": alt.hconcat(sparsity_chart(t["sparsity"]),
                                         sparsity_chart(t["sparsity"], "dictionary_share", "share of the dictionary used"),
                                         sparsity_summary_chart(t["sparsity"])),
        "7_richness": alt.vconcat(richness_chart(t["richness"]), transition_multiples(t["transitions"], t["richness"])),
    }


def save_charts(t: dict[str, pd.DataFrame]) -> None:
    import vl_convert as vlc
    alt.data_transformers.disable_max_rows()
    d = FIG_DIR / "altair"
    d.mkdir(parents=True, exist_ok=True)
    for name, ch in all_charts(t).items():
        ch = ch.configure_view(stroke=None).configure_axis(grid=False)
        ch.save(d / f"{name}.html")
        (d / f"{name}.png").write_bytes(vlc.vegalite_to_png(ch.to_json(), scale=2))
        print("wrote", d / f"{name}.png")


if __name__ == "__main__":
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t = build(jobs)
    write(t)
    save_charts(read())
    from analysis.claim2_2 import gofish
    gofish.write_all(read())
