"""Distributions, bridging, sparsity and the paths-to-one-insight table for claim 2.2.

Amendments A6.1-A6.4 (see the task-6 brief) add three things on top of the base
interfaces:

- `apply_view` / a `view` keyword threaded through `signature_table`,
  `length_table`, `component_table` and `bridging` so every headline metric can be
  reported under three filters: "all", "minus_echo" (drop mentions of groups
  flagged `prompt_echo`), and "minus_boilerplate" (drop boilerplate sentences).
- `sparsity` gains a `sufficient` column (>= 30 sentences in that category), and
  `sparsity_by_variant` aggregates per-article sparsity frames across the
  community-asked / community-not-asked prompt split, bootstrapping over articles.
- `paths_to_insight` falls back to keying by the chain's terminal INF sentence
  when claim 3's symbolic statement ids do not exist for the article (they only
  exist for the three pilots), instead of silently dropping the article.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.claim2_2 import common, entities, graph
from analysis.claim2_2.graph import JOIN

SUFFICIENT_N = 30
VIEWS = ("all", "minus_echo", "minus_boilerplate")


def boot_ci(values, stat=np.mean, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = [float(stat(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    return float(stat(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def apply_view(m: pd.DataFrame, sents: pd.DataFrame, groups: list[dict],
                view: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter mentions/sentences for one of the three headline views.

    "all"               every sentence, every entity group (the naive view)
    "minus_echo"        drop mentions of groups flagged prompt_echo
    "minus_boilerplate" drop sentences where boilerplate is True (mentions follow)
    """
    if view == "all":
        return m, sents
    if view == "minus_echo":
        echo_ids = {g["id"] for g in groups if g.get("prompt_echo")}
        m2 = m[~m["entity_id"].isin(echo_ids)].reset_index(drop=True)
        return m2, sents
    if view == "minus_boilerplate":
        if "boilerplate" not in sents.columns:
            raise ValueError("sents needs a boilerplate column for the minus_boilerplate view")
        sents2 = sents[~sents["boilerplate"].astype(bool)].reset_index(drop=True)
        if "key" in m.columns and "key" in sents2.columns:
            keep = set(sents2["key"])
            m2 = m[m["key"].isin(keep)].reset_index(drop=True)
        else:
            keep = set(zip(sents2["comment_index"].astype(int), sents2["sent_id"].astype(int)))
            m2 = m[[(int(ci), int(sid)) in keep
                    for ci, sid in zip(m["comment_index"], m["sent_id"])]].reset_index(drop=True)
        return m2, sents2
    raise ValueError(f"unknown view: {view!r}")


def _recompute_chains(m: pd.DataFrame, sents: pd.DataFrame, groups: list[dict],
                       view: str) -> pd.DataFrame:
    m, sents = apply_view(m, sents, groups, view)
    msets = graph.mention_sets(m)
    return graph.chains(sents, msets)


def _recompute_edges(m: pd.DataFrame, sents: pd.DataFrame, groups: list[dict],
                      view: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m, sents = apply_view(m, sents, groups, view)
    msets = graph.mention_sets(m)
    return m, sents, graph.edges(sents, msets)


def _require(view: str, **kwargs):
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"view={view!r} requires {', '.join(missing)}")


def signature_table(ch: pd.DataFrame, *, m: pd.DataFrame | None = None,
                     sents: pd.DataFrame | None = None, groups: list[dict] | None = None,
                     view: str = "all") -> pd.DataFrame:
    if view != "all":
        _require(view, m=m, sents=sents, groups=groups)
        ch = _recompute_chains(m, sents, groups, view)
    g = ch.groupby("signature").agg(n_chains=("signature", "size"),
                                    n_comments=("comment_index", "nunique"),
                                    mean_length=("length", "mean")).reset_index()
    return g.sort_values("n_chains", ascending=False).reset_index(drop=True)


def length_table(ch: pd.DataFrame, *, m: pd.DataFrame | None = None,
                  sents: pd.DataFrame | None = None, groups: list[dict] | None = None,
                  view: str = "all") -> pd.DataFrame:
    if view != "all":
        _require(view, m=m, sents=sents, groups=groups)
        ch = _recompute_chains(m, sents, groups, view)
    g = ch.groupby("length").size().rename("n_chains").reset_index()
    g["share"] = g["n_chains"] / g["n_chains"].sum()
    return g.sort_values("length").reset_index(drop=True)


def component_table(comp: pd.DataFrame, *, m: pd.DataFrame | None = None,
                     sents: pd.DataFrame | None = None, groups: list[dict] | None = None,
                     view: str = "all") -> pd.DataFrame:
    if view != "all":
        _require(view, m=m, sents=sents, groups=groups)
        _, sents2, e = _recompute_edges(m, sents, groups, view)
        comp = graph.components(sents2, e)
    g = comp.groupby("size").size().rename("n_components").reset_index()
    g["share"] = g["n_components"] / g["n_components"].sum()
    return g.sort_values("size").reset_index(drop=True)


def bridging(m: pd.DataFrame, edges_df: pd.DataFrame, groups: list[dict], *,
             sents: pd.DataFrame | None = None, view: str = "all") -> pd.DataFrame:
    if view != "all":
        _require(view, sents=sents)
        m, sents2, edges_df = _recompute_edges(m, sents, groups, view)
    meta = {g["id"]: g for g in groups}
    n_edges: dict[str, int] = {}
    pairs: dict[str, set[str]] = {}
    cats_by_sent = m.groupby(["comment_index", "sent_id"])["category"].first().to_dict()
    for r in edges_df.itertuples():
        ca = cats_by_sent.get((int(r.comment_index), int(r.sent_a)), common.REDUNDANT)
        cb = cats_by_sent.get((int(r.comment_index), int(r.sent_b)), common.REDUNDANT)
        pair = "-".join(sorted([ca, cb]))
        for eid in str(r.shared_entities).split(JOIN):
            n_edges[eid] = n_edges.get(eid, 0) + 1
            pairs.setdefault(eid, set()).add(pair)
    per = m.groupby("entity_id").agg(n_sentences=("key", "nunique"),
                                     n_comments=("comment_index", "nunique")).reset_index()
    per["canonical"] = [meta.get(e, {}).get("canonical", e) for e in per["entity_id"]]
    per["type"] = [meta.get(e, {}).get("type", "entity") for e in per["entity_id"]]
    per["n_edges"] = [n_edges.get(e, 0) for e in per["entity_id"]]
    per["category_pairs"] = [JOIN.join(sorted(pairs.get(e, set()))) for e in per["entity_id"]]
    per["bridges"] = per["n_edges"] > 0
    cols = ["entity_id", "canonical", "type", "n_sentences", "n_comments",
            "n_edges", "category_pairs", "bridges"]
    return per[cols].sort_values("n_edges", ascending=False).reset_index(drop=True)


def _per_comment_rates(m: pd.DataFrame, sents: pd.DataFrame, cat: str):
    """(mentions per sentence, distinct entities per sentence) for each comment, in one category."""
    s = sents[sents["category"] == cat]
    mm = m[m["category"] == cat]
    rows_m, rows_d = [], []
    for ci, sub in s.groupby("comment_index"):
        sel = mm[mm["comment_index"] == ci]
        rows_m.append(len(sel) / len(sub))
        rows_d.append(sel.groupby("sent_id")["entity_id"].nunique().reindex(sub["sent_id"]).fillna(0).mean())
    return rows_m, rows_d


def sparsity(m: pd.DataFrame, sents: pd.DataFrame, edges_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for cat in common.CATEGORIES:
        s = sents[sents["category"] == cat]
        mm = m[m["category"] == cat]
        rates_m, rates_d = _per_comment_rates(m, sents, cat)
        mps = len(mm) / len(s) if len(s) else 0.0
        dps = (mm.groupby(["comment_index", "sent_id"])["entity_id"].nunique().sum() / len(s)) if len(s) else 0.0
        _, mlo, mhi = boot_ci(rates_m)
        _, dlo, dhi = boot_ci(rates_d)
        density = np.nan
        if edges_df is not None and len(s):
            keep = set(zip(s["comment_index"].astype(int), s["sent_id"].astype(int)))
            e = sum(1 for r in edges_df.itertuples()
                    if (int(r.comment_index), int(r.sent_a)) in keep
                    and (int(r.comment_index), int(r.sent_b)) in keep)
            possible = sum(len(g) * (len(g) - 1) / 2 for _, g in s.groupby("comment_index"))
            density = e / possible if possible else 0.0
        rows.append({"category": cat, "n_sentences": len(s),
                     "mentions_per_sentence": mps, "mps_lo": mlo, "mps_hi": mhi,
                     "distinct_per_sentence": dps, "dps_lo": dlo, "dps_hi": dhi,
                     "edge_density": density, "sufficient": len(s) >= SUFFICIENT_N})
    return pd.DataFrame(rows)


def sparsity_by_variant(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (prompt_variant, category): mentions and distinct entities per sentence, edge density,
    and the bootstrap CI of each, over ARTICLES rather than over comments.

    `frames` maps article_id -> that article's `sparsity()` output (one row per category,
    with a `sufficient` column). Only sufficient rows (>= SUFFICIENT_N sentences in that
    category) contribute to the per-article distributions and their bootstrap CIs, per A6.3;
    `n_articles` counts all rows seen for that (variant, category), `n_sufficient` counts the
    ones actually used.
    """
    long_rows = []
    for article_id, df in frames.items():
        variant = common.prompt_variant(article_id)
        for r in df.itertuples():
            long_rows.append({
                "article_id": article_id, "prompt_variant": variant, "category": r.category,
                "n_sentences": r.n_sentences,
                "sufficient": bool(getattr(r, "sufficient", r.n_sentences >= SUFFICIENT_N)),
                "mentions_per_sentence": r.mentions_per_sentence,
                "distinct_per_sentence": r.distinct_per_sentence,
                "edge_density": r.edge_density,
            })
    long = pd.DataFrame(long_rows)
    out = []
    if len(long):
        for (variant, cat), sub in long.groupby(["prompt_variant", "category"]):
            suf = sub[sub["sufficient"]]
            mps_pt, mps_lo, mps_hi = boot_ci(suf["mentions_per_sentence"])
            dps_pt, dps_lo, dps_hi = boot_ci(suf["distinct_per_sentence"])
            dens_vals = suf["edge_density"].dropna()
            dens_pt, dens_lo, dens_hi = boot_ci(dens_vals)
            out.append({"prompt_variant": variant, "category": cat,
                        "n_articles": len(sub), "n_sufficient": len(suf),
                        "mentions_per_sentence": mps_pt, "mps_lo": mps_lo, "mps_hi": mps_hi,
                        "distinct_per_sentence": dps_pt, "dps_lo": dps_lo, "dps_hi": dps_hi,
                        "edge_density": dens_pt, "edge_density_lo": dens_lo, "edge_density_hi": dens_hi})
    cols = ["prompt_variant", "category", "n_articles", "n_sufficient",
            "mentions_per_sentence", "mps_lo", "mps_hi",
            "distinct_per_sentence", "dps_lo", "dps_hi",
            "edge_density", "edge_density_lo", "edge_density_hi"]
    return pd.DataFrame(out, columns=cols).sort_values(["prompt_variant", "category"]).reset_index(drop=True)


_PATH_COLS = ["sym_id", "sym_label", "n_chains", "n_signatures", "n_entity_sets", "keyed_by"]


def paths_to_insight(article_id: str, ch: pd.DataFrame, sents: pd.DataFrame) -> pd.DataFrame:
    """Per distinct terminal insight: how many chains, signatures and entity sets reach it.

    Claim 3's symbolic statement ids exist only for the three pilot articles (181, 183, 91).
    When they exist, paths are keyed by statement id (`keyed_by == "statement"`). Otherwise
    (A6.4) the article is not dropped: paths are keyed by the chain's terminal INF sentence
    instead (`keyed_by == "sentence"`), and `sym_id` is reported as None throughout since no
    statement id exists -- `sym_label` (the sentence text) is what distinguishes rows.
    """
    from analysis.claim3 import common as c3common
    from analysis.claim3 import statements as c3s

    csv_path = c3common.OUT_DIR / str(article_id) / "statements.csv"
    if csv_path.exists():
        st = c3s.read(str(article_id))
        sym = {(int(r.comment_index), int(r.sent_id)): (r.sym_id, r.sym_label)
               for r in st.itertuples() if r.category == "INF"}
        keyed_by = "statement"
    else:
        inf = sents[sents["category"] == "INF"]
        sym = {(int(r.comment_index), int(r.sent_id)): (r.key, r.text) for r in inf.itertuples()}
        keyed_by = "sentence"

    rows = []
    for r in ch.itertuples():
        if not r.has_inf:
            continue
        ids = [int(x) for x in str(r.sent_ids).split(JOIN)]
        last_inf = None
        for s in ids:
            if (int(r.comment_index), s) in sym:
                last_inf = sym[(int(r.comment_index), s)]
        if last_inf is None:
            continue
        rows.append({"group_key": last_inf[0], "sym_label": last_inf[1],
                     "signature": r.signature, "entities": r.entities,
                     "comment_index": int(r.comment_index)})
    if not rows:
        return pd.DataFrame(columns=_PATH_COLS)
    df = pd.DataFrame(rows)
    g = df.groupby(["group_key", "sym_label"]).agg(n_chains=("signature", "size"),
                                                    n_signatures=("signature", "nunique"),
                                                    n_entity_sets=("entities", "nunique")).reset_index()
    g["sym_id"] = g["group_key"] if keyed_by == "statement" else None
    g["keyed_by"] = keyed_by
    return g[_PATH_COLS].sort_values("n_chains", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permutation null (Task 7) and the per-article / cross-article writers
# ---------------------------------------------------------------------------


_PERMUTE_RETRIES = 500


def permute_msets(msets: dict, sents: pd.DataFrame, rng) -> dict:
    """Reassign mention sets among a comment's sentences, preserving each sentence's
    mention count and the comment's mention multiset.

    Naively slicing a shuffled flat pool of entity ids and wrapping each slice in
    `set(...)` can silently collapse a sentence's count: if the same entity value
    (contributed by two different original sentences, e.g. an entity that links two
    sentences into a chain) lands twice in one target slice, `set()` dedupes it and
    that sentence ends up with fewer distinct entities than its count requires. We
    reshuffle and retry until every sentence's slice has no internal duplicate; for
    comments where a value's multiplicity cannot be spread across enough distinct
    sentences (pathological, no such case observed in this corpus), we fall back to
    a deterministic round-robin placement that still respects both invariants.
    """
    out: dict = {}
    for ci, sub in sents.groupby("comment_index"):
        ci = int(ci)
        ids = [int(s) for s in sub["sent_id"]]
        counts = [len(msets.get((ci, s), set())) for s in ids]
        pool = [e for s in ids for e in sorted(msets.get((ci, s), set()))]

        assign = None
        for _ in range(_PERMUTE_RETRIES):
            rng.shuffle(pool)
            pos, candidate, ok = 0, {}, True
            for s, k in zip(ids, counts):
                take = pool[pos:pos + k]
                pos += k
                if len(set(take)) != len(take):
                    ok = False
                    break
                candidate[s] = set(take)
            if ok:
                assign = candidate
                break

        if assign is None:
            # Deterministic fallback: place each pool entry into the next sentence
            # (in round-robin id order) that still needs entries and does not already
            # hold that value, guaranteeing the same two invariants without randomness.
            need = dict(zip(ids, counts))
            candidate = {s: set() for s in ids}
            order = sorted(ids, key=lambda s: -need[s])
            for e in pool:
                for s in order:
                    if need[s] > 0 and e not in candidate[s]:
                        candidate[s].add(e)
                        need[s] -= 1
                        break
            assign = candidate

        for s in ids:
            out[(ci, s)] = assign.get(s, set())
    return out


def null_draws(sents: pd.DataFrame, msets: dict, n: int = 1000, seed: int = 0,
               graphs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Chain statistics and (optionally) statement-graph statistics under the within-comment
    permutation null, both computed on the same draws. The graph measures consume no
    randomness, so the chain half is identical to what this seed gave before they existed."""
    rng = np.random.default_rng(seed)
    chain_rows, graph_rows = [], []
    for d in range(n):
        pm = permute_msets(msets, sents, rng)
        ch = graph.chains(sents, pm)
        chain_rows.append({"draw": d, "mean_length": float(ch["length"].mean()),
                           "share_len1": float((ch["length"] == 1).mean()),
                           "n_chains": int(len(ch))})
        if graphs:
            graph_rows.append({"draw": d, **graph.graph_summary(graph.comment_graph_metrics(sents, pm))})
    return pd.DataFrame(chain_rows), (pd.DataFrame(graph_rows) if graphs else None)


def null_chains(sents: pd.DataFrame, msets: dict, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    return null_draws(sents, msets, n=n, seed=seed, graphs=False)[0]


def z_against_null(observed: float, null_values) -> tuple[float, float]:
    """z-score and add-one-corrected permutation p-value.

    `p = (1 + count) / (1 + n)` rather than `mean(...)`: the raw count-based
    mean returns exactly 0 whenever no draw is as extreme as the observation
    (the common case at n_null=200), which both misstates "no evidence against
    the null was seen" as "p is exactly zero" and gives p a resolution of only
    1/n_null. The add-one correction is the standard fix (Davison & Hinkley
    1997; North, Curtis & Sham 2002) and guarantees 1/(n+1) <= p <= 1.
    """
    v = np.asarray(list(null_values), dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    z = 0.0 if sd == 0 else (observed - float(v.mean())) / sd
    count = int((np.abs(v - v.mean()) >= abs(observed - v.mean())).sum())
    p = (1 + count) / (1 + len(v))
    return z, p


def _view_frames(m: pd.DataFrame, sents: pd.DataFrame, groups: list[dict], view: str):
    if view == "all":
        return m, sents
    return apply_view(m, sents, groups, view)


def _view_metrics(article_id: str, m: pd.DataFrame, sents: pd.DataFrame, groups: list[dict],
                   view: str, n_null: int) -> dict:
    """Every table for one view, including a null re-derived under that same view (A7.4)."""
    mv, sv = _view_frames(m, sents, groups, view)
    msets = graph.mention_sets(mv)
    e = graph.edges(sv, msets)
    ch = graph.chains(sv, msets)
    comp = graph.components(sv, e)
    nd, ng = null_draws(sv, msets, n=n_null)
    return {
        "m": mv, "sents": sv, "e": e, "ch": ch, "comp": comp,
        "sig": signature_table(ch), "lens": length_table(ch),
        "comps": component_table(comp), "br": bridging(mv, e, groups),
        "sp": sparsity(mv, sv, e), "pti": paths_to_insight(article_id, ch, sv),
        "nd": nd, "pc": graph.comment_graph_metrics(sv, msets), "ng": ng,
    }


def _view_headline(vd: dict) -> dict:
    ch, comp, sents, nd = vd["ch"], vd["comp"], vd["sents"], vd["nd"]
    mean_len = float(ch["length"].mean()) if len(ch) else float("nan")
    share1 = float((ch["length"] == 1).mean()) if len(ch) else float("nan")
    zl, pl = z_against_null(mean_len, nd["mean_length"])
    z1, p1 = z_against_null(share1, nd["share_len1"])
    isolated = float((comp["size"] == 1).sum() / len(sents)) if len(sents) else float("nan")
    largest_share = float("nan")
    if len(sents):
        largest = comp.groupby("comment_index")["size"].max()
        per_comment_n = sents.groupby("comment_index").size()
        largest_share = float((largest / per_comment_n).mean())
    return {
        "n_chains": int(len(ch)),
        "mean_chain_length": mean_len,
        "share_len1": share1,
        "isolated_sentence_rate": isolated,
        "largest_component_share": largest_share,
        "null": {"mean_length_z": zl, "mean_length_p": pl,
                 "share_len1_z": z1, "share_len1_p": p1},
        "graph": graph_headline(vd["pc"], vd["ng"]),
    }


def graph_headline(per_comment: pd.DataFrame, null_graphs: pd.DataFrame) -> dict:
    """Each article-level statement-graph number, observed vs the same permutation null."""
    out = {}
    for k, obs in graph.graph_summary(per_comment).items():
        null = null_graphs[k].dropna()
        z, p = z_against_null(obs, null) if len(null) else (float("nan"), float("nan"))
        out[k] = {"observed": obs, "null_mean": float(null.mean()) if len(null) else float("nan"),
                  "z": z, "p": p}
    return out


def write(article_id: str, n_null: int = 1000) -> Path:
    aid = str(article_id)
    sents = common.sentences(aid)
    groups = entities.load(aid)
    m = entities.mentions(aid)
    out = common.out_dir(aid)

    per_view = {}
    for view in VIEWS:
        vd = _view_metrics(aid, m, sents, groups, view, n_null)
        per_view[view] = vd
        suffix = "" if view == "all" else f"__{view}"
        for name, df in [("signatures", vd["sig"]), ("lengths", vd["lens"]),
                         ("components_dist", vd["comps"]), ("bridging", vd["br"]),
                         ("sparsity", vd["sp"]), ("paths_to_insight", vd["pti"]),
                         ("null_chains", vd["nd"]), ("comment_graphs", vd["pc"]),
                         ("null_graphs", vd["ng"])]:
            df.to_csv(out / f"{name}{suffix}.csv", index=False)

    base = per_view["all"]
    ch, e, comp, sig, br, sp, nd = (base["ch"], base["e"], base["comp"], base["sig"],
                                     base["br"], base["sp"], base["nd"])

    mean_len, share1 = float(ch["length"].mean()), float((ch["length"] == 1).mean())
    zl, pl = z_against_null(mean_len, nd["mean_length"])
    z1, p1 = z_against_null(share1, nd["share_len1"])
    largest = comp.groupby("comment_index")["size"].max()
    per_comment_n = sents.groupby("comment_index").size()

    echo_ids = {g["id"] for g in groups if g.get("prompt_echo")}
    boilerplate_share = float(sents["boilerplate"].astype(bool).mean()) if len(sents) else 0.0
    sufficient = {str(r.category): bool(r.sufficient) for r in sp.itertuples()}

    summary = {
        "article_id": aid,
        "n_comments": int(sents["comment_index"].nunique()),
        "n_sentences": int(len(sents)),
        "coverage": entities.coverage(m, sents),
        "n_entities": len(groups),
        "n_edges": int(len(e)),
        "n_chains": int(len(ch)),
        "mean_chain_length": mean_len,
        "share_len1": share1,
        "isolated_sentence_rate": float((comp["size"] == 1).sum() / len(sents)),
        "largest_component_share": float((largest / per_comment_n).mean()),
        "top_signatures": [[r.signature, int(r.n_chains)] for r in sig.head(10).itertuples()],
        "n_bridging_entities": int(br["bridges"].sum()),
        "n_never_bridging": int((~br["bridges"]).sum()),
        "null": {"mean_length_z": zl, "mean_length_p": pl,
                 "share_len1_z": z1, "share_len1_p": p1},
        "n_null": int(n_null),
        # A7.2: article context
        "prompt_variant": common.prompt_variant(aid),
        "n_prompt_echo_groups": len(echo_ids),
        "boilerplate_share": boilerplate_share,
        "sufficient": sufficient,
        # A7.1: headline numbers under every view, null included per-view (A7.4)
        "views": {view: _view_headline(vd) for view, vd in per_view.items()},
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


def write_cross_article(articles=None) -> Path:
    articles = list(articles or common.ARTICLES)
    frames = []
    sparsity_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for aid in articles:
        sents = common.sentences(aid)
        groups = entities.load(aid)
        m = entities.mentions(aid)
        e = graph.edges(sents, graph.mention_sets(m))
        sp = sparsity(m, sents, e)
        sparsity_frames[aid] = sp

        sp_out = sp.copy()
        sp_out.insert(0, "article_id", aid)
        frames.append(sp_out)

        row = {"article_id": aid, "prompt_variant": common.prompt_variant(aid),
               "coverage": entities.coverage(m, sents).get("overall")}
        for view in VIEWS:
            mv, sv = _view_frames(m, sents, groups, view)
            vch = graph.chains(sv, graph.mention_sets(mv))
            row[f"n_chains__{view}"] = int(len(vch))
            row[f"mean_chain_length__{view}"] = float(vch["length"].mean()) if len(vch) else float("nan")
            row[f"share_len1__{view}"] = float((vch["length"] == 1).mean()) if len(vch) else float("nan")
            s_path = common.out_dir(aid) / "summary.json"
            gview = (json.loads(s_path.read_text()).get("views", {}).get(view, {}).get("graph", {})
                     if s_path.exists() else {})
            for k, v in gview.items():
                row[f"graph_{k}__{view}"] = v["observed"]
                row[f"graph_{k}_z__{view}"] = v["z"]
        for r in sp.itertuples():
            row[f"sufficient__{r.category}"] = bool(r.sufficient)
        summary_rows.append(row)

    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = common.OUT_DIR / "cross_article_sparsity.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)

    # A7.3: variant split and the article-level summary the README's tables are built from.
    variant_path = common.OUT_DIR / "cross_article_by_variant.csv"
    sparsity_by_variant(sparsity_frames).to_csv(variant_path, index=False)

    summary_path = common.OUT_DIR / "cross_article_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return path


# The README's documented null sizes: 1,000 draws for these eight, 200 for everything else.
DEEP_NULL = {"181", "183", "91", "3", "4", "9", "8", "11"}


def _write_one(aid: str) -> tuple[str, float]:
    import time
    t = time.time()
    entities.write_mentions(aid)
    graph.write(aid)
    write(aid, n_null=1000 if aid in DEEP_NULL else 200)
    return aid, time.time() - t


def write_all(jobs: int = 8) -> Path:
    """Rebuild every article's outputs (mentions, graph tables, metrics with their null), in
    parallel over articles, largest first so article 41 doesn't finish last; then the
    cross-article tables."""
    import time
    from multiprocessing import Pool
    arts = common.coded_articles()
    size = {a: len(common.sentences(a)) * (5 if a in DEEP_NULL else 1) for a in arts}
    arts.sort(key=lambda a: -size[a])
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (aid, dt) in enumerate(pool.imap_unordered(_write_one, arts), start=1):
            print(f"{k}/{len(arts)} article {aid}: {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return write_cross_article()


if __name__ == "__main__":
    # python -m analysis.claim2_2.metrics 181          one article (1,000-draw null)
    # python -m analysis.claim2_2.metrics --all [N]    every article, N parallel jobs (default 8)
    # python -m analysis.claim2_2.metrics              cross-article tables only
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print(write_all(int(sys.argv[2]) if len(sys.argv) > 2 else 8))
    elif len(sys.argv) > 1:
        print(write(sys.argv[1]))
    else:
        print(write_cross_article())
