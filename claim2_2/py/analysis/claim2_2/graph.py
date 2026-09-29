"""Per-comment sentence graph: entity-shared edges, reading-order chains, components."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from analysis.claim2_2 import common, entities

JOIN = "|"


def mention_sets(m: pd.DataFrame) -> dict[tuple[int, int], set[str]]:
    out: dict[tuple[int, int], set[str]] = {}
    for r in m.itertuples():
        out.setdefault((int(r.comment_index), int(r.sent_id)), set()).add(r.entity_id)
    return out


def _by_comment(sents: pd.DataFrame):
    for ci, sub in sents.sort_values(["comment_index", "order", "sent_id"], kind="stable").groupby("comment_index"):
        yield int(ci), sub.reset_index(drop=True)


def edges(sents: pd.DataFrame, msets: dict) -> pd.DataFrame:
    rows = []
    for ci, sub in _by_comment(sents):
        ids = [int(s) for s in sub["sent_id"]]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                shared = msets.get((ci, ids[i]), set()) & msets.get((ci, ids[j]), set())
                if shared:
                    rows.append({"comment_index": ci, "sent_a": ids[i], "sent_b": ids[j],
                                 "shared_entities": JOIN.join(sorted(shared)),
                                 "n_shared": len(shared), "adjacent": j == i + 1})
    cols = ["comment_index", "sent_a", "sent_b", "shared_entities", "n_shared", "adjacent"]
    return pd.DataFrame(rows, columns=cols)


def chains(sents: pd.DataFrame, msets: dict) -> pd.DataFrame:
    rows = []
    for ci, sub in _by_comment(sents):
        ids = [int(s) for s in sub["sent_id"]]
        cats = list(sub["category"])
        start = 0
        for i in range(1, len(ids) + 1):
            broken = i == len(ids) or not (msets.get((ci, ids[i - 1]), set()) & msets.get((ci, ids[i]), set()))
            if not broken:
                continue
            seg_ids, seg_cats = ids[start:i], cats[start:i]
            seg_ents = set()
            for s in seg_ids:
                seg_ents |= msets.get((ci, s), set())
            kept = [c for c in seg_cats if c != common.REDUNDANT]
            rows.append({"comment_index": ci, "chain_id": len(rows),
                         "sent_ids": JOIN.join(str(s) for s in seg_ids),
                         "length": len(seg_ids),
                         "signature": "-".join(kept),
                         "signature_raw": "-".join(seg_cats),
                         "entities": JOIN.join(sorted(seg_ents)),
                         "has_inf": "INF" in seg_cats})
            start = i
    cols = ["comment_index", "chain_id", "sent_ids", "length", "signature",
            "signature_raw", "entities", "has_inf"]
    return pd.DataFrame(rows, columns=cols)


def components(sents: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ci, sub in _by_comment(sents):
        ids = [int(s) for s in sub["sent_id"]]
        cat = dict(zip(ids, sub["category"]))
        parent = {s: s for s in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ents: dict[int, set[str]] = {s: set() for s in ids}
        sel = edges_df[edges_df["comment_index"] == ci]
        for r in sel.itertuples():
            a, b = find(int(r.sent_a)), find(int(r.sent_b))
            if a != b:
                parent[a] = b
            for s in (int(r.sent_a), int(r.sent_b)):
                ents[s] |= set(str(r.shared_entities).split(JOIN))
        groups: dict[int, list[int]] = {}
        for s in ids:
            groups.setdefault(find(s), []).append(s)
        for k, members in groups.items():
            members = sorted(members)
            e = set()
            for s in members:
                e |= ents[s]
            rows.append({"comment_index": ci, "component_id": len(rows),
                         "sent_ids": JOIN.join(str(s) for s in members),
                         "size": len(members),
                         "categories": JOIN.join(cat[s] for s in members),
                         "entities": JOIN.join(sorted(e))})
    cols = ["comment_index", "component_id", "sent_ids", "size", "categories", "entities"]
    return pd.DataFrame(rows, columns=cols)


GRAPH_COLS = (
    ["comment_index", "n_nodes", "n_edges", "density", "n_components", "n_isolated", "cycle_rank",
     "max_clique", "n_max_cliques", "n_articulation", "n_edges_same", "n_edges_cross"]
    + [f"{k}_{c}" for c in common.CATEGORIES for k in ("n", "degree", "n_articulation")]
)


def comment_graph_metrics(sents: pd.DataFrame, msets: dict) -> pd.DataFrame:
    """One row per comment describing its undirected statement graph.

    Nodes are the comment's statements (redundant ones included, as in `chains` and
    `components`); two statements are joined when they share an entity. Per comment:
    density, components, isolated statements, cycle rank (edges - nodes + components: 0 for a
    tree or chain, above 0 when the reader ties points back together), the largest clique and
    the number of maximal cliques of size >= 2, articulation points (statements whose removal
    disconnects the rest), and, over VO/PK/INF statements only, per-category node counts,
    summed degree and articulation points, and edges joining the same vs different categories.
    """
    import networkx as nx

    rows = []
    for ci, sub in _by_comment(sents):
        ids = [int(s) for s in sub["sent_id"]]
        cat = dict(zip(ids, sub["category"]))
        g = nx.Graph()
        g.add_nodes_from(ids)
        for i in range(len(ids)):
            ei = msets.get((ci, ids[i]), set())
            if not ei:
                continue
            for j in range(i + 1, len(ids)):
                if ei & msets.get((ci, ids[j]), set()):
                    g.add_edge(ids[i], ids[j])
        n, e = g.number_of_nodes(), g.number_of_edges()
        comps = nx.number_connected_components(g)
        cliques = [c for c in nx.find_cliques(g) if len(c) >= 2] if e else []
        art = set(nx.articulation_points(g)) if e else set()
        same = cross = 0
        for a, b in g.edges():
            ca, cb = cat[a], cat[b]
            if ca in common.CATEGORIES and cb in common.CATEGORIES:
                same += ca == cb
                cross += ca != cb
        row = {"comment_index": ci, "n_nodes": n, "n_edges": e,
               "density": e / (n * (n - 1) / 2) if n > 1 else float("nan"),
               "n_components": comps, "n_isolated": sum(1 for s in ids if g.degree(s) == 0),
               "cycle_rank": e - n + comps,
               "max_clique": max((len(c) for c in cliques), default=1),
               "n_max_cliques": len(cliques), "n_articulation": len(art),
               "n_edges_same": same, "n_edges_cross": cross}
        for c in common.CATEGORIES:
            members = [s for s in ids if cat[s] == c]
            row[f"n_{c}"] = len(members)
            row[f"degree_{c}"] = sum(g.degree(s) for s in members)
            row[f"n_articulation_{c}"] = sum(1 for s in members if s in art)
        rows.append(row)
    return pd.DataFrame(rows, columns=GRAPH_COLS)


def graph_summary(per_comment: pd.DataFrame) -> dict:
    """Article-level numbers from `comment_graph_metrics`, the ones tested against the null.
    Means are over comments (density only over comments with at least two statements);
    rates and degrees are pooled over statements; cross_edge_share is pooled over edges."""
    pc = per_comment
    nodes = pc["n_nodes"].sum()
    typed_edges = pc["n_edges_same"].sum() + pc["n_edges_cross"].sum()
    out = {
        "mean_density": float(pc["density"].mean()),  # NaN rows (single statement) skipped
        "mean_cycle_rank": float(pc["cycle_rank"].mean()),
        "share_cyclic": float((pc["cycle_rank"] > 0).mean()),
        "mean_max_clique": float(pc["max_clique"].mean()),
        "isolated_rate": float(pc["n_isolated"].sum() / nodes) if nodes else float("nan"),
        "articulation_rate": float(pc["n_articulation"].sum() / nodes) if nodes else float("nan"),
        "cross_edge_share": float(pc["n_edges_cross"].sum() / typed_edges) if typed_edges else float("nan"),
    }
    for c in common.CATEGORIES:
        k = pc[f"n_{c}"].sum()
        out[f"articulation_rate_{c}"] = float(pc[f"n_articulation_{c}"].sum() / k) if k else float("nan")
        out[f"mean_degree_{c}"] = float(pc[f"degree_{c}"].sum() / k) if k else float("nan")
    return out


def node_label(entity_id: str, category: str) -> str:
    return f"{entity_id}·{category}"


def entity_links(sents: pd.DataFrame, msets: dict, edges_df: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Article-level graph over <entity, category> nodes, categories VO/PK/INF only.

    Within a comment, two nodes are connected when they are mentioned in the same statement, or
    in two statements joined by a shared-entity edge (every entity of one statement to every
    entity of the other). Redundant statements contribute no nodes and no connections.

    Returns (nodes, links). A node's `n_comments` counts comments containing it. Each link is
    keyed by its two labels in sorted order (a < b); `n_connected` counts comments in which the
    pair is connected, `p_b_given_a = n_connected / n_comments(a)` and vice versa, and `p` is
    the larger of the two.
    """
    cat = {(int(c), int(s)): k for c, s, k in zip(sents["comment_index"], sents["sent_id"], sents["category"])}
    stmt_nodes: dict[tuple[int, int], set[str]] = {}
    for key, ents in msets.items():
        k = cat.get(key)
        if k in common.CATEGORIES and ents:
            stmt_nodes[key] = {node_label(e, k) for e in ents}
    comment_nodes: dict[int, set[str]] = {}
    comment_pairs: dict[int, set[tuple[str, str]]] = {}
    for (ci, _), ns in stmt_nodes.items():
        comment_nodes.setdefault(ci, set()).update(ns)
        ordered = sorted(ns)
        comment_pairs.setdefault(ci, set()).update(
            (ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered)))
    for r in edges_df.itertuples():
        ci = int(r.comment_index)
        na, nb = stmt_nodes.get((ci, int(r.sent_a))), stmt_nodes.get((ci, int(r.sent_b)))
        if not na or not nb:
            continue
        comment_pairs.setdefault(ci, set()).update(
            (min(x, y), max(x, y)) for x in na for y in nb if x != y)

    n_comments: dict[str, int] = {}
    for ns in comment_nodes.values():
        for n in ns:
            n_comments[n] = n_comments.get(n, 0) + 1
    n_conn: dict[tuple[str, str], int] = {}
    for ps in comment_pairs.values():
        for p in ps:
            n_conn[p] = n_conn.get(p, 0) + 1

    nodes = pd.DataFrame(
        [{"node": n, "entity_id": n.rsplit("·", 1)[0], "category": n.rsplit("·", 1)[1], "n_comments": k}
         for n, k in n_comments.items()],
        columns=["node", "entity_id", "category", "n_comments"])
    links = pd.DataFrame(
        [{"a": a, "b": b, "n_connected": k,
          "p_b_given_a": k / n_comments[a], "p_a_given_b": k / n_comments[b]}
         for (a, b), k in n_conn.items()],
        columns=["a", "b", "n_connected", "p_b_given_a", "p_a_given_b"])
    links["p"] = links[["p_b_given_a", "p_a_given_b"]].max(axis=1)
    return nodes, links


def directed_entity_links(sents: pd.DataFrame, msets: dict, edges_df: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Article-level directed graph over <entity, category> nodes (VO/PK/INF only), oriented by
    reading order.

    Within a comment, a -> b when a statement mentioning a is joined by a shared-entity edge to a
    *later* statement mentioning b (every entity of the earlier statement to every entity of the
    later one). Mentions inside one statement have no order and give no directed link; a node never
    links to itself. Direction is the order readers wrote in, not an inferred dependency.

    Returns (nodes, links). A node's `n_comments` counts comments containing it (as in
    `entity_links`). Each link a -> b has `n_forward` (comments where a -> b), `p = n_forward /
    n_comments(a)` -- the probability, across the article's comments, that a is followed by a linked
    mention of b -- plus `n_reverse` (comments where b -> a) and `forward_share = n_forward /
    (n_forward + n_reverse)`; a comment can count in both directions.
    """
    order = {}
    for ci, sub in _by_comment(sents):
        for k, s in enumerate(sub["sent_id"]):
            order[(ci, int(s))] = k
    cat = {(int(c), int(s)): k for c, s, k in zip(sents["comment_index"], sents["sent_id"], sents["category"])}
    stmt_nodes: dict[tuple[int, int], set[str]] = {}
    for key, ents in msets.items():
        k = cat.get(key)
        if k in common.CATEGORIES and ents and key in order:
            stmt_nodes[key] = {node_label(e, k) for e in ents}
    comment_nodes: dict[int, set[str]] = {}
    for (ci, _), ns in stmt_nodes.items():
        comment_nodes.setdefault(ci, set()).update(ns)
    comment_pairs: dict[int, set[tuple[str, str]]] = {}
    for r in edges_df.itertuples():
        ci = int(r.comment_index)
        ka, kb = (ci, int(r.sent_a)), (ci, int(r.sent_b))
        if ka not in stmt_nodes or kb not in stmt_nodes:
            continue
        if order[ka] > order[kb]:
            ka, kb = kb, ka
        comment_pairs.setdefault(ci, set()).update(
            (x, y) for x in stmt_nodes[ka] for y in stmt_nodes[kb] if x != y)

    n_comments: dict[str, int] = {}
    for ns in comment_nodes.values():
        for n in ns:
            n_comments[n] = n_comments.get(n, 0) + 1
    n_dir: dict[tuple[str, str], int] = {}
    for ps in comment_pairs.values():
        for p in ps:
            n_dir[p] = n_dir.get(p, 0) + 1

    nodes = pd.DataFrame(
        [{"node": n, "entity_id": n.rsplit("·", 1)[0], "category": n.rsplit("·", 1)[1], "n_comments": k}
         for n, k in n_comments.items()],
        columns=["node", "entity_id", "category", "n_comments"])
    rows = []
    for (a, b), k in n_dir.items():
        rev = n_dir.get((b, a), 0)
        rows.append({"src": a, "dst": b, "n_forward": k, "n_reverse": rev,
                     "p": k / n_comments[a], "p_reverse": rev / n_comments[b],
                     "forward_share": k / (k + rev)})
    links = pd.DataFrame(rows, columns=["src", "dst", "n_forward", "n_reverse", "p", "p_reverse",
                                        "forward_share"])
    return nodes, links


def write(article_id: str) -> dict[str, Path]:
    sents = common.sentences(article_id)
    msets = mention_sets(entities.mentions(article_id))
    e = edges(sents, msets)
    ch = chains(sents, msets)
    comp = components(sents, e)
    out = common.out_dir(article_id)
    paths = {}
    for name, df in [("edges", e), ("chains", ch), ("components", comp)]:
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


if __name__ == "__main__":
    aid = sys.argv[1] if len(sys.argv) > 1 else "181"
    for name, p in write(aid).items():
        print(name, p)
