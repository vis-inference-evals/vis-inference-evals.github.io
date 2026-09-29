"""
Per-article stacked bar chart of sentence subcategory proportions (fine-grained 1–10 order),
excluding Meta / Paratext and Uncategorizable — same logic as 1_across_plot.ipynb.

Writes HTML/PDF charts (default sort by Beyond-plot; second pair sorted least→most by
unique comment count per article) and CSVs under analysis/figures/ by default.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import altair as alt
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PIPELINE = SCRIPT_DIR.parent
REPO_ROOT = REPO_PIPELINE.parent
COMBINED_DIR = REPO_PIPELINE / "data" / "combined_data"
FIGURES_DIR = SCRIPT_DIR / "figures"

DEFAULT_HTML = FIGURES_DIR / "per_article_subcategory_stacked_bars.html"
DEFAULT_PDF = FIGURES_DIR / "per_article_subcategory_stacked_bars.pdf"
DEFAULT_HTML_BY_COMMENTS = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_comment_count.html"
DEFAULT_PDF_BY_COMMENTS = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_comment_count.pdf"
DEFAULT_SUBCAT_COUNTS_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_subcat_counts.csv"
DEFAULT_FINE_PROPS_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_fine_props.csv"
DEFAULT_SUMMARY_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_summary.csv"
DEFAULT_HTML_BY_GRAPH = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_graph_type.html"
DEFAULT_PDF_BY_GRAPH = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_graph_type.pdf"
DEFAULT_HTML_BY_TOPIC = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_topic.html"
DEFAULT_PDF_BY_TOPIC = FIGURES_DIR / "per_article_subcategory_stacked_bars_by_topic.pdf"

_TAG_CLEANUP = {
    # Legacy / alternate classifier outputs → canonical new names
    "L3: Trend and pattern analysis": "Relational data observations",
    "L1: Elemental and encoded properties": "Non-data observations",
    "L2: Statistical concepts and relations": "Discrete data observations",
    "VO1: Chart Structure, Layout & Text": "Non-data observations",
    "VO2: Data Point Reading": "Discrete data observations",
    "VO3: Comparisons, Trends & Patterns": "Relational data observations",
    "Personal/episodic retrieval": "Personal knowledge",
    "Prior Knowledge: Personal /Episodic": "Personal knowledge",
    "Evaluative / affective judgment": "Reactive evaluations",
    "Explanatory inference": "Causal explanations",
    "Predictive / counterfactual inference": "Hypothetical arguments",
    "Information need / curiosity": "Open-ended questions",
    "Meta /Paratext": "Meta / Paratext",
    "Meta / paratext": "Meta / Paratext",
    "Visual Observation: Chart Structure & Text": "Non-data observations",
    "Visual Observation: Data Point Extraction": "Discrete data observations",
    "Visual Observation: Cross-point Pattern Recognition": "Relational data observations",
    "Prior Knowledge: Background": "Background knowledge",
    "Prior Knowledge: Personal / Episodic": "Personal knowledge",
    "Inference: Explanatory": "Causal explanations",
    "Inference: Predictive / Hypothetical": "Hypothetical arguments",
    # Current canonical names in combined_data/ → new display-ready names
    "Visual Observation: Chart Elements": "Non-data observations",
    "Visual Observation: Data Extraction": "Discrete data observations",
    "Visual Observation: Pattern Recognition": "Relational data observations",
    "Inference: Causal": "Causal explanations",
    "Inference: Hypothetical": "Hypothetical arguments",
    "Curiosity": "Open-ended questions",
    "Prior Knowledge: General": "Background knowledge",
    "Prior Knowledge: Personal": "Personal knowledge",
    "Evaluative: Prescriptive": "Prescriptive evaluations",
    "Evaluative: Reactive": "Reactive evaluations",
}
EXCLUDED = {
    "Meta / Paratext",
    "Uncategorizable",
    "unprompted",
    "instruction_repeat",
    "unknown",
}


FINEGRAIN_CATS = [
    "Non-data observations",
    "Discrete data observations",
    "Relational data observations",
    "Causal explanations",
    "Hypothetical arguments",
    "Open-ended questions",
    "Background knowledge",
    "Personal knowledge",
    "Prescriptive evaluations",
    "Reactive evaluations",
]

FINEGRAIN_COLORS = [
    "#1F3A5F",  # Non-data observations
    "#4A7FBA",  # Discrete data observations
    "#9BB8DC",  # Relational data observations
    "#2D6A4F",  # Causal explanations
    "#5B9468",  # Hypothetical arguments
    "#A8CC8E",  # Open-ended questions
    "#D68910",  # Background knowledge
    "#F4C77A",  # Personal knowledge
    "#9D1B2F",  # Prescriptive evaluations
    "#E04050",  # Reactive evaluations
]

# Prefixed labels so ascending alphabetical order = stack bottom → top (a bottom, j top).
FINEGRAIN_PLOT_LABELS = [
    "a. Non-data observations",
    "b. Discrete data observations",
    "c. Relational data observations",
    "d. Causal explanations",
    "e. Hypothetical arguments",
    "f. Open-ended questions",
    "g. Background knowledge",
    "h. Personal knowledge",
    "i. Prescriptive evaluations",
    "j. Reactive evaluations",
]
SUBCAT_TO_PLOT_LABEL = dict(zip(FINEGRAIN_CATS, FINEGRAIN_PLOT_LABELS))

SUBCAT_TO_GROUP = {
    "Non-data observations": "Visual Observation",
    "Discrete data observations": "Visual Observation",
    "Relational data observations": "Visual Observation",
    "Background knowledge": "Prior Knowledge",
    "Personal knowledge": "Prior Knowledge",
    "Prescriptive evaluations": "Evaluative Response",
    "Reactive evaluations": "Evaluative Response",
    "Causal explanations": "Inference",
    "Hypothetical arguments": "Inference",
    "Open-ended questions": "Curiosity",
}

VO_SUBCATS = [
    "Non-data observations",
    "Discrete data observations",
    "Relational data observations",
]


def load_sentence_classifications_df() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(COMBINED_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            article = json.load(f)
        if not isinstance(article, dict):
            continue
        article_id = str(article.get("article_id", path.stem))
        for comment in article.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            cid = comment.get("comment_index")
            for sentence in comment.get("sentences") or []:
                if not isinstance(sentence, dict):
                    continue
                tag = sentence.get("comment_tag")
                if not isinstance(tag, str) or not tag:
                    continue
                rows.append({"article_id": article_id, "comment_id": cid, "comment_tag": tag})
    if not rows:
        return pd.DataFrame(columns=["article_id", "comment_id", "comment_tag"])
    df_raw = pd.DataFrame(rows)
    df_raw["article_id"] = df_raw["article_id"].astype(str)
    df_raw["comment_id"] = pd.to_numeric(df_raw["comment_id"], errors="coerce").fillna(-1).astype(int)
    df_raw["comment_tag"] = df_raw["comment_tag"].replace(_TAG_CLEANUP)
    return df_raw[~df_raw["comment_tag"].isin(EXCLUDED)].copy()


def load_article_metadata() -> pd.DataFrame:
    """Load article-level metadata (graph type, topic) from combined_data/*.json."""
    rows: list[dict] = []
    for path in sorted(COMBINED_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            article = json.load(f)
        if not isinstance(article, dict):
            continue
        article_id = str(article.get("article_id", path.stem))
        graph_type = (article.get("graph_type") or "").upper().strip() or "UNCLASSIFIED"
        topic = (article.get("topic") or "").strip() or "UNCLASSIFIED"
        rows.append({"article_id": article_id, "graph_type": graph_type, "topic": topic})
    return pd.DataFrame(rows, columns=["article_id", "graph_type", "topic"])


def build_subcat_counts_and_chart(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, alt.Chart, alt.Chart]:
    df_fine = df[df["comment_tag"].isin(FINEGRAIN_CATS)].copy()
    df_fine["subcat"] = df_fine["comment_tag"]
    df_fine["group"] = df_fine["subcat"].map(SUBCAT_TO_GROUP)

    fine_counts = df_fine.groupby(["article_id", "subcat"], observed=True).size().unstack(fill_value=0)
    fine_props = fine_counts.div(fine_counts.sum(axis=1), axis=0)

    for c in FINEGRAIN_CATS:
        if c not in fine_props.columns:
            fine_props[c] = 0.0
    fine_props = fine_props[FINEGRAIN_CATS]

    summary = fine_props[FINEGRAIN_CATS].describe().T

    fine_props_out = fine_props.copy()
    fine_props_out["Beyond-plot"] = 1 - fine_props_out[VO_SUBCATS].sum(axis=1)
    article_order = sorted(
        fine_props_out.index.tolist(),
        key=lambda a: (int(a) if str(a).isdigit() else float("inf"), str(a)),
    )

    comments_per_article = df.groupby("article_id", observed=True)["comment_id"].nunique()
    order_comments = pd.DataFrame({"article_id": fine_props_out.index})
    order_comments["n_comments"] = order_comments["article_id"].map(comments_per_article)
    article_order_by_comments = (
        order_comments.sort_values(["n_comments", "article_id"], ascending=[True, True])["article_id"]
        .tolist()
    )

    subcat_counts = (
        df_fine.groupby(["article_id", "group", "subcat"], observed=True)
        .size()
        .reset_index(name="n")
    )
    subcat_counts["group_total"] = subcat_counts.groupby(["article_id", "group"], observed=True)[
        "n"
    ].transform("sum")
    subcat_counts["subcat_prop"] = subcat_counts["n"] / subcat_counts["group_total"]
    subcat_counts["subcat_total"] = subcat_counts.groupby(["article_id"], observed=True)["n"].transform("sum")
    subcat_counts["top_prop"] = subcat_counts["n"] / subcat_counts["subcat_total"]
    subcat_counts["subcat_plot"] = subcat_counts["subcat"].map(SUBCAT_TO_PLOT_LABEL)

    n_all_sentences = len(df)
    global_by_subcat = (
        df_fine.groupby(["group", "subcat"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["group", "subcat"])
    )
    global_total_fine = int(global_by_subcat["n"].sum())
    global_by_subcat["subcat_total"] = global_total_fine
    global_by_subcat["top_prop"] = (
        global_by_subcat["n"] / global_by_subcat["subcat_total"] if global_total_fine else 0.0
    )
    global_by_subcat["group_total"] = global_by_subcat.groupby("group", observed=True)["n"].transform("sum")
    global_by_subcat["subcat_prop"] = global_by_subcat["n"] / global_by_subcat["group_total"]
    global_by_subcat["subcat_plot"] = global_by_subcat["subcat"].map(SUBCAT_TO_PLOT_LABEL)
    global_by_subcat["article_id"] = "All sentences"
    global_by_subcat["prop_of_all_sentences"] = (
        global_by_subcat["n"] / n_all_sentences if n_all_sentences else 0.0
    )

    color_enc = alt.Color(
        "subcat_plot:N",
        sort=FINEGRAIN_PLOT_LABELS,
        scale=alt.Scale(domain=FINEGRAIN_PLOT_LABELS, range=FINEGRAIN_COLORS),
        title="Subcategory",
        legend=alt.Legend(orient="right", title="Subcategory"),
    )

    def article_stacked_chart(article_sort: list[str], title_suffix: str | None = None) -> alt.Chart:
        base_title = (
            "Sentence subcategory proportions by article\n"
            "(Stacked order 1–10: VO → Curiosity → Inference → Prior Knowledge → Evaluative)"
        )
        if title_suffix:
            base_title = f"{base_title}\n{title_suffix}"
        return (
            alt.Chart(
                subcat_counts,
                title=base_title,
            )
            .mark_bar()
            .encode(
                x=alt.X("article_id:N", sort=article_sort, title="Article"),
                y=alt.Y("top_prop:Q", stack="normalize", title="Proportion of sentences"),
                color=color_enc,
                order=alt.Order("subcat_plot:N", sort="ascending"),
                tooltip=[
                    "article_id",
                    "group",
                    "subcat",
                    alt.Tooltip("top_prop:Q", format=".1%", title="Article prop (of all)"),
                    alt.Tooltip("subcat_prop:Q", format=".1%", title="Subcat prop (of group)"),
                    "n",
                ],
            )
            .properties(width=800, height=350)
        )

    chart_articles = article_stacked_chart(article_order)
    chart_articles_by_comments = article_stacked_chart(
        article_order_by_comments,
        "Articles sorted by number of comments (least → most).",
    )

    chart_all = (
        alt.Chart(
            global_by_subcat,
            title="Across all sentences",
        )
        .mark_bar()
        .encode(
            x=alt.X("article_id:N", title=None),
            y=alt.Y("top_prop:Q", stack="normalize", title="Proportion of sentences"),
            color=color_enc,
            order=alt.Order("subcat_plot:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("subcat:O", title="Subcategory"),
                "group",
                alt.Tooltip("top_prop:Q", format=".1%", title="Prop (of fine-subcat sentences)"),
                alt.Tooltip("prop_of_all_sentences:Q", format=".1%", title="Prop (of all sentences)"),
                alt.Tooltip("subcat_prop:Q", format=".1%", title="Subcat prop (of group)"),
                "n",
            ],
        )
        .properties(width=72, height=350)
    )

    chart = alt.hconcat(chart_all, chart_articles, spacing=16).resolve_scale(color="shared", y="shared")
    chart_by_comments = alt.hconcat(chart_all, chart_articles_by_comments, spacing=16).resolve_scale(
        color="shared", y="shared"
    )

    return subcat_counts, fine_props_out, summary, chart, chart_by_comments


def build_grouped_charts(
    subcat_counts: pd.DataFrame,
    fine_props_out: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[alt.Chart, alt.Chart]:
    """Build two faceted charts: articles grouped by graph type and by topic."""
    article_order = sorted(
        fine_props_out.index.tolist(),
        key=lambda a: (int(a) if str(a).isdigit() else float("inf"), str(a)),
    )

    merged = subcat_counts.merge(meta, on="article_id", how="left")
    merged["graph_type"] = merged["graph_type"].fillna("UNCLASSIFIED")
    merged["topic"] = merged["topic"].fillna("UNCLASSIFIED")

    color_enc = alt.Color(
        "subcat_plot:N",
        sort=FINEGRAIN_PLOT_LABELS,
        scale=alt.Scale(domain=FINEGRAIN_PLOT_LABELS, range=FINEGRAIN_COLORS),
        title="Subcategory",
        legend=alt.Legend(orient="right", title="Subcategory"),
    )

    def _faceted_chart(group_col: str, group_title: str) -> alt.Chart:
        return (
            alt.Chart(merged)
            .mark_bar()
            .encode(
                x=alt.X("article_id:N", sort=article_order, title="Article"),
                y=alt.Y("top_prop:Q", stack="normalize", title="Proportion of sentences"),
                color=color_enc,
                order=alt.Order("subcat_plot:N", sort="ascending"),
                tooltip=[
                    "article_id",
                    alt.Tooltip(f"{group_col}:N", title=group_title),
                    "group",
                    "subcat",
                    alt.Tooltip("top_prop:Q", format=".1%", title="Article prop (of all)"),
                    alt.Tooltip("subcat_prop:Q", format=".1%", title="Subcat prop (of group)"),
                    "n",
                ],
            )
            .properties(width=alt.Step(18), height=350)
            .facet(
                column=alt.Column(
                    f"{group_col}:N",
                    title=group_title,
                    header=alt.Header(labelAngle=-45, labelAlign="right"),
                ),
                title=(
                    f"Sentence subcategory proportions by article — grouped by {group_title}\n"
                    "(Stacked order 1–10: VO → Curiosity → Inference → Prior Knowledge → Evaluative)"
                ),
            )
            .resolve_scale(x="independent")
        )

    return _faceted_chart("graph_type", "Graph Type"), _faceted_chart("topic", "Topic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output Altair HTML (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Output Altair PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--html-by-comment-count",
        type=Path,
        default=DEFAULT_HTML_BY_COMMENTS,
        help=f"Altair HTML sorted by comment count (default: {DEFAULT_HTML_BY_COMMENTS})",
    )
    parser.add_argument(
        "--pdf-by-comment-count",
        type=Path,
        default=DEFAULT_PDF_BY_COMMENTS,
        help=f"Altair PDF sorted by comment count (default: {DEFAULT_PDF_BY_COMMENTS})",
    )
    parser.add_argument(
        "--subcat-counts-csv",
        type=Path,
        default=DEFAULT_SUBCAT_COUNTS_CSV,
        help="CSV for the plotting table (article, group, subcat, counts, proportions)",
    )
    parser.add_argument(
        "--fine-props-csv",
        type=Path,
        default=DEFAULT_FINE_PROPS_CSV,
        help="CSV: per-article proportion columns + Beyond-plot",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV: describe().T over articles per subcategory",
    )
    parser.add_argument(
        "--html-by-graph-type",
        type=Path,
        default=DEFAULT_HTML_BY_GRAPH,
        help=f"Altair HTML grouped by graph type (default: {DEFAULT_HTML_BY_GRAPH})",
    )
    parser.add_argument(
        "--pdf-by-graph-type",
        type=Path,
        default=DEFAULT_PDF_BY_GRAPH,
        help=f"Altair PDF grouped by graph type (default: {DEFAULT_PDF_BY_GRAPH})",
    )
    parser.add_argument(
        "--html-by-topic",
        type=Path,
        default=DEFAULT_HTML_BY_TOPIC,
        help=f"Altair HTML grouped by topic (default: {DEFAULT_HTML_BY_TOPIC})",
    )
    parser.add_argument(
        "--pdf-by-topic",
        type=Path,
        default=DEFAULT_PDF_BY_TOPIC,
        help=f"Altair PDF grouped by topic (default: {DEFAULT_PDF_BY_TOPIC})",
    )
    parser.add_argument(
        "--no-vegafusion",
        action="store_true",
        help="Do not enable vegafusion (use if unsupported in your environment)",
    )
    args = parser.parse_args()

    if not COMBINED_DIR.is_dir():
        print(f"Missing directory: {COMBINED_DIR}", file=sys.stderr)
        return 1

    if not args.no_vegafusion:
        try:
            alt.data_transformers.enable("vegafusion")
        except Exception:
            pass

    df = load_sentence_classifications_df()
    if df.empty:
        print(f"No classification rows under {COMBINED_DIR}", file=sys.stderr)
        return 1

    subcat_counts, fine_props_out, summary, chart, chart_by_comments = build_subcat_counts_and_chart(df)

    meta = load_article_metadata()
    chart_by_graph, chart_by_topic = build_grouped_charts(subcat_counts, fine_props_out, meta)

    args.html.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.html))
    chart.save(str(args.pdf))
    chart_by_comments.save(str(args.html_by_comment_count))
    chart_by_comments.save(str(args.pdf_by_comment_count))
    chart_by_graph.save(str(args.html_by_graph_type))
    chart_by_graph.save(str(args.pdf_by_graph_type))
    chart_by_topic.save(str(args.html_by_topic))
    chart_by_topic.save(str(args.pdf_by_topic))
    subcat_counts.to_csv(args.subcat_counts_csv, index=False)
    fine_props_out.to_csv(args.fine_props_csv)
    summary.round(6).to_csv(args.summary_csv)

    n_fine = len(df[df["comment_tag"].isin(FINEGRAIN_CATS)])
    print(
        f"Wrote {args.html}\n"
        f"       {args.html_by_comment_count}\n"
        f"       {args.html_by_graph_type}\n"
        f"       {args.html_by_topic}\n"
        f"       {args.subcat_counts_csv} ({len(subcat_counts)} rows)\n"
        f"       {args.fine_props_csv} ({len(fine_props_out)} articles)\n"
        f"       {args.summary_csv}"
    )
    print(f"Sentences (excl. Meta/Uncategorizable): {len(df):,}; in fine subcats: {n_fine:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
