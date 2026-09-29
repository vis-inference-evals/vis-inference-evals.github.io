"""
Claim 1 figure: sentence subcategory proportions, stacked bars.

Layout (left → right):
  1. "All sentences"                          – one stacked bar over the whole corpus
  2. Three stacked bars, one per prompt       – notice / wonder / relate
  3. One stacked bar per article, sorted least → most unique comments

Differences from analysis/per_article_subcategory_stacked_bars.py:
  * drops every sentence tagged Meta / Paratext, Uncategorizable, unknown,
    instruction_repeat or unprompted, and every sentence whose section_label is
    instruction_repeat (prompt scaffolding rather than reader text);
  * drops boilerplate prompt scaffolding via analysis/claim1/boilerplate.py,
    a residual-content test that replaces the unanchored regexes in
    visual-inference-analysis/2_sentence_similarity_filtering.ipynb.

Outputs go to analysis/claim1/figures/ by default.
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
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from per_article_subcategory_stacked_bars import (  # noqa: E402
    COMBINED_DIR,
    EXCLUDED,
    FINEGRAIN_CATS,
    VO_SUBCATS,
    _TAG_CLEANUP,
)

FIGURES_DIR = SCRIPT_DIR / "figures"
STEM = "per_article_subcategory_stacked_bars_by_comment_count"

# Claim-local taxonomy: the two evaluative subcategories sit in the same coarse
# category as causal / hypothetical / open-ended, so the five of them share one
# green ramp and stack contiguously.  Overrides the shared constants in
# analysis/per_article_subcategory_stacked_bars.py, which other analyses keep.
SUBCAT_TO_GROUP = {
    "Non-data observations": "Visual Observation",
    "Discrete data observations": "Visual Observation",
    "Relational data observations": "Visual Observation",
    "Causal explanations": "Inference",
    "Hypothetical arguments": "Inference",
    "Open-ended questions": "Inference",
    "Prescriptive evaluations": "Inference",
    "Reactive evaluations": "Inference",
    "Background knowledge": "Prior Knowledge",
    "Personal knowledge": "Prior Knowledge",
}

# Stack order bottom → top, with the plot-label prefixes doing the sorting.
FINEGRAIN_PLOT_ORDER = [
    ("Non-data observations", "#1F3A5F"),
    ("Discrete data observations", "#4A7FBA"),
    ("Relational data observations", "#9BB8DC"),
    ("Causal explanations", "#1B4D3A"),
    ("Hypothetical arguments", "#2D6A4F"),
    ("Open-ended questions", "#5B9468"),
    ("Prescriptive evaluations", "#8FC08A"),
    ("Reactive evaluations", "#C3DFAE"),
    ("Background knowledge", "#D68910"),
    ("Personal knowledge", "#F4C77A"),
]
FINEGRAIN_PLOT_LABELS = [
    f"{chr(ord('a') + i)}. {name}" for i, (name, _) in enumerate(FINEGRAIN_PLOT_ORDER)
]
FINEGRAIN_COLORS = [color for _, color in FINEGRAIN_PLOT_ORDER]
SUBCAT_TO_PLOT_LABEL = {
    name: label
    for (name, _), label in zip(FINEGRAIN_PLOT_ORDER, FINEGRAIN_PLOT_LABELS, strict=True)
}

# Section labels that are prompt scaffolding, not reader text.
EXCLUDED_SECTIONS = {"instruction_repeat"}

# The three WGOITG prompts, in the order they are asked.
PROMPT_SECTIONS = ["notice", "wonder", "relate"]
PROMPT_TITLES = {
    "notice": "Notice",
    "wonder": "Wonder",
    "relate": "Relate",
}

from boilerplate import is_boilerplate  # noqa: E402

def load_sentences() -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one row per kept sentence plus a dict of how many rows each filter dropped."""
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
                rows.append(
                    {
                        "article_id": article_id,
                        "comment_id": cid,
                        "comment_tag": tag,
                        "section_label": sentence.get("section_label") or "none",
                        "text": sentence.get("text") or "",
                    }
                )
    df = pd.DataFrame(rows)
    df["comment_id"] = pd.to_numeric(df["comment_id"], errors="coerce").fillna(-1).astype(int)
    df["comment_tag"] = df["comment_tag"].replace(_TAG_CLEANUP)

    dropped: dict[str, int] = {"total_raw": len(df)}
    m = df["comment_tag"].isin(EXCLUDED)
    dropped["tag_meta_other_instruction"] = int(m.sum())
    df = df[~m]
    m = df["section_label"].isin(EXCLUDED_SECTIONS)
    dropped["section_instruction_repeat"] = int(m.sum())
    df = df[~m]
    m = df["text"].map(is_boilerplate)
    dropped["boilerplate_scaffolding"] = int(m.sum())
    df = df[~m]
    m = ~df["comment_tag"].isin(FINEGRAIN_CATS)
    dropped["not_in_10_classes"] = int(m.sum())
    df = df[~m].copy()
    dropped["kept"] = len(df)
    return df, dropped


def _props(df: pd.DataFrame, key: str, key_name: str) -> pd.DataFrame:
    out = df.groupby([key, "comment_tag"], observed=True).size().reset_index(name="n")
    out = out.rename(columns={key: key_name, "comment_tag": "subcat"})
    out["group"] = out["subcat"].map(SUBCAT_TO_GROUP)
    out["total"] = out.groupby(key_name, observed=True)["n"].transform("sum")
    out["prop"] = out["n"] / out["total"]
    out["subcat_plot"] = out["subcat"].map(SUBCAT_TO_PLOT_LABEL)
    return out


def build_chart(df: pd.DataFrame) -> tuple[alt.Chart, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["subcat"] = df["comment_tag"]

    # 1. all sentences
    df["_all"] = "All sentences"
    all_counts = _props(df, "_all", "unit")

    # 2. three prompts
    df_prompt = df[df["section_label"].isin(PROMPT_SECTIONS)].copy()
    df_prompt["prompt"] = df_prompt["section_label"].map(PROMPT_TITLES)
    prompt_counts = _props(df_prompt, "prompt", "unit")
    prompt_order = [PROMPT_TITLES[s] for s in PROMPT_SECTIONS]

    # 3. per article, sorted by unique comment count
    article_counts = _props(df, "article_id", "article_id")
    n_comments = df.groupby("article_id", observed=True)["comment_id"].nunique().rename("n_comments")
    article_counts = article_counts.merge(n_comments, left_on="article_id", right_index=True)
    article_order = (
        n_comments.reset_index()
        .assign(_k=lambda d: d["article_id"].map(lambda a: int(a) if str(a).isdigit() else 10**9))
        .sort_values(["n_comments", "_k"])["article_id"]
        .tolist()
    )

    color_enc = alt.Color(
        "subcat_plot:N",
        sort=FINEGRAIN_PLOT_LABELS,
        scale=alt.Scale(domain=FINEGRAIN_PLOT_LABELS, range=FINEGRAIN_COLORS),
        legend=alt.Legend(orient="right", title="Subcategory"),
    )
    y_enc = alt.Y("prop:Q", stack="normalize", title="Proportion of sentences", axis=alt.Axis(format="%"))
    order_enc = alt.Order("subcat_plot:N", sort="ascending")
    tip_common = [
        alt.Tooltip("subcat:N", title="Subcategory"),
        alt.Tooltip("group:N", title="Category"),
        alt.Tooltip("prop:Q", format=".1%", title="Proportion"),
        alt.Tooltip("n:Q", title="Sentences"),
        alt.Tooltip("total:Q", title="Sentences in bar"),
    ]

    chart_all = (
        alt.Chart(all_counts, title="All sentences")
        .mark_bar()
        .encode(x=alt.X("unit:N", title=None, axis=alt.Axis(labels=False, ticks=False)), y=y_enc, color=color_enc, order=order_enc, tooltip=tip_common)
        .properties(width=60, height=350)
    )
    chart_prompts = (
        alt.Chart(prompt_counts, title=alt.TitleParams("By prompt", subtitle="notice / wonder / relate"))
        .mark_bar()
        .encode(
            x=alt.X("unit:N", sort=prompt_order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=y_enc,
            color=color_enc,
            order=order_enc,
            tooltip=[alt.Tooltip("unit:N", title="Prompt")] + tip_common,
        )
        .properties(width=alt.Step(70), height=350)
    )
    chart_articles = (
        alt.Chart(
            article_counts,
            title=alt.TitleParams(
                "By article, sorted by number of comments (least → most)",
                subtitle="Meta / paratext, uncategorizable, instruction text and boilerplate prompt echoes removed",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("article_id:N", sort=article_order, title="Article (196; hover for id)", axis=alt.Axis(labels=False, ticks=False)),
            y=y_enc,
            color=color_enc,
            order=order_enc,
            tooltip=[alt.Tooltip("article_id:N", title="Article"), alt.Tooltip("n_comments:Q", title="Comments")] + tip_common,
        )
        .properties(width=800, height=350)
    )

    chart = (
        alt.hconcat(chart_all, chart_prompts, chart_articles, spacing=20)
        .resolve_scale(color="shared", y="shared")
        .properties(
            title=alt.TitleParams(
                "Sentence subcategory proportions",
                subtitle="Stack order bottom → top: visual observation → inference (explanation, hypothetical, question, evaluation) → prior knowledge",
                anchor="start",
            )
        )
    )
    return chart, all_counts, prompt_counts, article_counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    if not COMBINED_DIR.is_dir():
        print(f"Missing directory: {COMBINED_DIR}", file=sys.stderr)
        return 1

    df, dropped = load_sentences()
    if df.empty:
        print("No sentences left after filtering", file=sys.stderr)
        return 1

    chart, all_counts, prompt_counts, article_counts = build_chart(df)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    chart.save(str(out / f"{STEM}.html"))
    if not args.no_pdf:
        chart.save(str(out / f"{STEM}.pdf"))
    all_counts.to_csv(out / f"{STEM}_all_sentences.csv", index=False)
    prompt_counts.to_csv(out / f"{STEM}_by_prompt.csv", index=False)
    article_counts.to_csv(out / f"{STEM}_by_article.csv", index=False)

    # wide per-article table with Beyond-plot, mirroring the parent script
    wide = article_counts.pivot(index="article_id", columns="subcat", values="prop").fillna(0.0)
    for c in FINEGRAIN_CATS:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide[FINEGRAIN_CATS]
    wide["Beyond-plot"] = 1 - wide[VO_SUBCATS].sum(axis=1)
    wide["n_comments"] = article_counts.drop_duplicates("article_id").set_index("article_id")["n_comments"]
    wide.to_csv(out / f"{STEM}_fine_props.csv")
    pd.Series(dropped).to_csv(out / f"{STEM}_filter_counts.csv", header=["n"])

    print(f"Wrote {out}/{STEM}.html (+ .pdf, CSVs)")
    for k, v in dropped.items():
        print(f"  {k:32s} {v:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
