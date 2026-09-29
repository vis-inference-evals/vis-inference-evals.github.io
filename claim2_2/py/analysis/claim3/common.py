"""Loading + category constants for the claim-3 analysis (one article at a time)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
COMBINED_DIR = REPO_ROOT / "data" / "combined_data"
OUT_DIR = PKG_DIR / "out"
CACHE_DIR = PKG_DIR / "cache"
CODEBOOK_DIR = PKG_DIR / "codebooks"

CATEGORIES = ["VO", "PK", "INF"]
REDUNDANT = "redundant"


def category_of(tag: str) -> str:
    if not isinstance(tag, str):
        return REDUNDANT
    if tag.startswith("Visual Observation"):
        return "VO"
    if tag.startswith("Prior Knowledge"):
        return "PK"
    if tag.startswith("Inference") or tag.startswith("Evaluative"):
        return "INF"
    return REDUNDANT


def load_article(article_id: str) -> dict:
    with (COMBINED_DIR / f"{article_id}.json").open(encoding="utf-8") as f:
        return json.load(f)


def sentence_table(article_id: str) -> pd.DataFrame:
    art = load_article(article_id)
    rows = []
    for c in art.get("comments") or []:
        for s in c.get("sentences") or []:
            tag = s.get("comment_tag")
            l2 = s.get("l2_predicate")
            rows.append({
                "article_id": str(art.get("article_id", article_id)),
                "comment_index": int(c["comment_index"]),
                "sent_id": int(s["id"]),
                "order": s.get("order"),
                "section": s.get("section_label"),
                "tag": tag,
                "category": category_of(tag),
                "text": s.get("text") or "",
                "spans": [x for x in (s.get("source_spans") or []) if isinstance(x, str)],
                "l2": l2 if isinstance(l2, dict) else None,
            })
    return pd.DataFrame(rows)


def sentence_key(row) -> str:
    return f"{row['article_id']}#{row['comment_index']}#{row['sent_id']}"


def key_parts(key: str) -> tuple[str, int, int]:
    a, c, s = key.split("#")
    return a, int(c), int(s)


# --- stlite build: appended by analysis/claim2_2/site/build.py -------------------------------
# The browser has no data/combined_data/ (579 MB); the build ships each article's sentence table
# as CSV instead, and COMBINED_DIR holds empty placeholders so globbing it still lists the corpus.
import webdata as _webdata  # noqa: E402

COMBINED_DIR = _webdata.CORPUS_DIR
_STR_COLS = {"article_id": str, "section": str, "tag": str, "text": str}


def load_article(article_id: str) -> dict:
    raise NotImplementedError("the raw corpus is not shipped with the web build")


def sentence_table(article_id: str) -> pd.DataFrame:
    _webdata.ensure(article_id)
    df = pd.read_csv(_webdata.SENT_DIR / f"{article_id}.csv", dtype=_STR_COLS,
                     keep_default_na=False, na_values={"section": [""], "tag": [""], "order": [""]})
    df["category"] = df["tag"].map(category_of)
    return df
