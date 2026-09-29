"""Loading and shared paths for the claim 2.2 entity-chain analysis (one article at a time)."""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pandas as pd

from analysis.claim1.boilerplate import is_boilerplate
from analysis.claim3 import common as c3

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
DICT_DIR = PKG_DIR / "dicts"
WORKLIST_DIR = PKG_DIR / "worklists"
OUT_DIR = PKG_DIR / "out"

# Every article in the corpus is now coded. The comment cap that earlier versions of this
# analysis used (300, then 500, then 800, then 1200 comments) existed only to hold back article
# 41, which at 6,347 comments is roughly eight times the largest other article and was left for a
# pass of its own. That pass is done, so the cap is gone rather than raised: the article set is
# simply the whole corpus minus the articles EXCLUDED below for cause.
PILOTS = ["181", "183", "91"]

# Articles kept out of the analysis, with the reason. Article 125 (a faceted heatmap of Olympic
# medal rankings) carries article 124's comments: the two share 177 of 177 comments, and those
# comments mention "wedding" 726 times and "medal"/"olympic"/"gold"/"silver"/"bronze" zero times.
# 124 is the wedding chart, so the comments are its; 125's own comments are absent from the
# corpus, and any dictionary or metric built for 125 would describe weddings.
EXCLUDED = {
    "125": "carries article 124's wedding comments; its own comments are absent from the corpus",
}


def _corpus_articles() -> list[str]:
    """Every article in the corpus that is not EXCLUDED for cause, numerically sorted."""
    ids = [path.stem for path in c3.COMBINED_DIR.glob("*.json") if path.stem not in EXCLUDED]
    return sorted(ids, key=int)


ARTICLES = _corpus_articles()


def coded_articles() -> list[str]:
    """The articles that already have a committed entity dictionary."""
    return [a for a in ARTICLES if (DICT_DIR / f"{a}_entities.json").exists()]


CATEGORIES = list(c3.CATEGORIES)
REDUNDANT = c3.REDUNDANT
ENTITY_TYPES = ("entity", "data_variable", "value", "event")


def category_of(tag: str) -> str:
    """Claim 3's coarse category, except that "Curiosity" is INF. Claim 1 files it (plotted as
    "Open-ended questions") under Inference; claim 3's mapping never mentions it, so it would
    otherwise land in redundant. Scoped to claim 2.2 -- claim 3 keeps its own mapping."""
    return "INF" if tag == "Curiosity" else c3.category_of(tag)


def sentences(article_id: str) -> pd.DataFrame:
    """Claim 3's sentence table for one article, plus `key`, in reading order, with `category`
    re-derived by this module's `category_of`."""
    df = c3.sentence_table(str(article_id)).copy()
    df["category"] = df["tag"].map(category_of)
    df["key"] = [c3.sentence_key(r) for _, r in df.iterrows()]
    df["boilerplate"] = df["text"].astype(str).map(is_boilerplate)
    df = df.sort_values(["comment_index", "order", "sent_id"], kind="stable")
    return df.reset_index(drop=True)


def out_dir(article_id: str) -> Path:
    p = OUT_DIR / str(article_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


_COMMUNITY_WORD = re.compile(r"\bcommunit(?:y|ies)\b", re.IGNORECASE)


@functools.lru_cache(maxsize=None)
def prompt_variant(article_id: str) -> str:
    """'community_asked' or 'community_not_asked' -- whether readers were invited to say how the
    chart relates to them and their community. The instrument varied across articles, so this is
    the natural control for prompt-driven entity linkage.

    Derived from the corpus: community-word ("community"/"communities") mentions per comment.
    Zero mentions across the article means the prompt did not ask; the rollout found this split
    sharply bimodal (28 articles at zero, 86 at 0.5+ per comment), so no threshold tuning is
    needed between those two clusters.
    """
    df = sentences(article_id)
    n_comments = df["comment_index"].nunique()
    if n_comments == 0:
        return "community_not_asked"
    n_mentions = sum(len(_COMMUNITY_WORD.findall(str(t))) for t in df["text"])
    return "community_not_asked" if n_mentions == 0 else "community_asked"
