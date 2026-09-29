"""Per-article entity dictionary: worklist, validating loader, deterministic tokenizer.

The dictionary itself is an LLM-shaped coding step (see dicts/PROMPT_entities.md);
its output is committed data. Everything in this module is deterministic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from analysis.claim2_2 import common

REQUIRED_KEYS = ("id", "canonical", "type", "surface_forms")


def validate(groups: list[dict], corpus_text: str | None = None) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    owner: dict[str, str] = {}
    for i, g in enumerate(groups):
        where = f"group {i} ({g.get('id', '?')})"
        for k in REQUIRED_KEYS:
            if k not in g:
                errors.append(f"{where}: missing key {k}")
        gid = g.get("id")
        if not isinstance(gid, str) or not gid:
            errors.append(f"{where}: id must be a non-empty string")
        elif gid in seen_ids:
            errors.append(f"{where}: duplicate id {gid!r}")
        else:
            seen_ids.add(gid)
        if not g.get("canonical"):
            errors.append(f"{where}: missing canonical")
        if g.get("type") not in common.ENTITY_TYPES:
            errors.append(f"{where}: type {g.get('type')!r} not in {common.ENTITY_TYPES}")
        forms = g.get("surface_forms") or []
        if not isinstance(forms, list) or not forms:
            errors.append(f"{where}: surface_forms must be a non-empty list")
            continue
        for f in forms:
            if not isinstance(f, str) or not f.strip():
                errors.append(f"{where}: empty surface form")
                continue
            key = f.strip().lower()
            if key in owner and owner[key] != gid:
                errors.append(f"{where}: surface form {f!r} already claimed by {owner[key]!r}")
            else:
                owner[key] = gid
            if corpus_text is not None and key not in corpus_text:
                errors.append(f"{where}: surface form {f!r} matches nothing in the article")
    return errors


def load(article_id: str, check_corpus: bool = True) -> list[dict]:
    path = Path(common.DICT_DIR) / f"{article_id}_entities.json"
    groups = json.loads(path.read_text(encoding="utf-8"))
    corpus = None
    if check_corpus:
        corpus = " \n ".join(common.sentences(article_id)["text"].astype(str)).lower()
    errors = validate(groups, corpus_text=corpus)
    if errors:
        raise ValueError(f"{path.name}: {len(errors)} errors\n" + "\n".join(errors[:50]))
    return groups


def surface_index(groups: list[dict]) -> list[tuple[str, str]]:
    """(surface form, entity id) pairs, longest form first so longest match wins."""
    pairs = [(f.strip().lower(), g["id"]) for g in groups for f in g["surface_forms"]]
    return sorted(pairs, key=lambda p: (-len(p[0]), p[0]))


WORD = re.compile(r"[a-z0-9]")


def _boundary_ok(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not WORD.match(before) and not WORD.match(after)


def tokenize_text(text: str, index: list[tuple[str, str]]) -> list[dict]:
    """Non-overlapping longest matches on word boundaries, left to right."""
    low = text.lower()
    out, pos = [], 0
    while pos < len(low):
        best = None
        for form, gid in index:                      # index is longest-first
            if low.startswith(form, pos) and _boundary_ok(low, pos, pos + len(form)):
                best = (form, gid)
                break
        if best is not None:
            form, gid = best
            out.append({"entity_id": gid, "span_start": pos,
                        "span_end": pos + len(form), "surface": low[pos:pos + len(form)]})
            pos += len(form)
        else:
            pos += 1
    return out


def mentions(article_id: str) -> pd.DataFrame:
    groups = load(article_id)
    types = {g["id"]: g["type"] for g in groups}
    index = surface_index(groups)
    sents = common.sentences(article_id)
    rows = []
    for _, r in sents.iterrows():
        for m in tokenize_text(str(r["text"]), index):
            rows.append({"key": r["key"], "comment_index": int(r["comment_index"]),
                         "sent_id": int(r["sent_id"]), "category": r["category"],
                         "entity_id": m["entity_id"], "type": types[m["entity_id"]],
                         "span_start": m["span_start"], "span_end": m["span_end"],
                         "surface": m["surface"]})
    cols = ["key", "comment_index", "sent_id", "category", "entity_id", "type",
            "span_start", "span_end", "surface"]
    return pd.DataFrame(rows, columns=cols)


def coverage(m: pd.DataFrame, sents: pd.DataFrame) -> dict:
    hit = set(m["key"])
    out = {"overall": float(sents["key"].isin(hit).mean()) if len(sents) else 0.0}
    for cat in common.CATEGORIES + [common.REDUNDANT]:
        sel = sents[sents["category"] == cat]
        out[cat] = float(sel["key"].isin(hit).mean()) if len(sel) else 0.0
    return out


def write_mentions(article_id: str) -> Path:
    m = mentions(article_id)
    path = common.out_dir(article_id) / "mentions.csv"
    m.to_csv(path, index=False)
    return path


def dump_worklist(article_id: str) -> Path:
    df = common.sentences(article_id)
    rows = [{"key": r["key"], "comment_index": int(r["comment_index"]),
             "sent_id": int(r["sent_id"]), "category": r["category"],
             "tag": r["tag"], "text": r["text"]} for _, r in df.iterrows()]
    common.WORKLIST_DIR.mkdir(parents=True, exist_ok=True)
    path = common.WORKLIST_DIR / f"{article_id}_sentences.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    aid = sys.argv[1] if len(sys.argv) > 1 else "181"
    if "--worklist" in sys.argv:
        print(dump_worklist(aid))
    else:
        path = write_mentions(aid)
        m = pd.read_csv(path)
        print(path, len(m), "mentions;", coverage(m, common.sentences(aid)))
