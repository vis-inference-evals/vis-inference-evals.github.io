"""Browser-side data loader for the stlite build of the claim 2.2 dashboard.

The static site ships the dashboard's inputs as JSON bundles under data/: one global bundle
(entity dictionaries, cross-article tables, figure tables, corpus ids) fetched at startup, and one
bundle per article (its out/<id>/ tables and its sentence table) fetched when it is first picked.
Each bundle is unpacked onto Pyodide's in-memory file system at the paths the unmodified analysis
code reads from, so common/entities/figures/dashboard run as they do locally.
"""
from __future__ import annotations

import json
from pathlib import Path

from pyodide.http import open_url

import webconfig  # written by index.html: BASE = absolute URL of data/

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "analysis" / "claim2_2"
CORPUS_DIR = ROOT / "corpus"          # empty <id>.json placeholders, globbed by common.ARTICLES
SENT_DIR = ROOT / "sentences"         # <id>.csv, read by the claim 3 shim
_done: set[str] = set()


def _get(rel: str) -> dict:
    return json.loads(open_url(f"{webconfig.BASE}{rel}?v={webconfig.VERSION}").read())


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def init() -> None:
    if (ROOT / ".webdata_ready").exists():
        return
    g = _get("global.json")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for aid in g["corpus"]:
        (CORPUS_DIR / f"{aid}.json").write_text("{}")
    for aid, text in g["dicts"].items():
        _write(PKG / "dicts" / f"{aid}_entities.json", text)
    for name, text in g["out"].items():
        _write(PKG / "out" / name, text)
    (ROOT / ".webdata_ready").write_text("")


def ensure(article_id: str) -> None:
    aid = str(article_id)
    if aid in _done or (SENT_DIR / f"{aid}.csv").exists():
        _done.add(aid)
        return
    b = _get(f"articles/{aid}.json")
    for name, text in b["out"].items():
        _write(PKG / "out" / aid / name, text)
    _write(SENT_DIR / f"{aid}.csv", b["sentences"])
    _done.add(aid)
