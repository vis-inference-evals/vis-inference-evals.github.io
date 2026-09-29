"""stlite entrypoint: fetch the global data bundle, route per-article reads through the lazy
loader, then run the unmodified dashboard script."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import webdata  # noqa: E402

webdata.init()

from analysis.claim2_2 import common  # noqa: E402

if not getattr(common.out_dir, "_web", False):
    _out_dir = common.out_dir

    def out_dir(article_id):
        webdata.ensure(article_id)
        return _out_dir(article_id)

    out_dir._web = True
    common.out_dir = out_dir

_dash = ROOT / "analysis" / "claim2_2" / "dashboard.py"
exec(compile(_dash.read_text(encoding="utf-8"), str(_dash), "exec"),
     {"__file__": str(_dash), "__name__": "__main__"})
