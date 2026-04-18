"""Render the Flask dashboard to a static HTML site for GitHub Pages deployment.

Every entry in :data:`app.config.FLOW_TRACKER_HORIZONS` is rendered into its
own physical file so the horizon toggle works without a backend — the default
horizon becomes ``index.html`` and every other horizon becomes
``horizon-<key>.html``.  Horizon hrefs in the output HTML are rewritten
accordingly so clicking ``15d`` on GitHub Pages navigates to the pre-rendered
15d page instead of reloading the same default-horizon file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import app.web.server as server_mod
from app.config import FLOW_TRACKER_HORIZON_DEFAULT, FLOW_TRACKER_HORIZONS
from app.web.server import app

SITE_DIR = Path("_site")


def _horizon_filename(key: str) -> str:
    """Default horizon is served from ``index.html``; others get their own file."""
    if key == FLOW_TRACKER_HORIZON_DEFAULT:
        return "index.html"
    return f"horizon-{key}.html"


def _rewrite_horizon_links(html: str) -> str:
    """Point ``?horizon=X&tab=flow-tracker`` hrefs at their static-file targets."""
    for key in FLOW_TRACKER_HORIZONS:
        target = _horizon_filename(key)
        html = html.replace(
            f'href="?horizon={key}&tab=flow-tracker"',
            f'href="{target}?tab=flow-tracker"',
        )
    return html


def build() -> None:
    SITE_DIR.mkdir(exist_ok=True)

    _orig_page_size = server_mod.TABLE_PAGE_SIZE
    server_mod.TABLE_PAGE_SIZE = 9999

    try:
        with app.test_client() as client:
            for key in FLOW_TRACKER_HORIZONS:
                resp = client.get(f"/?horizon={key}")
                html = resp.data.decode()
                html = html.replace("/static/app.css", "static/app.css")
                html = html.replace('href="/?', 'href="?')
                html = _rewrite_horizon_links(html)
                out_path = SITE_DIR / _horizon_filename(key)
                out_path.write_text(html)
    finally:
        server_mod.TABLE_PAGE_SIZE = _orig_page_size

    static_src = Path(app.static_folder)
    static_dst = SITE_DIR / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    horizons = ", ".join(FLOW_TRACKER_HORIZONS)
    print(f"Static site built → {SITE_DIR}/ (horizons: {horizons})")


if __name__ == "__main__":
    build()
