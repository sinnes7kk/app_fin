"""Render the Flask dashboard to a static HTML site for GitHub Pages deployment."""

from __future__ import annotations

import shutil
from pathlib import Path

import app.web.server as server_mod
from app.web.server import app

SITE_DIR = Path("_site")


def build() -> None:
    SITE_DIR.mkdir(exist_ok=True)

    # Show all items on a single page — no server-side pagination in the
    # static build (there is no server to handle page 2+ requests).
    _orig_page_size = server_mod.TABLE_PAGE_SIZE
    server_mod.TABLE_PAGE_SIZE = 9999

    try:
        with app.test_client() as client:
            resp = client.get("/")
            html = resp.data.decode()
    finally:
        server_mod.TABLE_PAGE_SIZE = _orig_page_size

    # Rewrite absolute paths to relative so the site works under a subpath
    # (e.g. GitHub Pages at /repo-name/).
    html = html.replace("/static/app.css", "static/app.css")
    html = html.replace('href="/?', 'href="?')

    (SITE_DIR / "index.html").write_text(html)

    static_src = Path(app.static_folder)
    static_dst = SITE_DIR / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    print(f"Static site built → {SITE_DIR}/")


if __name__ == "__main__":
    build()
