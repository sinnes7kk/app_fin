"""Render the Flask dashboard to a static HTML site for GitHub Pages deployment."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.web.server import app

SITE_DIR = Path("_site")


def build() -> None:
    SITE_DIR.mkdir(exist_ok=True)

    with app.test_client() as client:
        resp = client.get("/")
        html = resp.data.decode()

    # Rewrite the Jinja url_for CSS path to a relative static/ reference
    html = html.replace("/static/app.css", "static/app.css")

    (SITE_DIR / "index.html").write_text(html)

    static_src = Path(app.static_folder)
    static_dst = SITE_DIR / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    print(f"Static site built → {SITE_DIR}/")


if __name__ == "__main__":
    build()
