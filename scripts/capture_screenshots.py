"""Capture README screenshots from a running app.

Usage:
  uv run streamlit run app/streamlit_app.py &      # or the deployed URL
  uv run python scripts/capture_screenshots.py [http://localhost:8501] [provider]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import expect, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
LONG = 240_000
expect.set_options(timeout=LONG)


SIDEBAR_PX = 310
MIN_HEIGHT = 800


def trim_bottom(path: Path) -> None:
    """Cut the empty space under the main content (the viewport is taller than the page)."""
    im = Image.open(path).convert("RGB")
    content = im.crop((SIDEBAR_PX, 0, im.width, im.height))
    bbox = ImageChops.difference(content, Image.new("RGB", content.size, "white")).getbbox()
    bottom = max(MIN_HEIGHT, min(im.height, (bbox[3] if bbox else im.height) + 40))
    im.crop((0, 0, im.width, bottom)).save(path, optimize=True)


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    provider = sys.argv[2] if len(sys.argv) > 2 else "auto"
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 2000})
        page.set_default_timeout(LONG)
        idle = page.get_by_test_id("stStatusWidget")
        page.goto(url)
        expect(page.get_by_text("Using:")).to_be_visible()
        page.get_by_test_id("stSidebar").get_by_test_id("stSelectbox").click()
        page.get_by_role("option", name=provider, exact=True).click()
        expect(idle).to_have_count(0)

        main_area = page.get_by_test_id("stMainBlockContainer")
        main_area.get_by_test_id("stSelectbox").first.click()
        page.get_by_role("option", name="High-risk example").click()
        expect(idle).to_have_count(0)
        main_area.get_by_role("button", name="Estimate risk").click()
        expect(main_area.get_by_text("Sources", exact=True)).to_be_visible()
        expect(idle).to_have_count(0)
        page.screenshot(path=OUT / "risk.png", full_page=True)

        page.get_by_role("tab", name="Ask the documents").click()
        main_area.get_by_role("button", name="What are the symptoms of diabetes?").click()
        # .last: the hidden Risk tab also holds a "Sources" header
        expect(main_area.get_by_text("Sources", exact=True).last).to_be_visible()
        expect(idle).to_have_count(0)
        main_area.get_by_test_id("stExpander").last.click()
        page.screenshot(path=OUT / "ask.png", full_page=True)

        main_area.get_by_label("Your question").fill(
            "I have chest pain. What is the right dose of metformin for me?"
        )
        main_area.get_by_role("button", name="Ask", exact=True).click()
        expect(main_area.get_by_text("not available in the provided documents")).to_be_visible()
        page.screenshot(path=OUT / "abstain.png", full_page=True)

        page.get_by_role("tab", name="Model & RAG report").click()
        expect(main_area.get_by_test_id("stImage").first).to_be_visible()
        page.screenshot(path=OUT / "report.png", full_page=True)
        browser.close()
    for shot in OUT.glob("*.png"):
        trim_bottom(shot)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
