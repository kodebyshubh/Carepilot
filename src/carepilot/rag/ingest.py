"""Fetch public health web pages, extract the main article text, clean it to Markdown.

Usage: python -m carepilot.rag.ingest [--offline]
  --offline  re-clean from cached raw HTML in data/docs/raw without network.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import requests
from bs4 import BeautifulSoup, Tag

from carepilot.config import DOCS_CLEAN_DIR, DOCS_RAW_DIR

log = logging.getLogger(__name__)
TIMEOUT_S = 30
USER_AGENT = "Mozilla/5.0 (compatible; CarePilot-demo/0.1; educational prototype)"

NIH_LICENSE = "US Government work, public domain (NIH)"
WHO_LICENSE = "WHO fact sheet, CC BY-NC-SA 3.0 IGO (non-commercial use with attribution)"


@dataclass(frozen=True)
class DocSource:
    doc_id: str
    title: str
    publisher: str
    url: str
    license: str
    topics: tuple[str, ...]


SOURCES: tuple[DocSource, ...] = (
    DocSource(
        "niddk_diabetes_symptoms",
        "Symptoms & Causes of Diabetes",
        "NIDDK (NIH)",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes",
        NIH_LICENSE,
        ("diabetes", "symptoms"),
    ),
    DocSource(
        "niddk_t2d_risk_factors",
        "Risk Factors for Type 2 Diabetes",
        "NIDDK (NIH)",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes",
        NIH_LICENSE,
        ("diabetes", "risk factors"),
    ),
    DocSource(
        "niddk_t2d_prevention",
        "Preventing Type 2 Diabetes",
        "NIDDK (NIH)",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-type-2-diabetes",
        NIH_LICENSE,
        ("diabetes", "prevention", "lifestyle"),
    ),
    DocSource(
        "niddk_managing_diabetes",
        "Managing Diabetes",
        "NIDDK (NIH)",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes",
        NIH_LICENSE,
        ("diabetes", "management", "lifestyle"),
    ),
    DocSource(
        "nhlbi_hbp_causes",
        "High Blood Pressure: Causes and Risk Factors",
        "NHLBI (NIH)",
        "https://www.nhlbi.nih.gov/health/high-blood-pressure/causes",
        NIH_LICENSE,
        ("hypertension", "risk factors"),
    ),
    DocSource(
        "nhlbi_hbp_treatment",
        "High Blood Pressure: Treatment",
        "NHLBI (NIH)",
        "https://www.nhlbi.nih.gov/health/high-blood-pressure/treatment",
        NIH_LICENSE,
        ("hypertension", "treatment", "lifestyle"),
    ),
    DocSource(
        "nhlbi_chd_risk_factors",
        "Coronary Heart Disease: Risk Factors",
        "NHLBI (NIH)",
        "https://www.nhlbi.nih.gov/health/coronary-heart-disease/risk-factors",
        NIH_LICENSE,
        ("heart disease", "risk factors"),
    ),
    DocSource(
        "nhlbi_heart_attack_symptoms",
        "Heart Attack: Symptoms",
        "NHLBI (NIH)",
        "https://www.nhlbi.nih.gov/health/heart-attack/symptoms",
        NIH_LICENSE,
        ("heart disease", "symptoms", "emergency"),
    ),
    DocSource(
        "who_hypertension",
        "Hypertension (fact sheet)",
        "World Health Organization",
        "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        WHO_LICENSE,
        ("hypertension",),
    ),
    DocSource(
        "who_diabetes",
        "Diabetes (fact sheet)",
        "World Health Organization",
        "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        WHO_LICENSE,
        ("diabetes",),
    ),
)

# Elements that never carry article content.
DROP_TAGS = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "noscript",
    "svg",
    "button",
    "iframe",
    "figure",
)
# Boilerplate lines seen on NIH / WHO pages.
BOILERPLATE = re.compile(
    r"^(share|print|email|on this page|related|last reviewed|related diabetes topics|"
    r"this content is provided as a service|clinical trials|was this page helpful)",
    re.IGNORECASE,
)
LANGUAGE_SWITCHER = frozenset({"Language switcher", "English", "Español"})
TRAILER = re.compile(
    r"\n#{1,4} (References|Clinical Trials|Related Topics|WHO response)\b.*", re.DOTALL
)


def fetch_html(src: DocSource) -> str:
    resp = requests.get(src.url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def _main_container(soup: BeautifulSoup) -> Tag:
    """Pick the element holding the article body, most specific first."""
    for selector in ("div.sf-detail-body-wrapper", "article", "main", "div#main-content", "body"):
        node = soup.select_one(selector)
        if node is not None and len(node.get_text(strip=True)) > 500:
            return node
    raise ValueError("no main content found")


def html_to_markdown(html: str) -> str:
    """Convert article HTML to light Markdown: headings, paragraphs, bullet items."""
    soup = BeautifulSoup(html, "html.parser")
    root = _main_container(soup)
    for tag in root.find_all(DROP_TAGS):
        tag.decompose()

    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        # Skip list items / paragraphs nested in another captured block to avoid duplicates.
        if el.name in ("p", "li") and el.find_parent("li") is not None:
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text or BOILERPLATE.match(text) or text in LANGUAGE_SWITCHER:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[el.name]
            lines.append(f"\n{level} {text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(f"\n{text}\n")
    md = "\n".join(lines)
    # Drop trailing back-matter (references, acknowledgements, trial listings).
    md = TRAILER.split(md, maxsplit=1)[0]
    md = re.sub(r"(?<=[.:;)])\s+\d{1,2}(?:,\s*\d{1,2})*(?=\s|$)", "", md)  # footnote markers
    md = re.sub(r"(?<=[a-z])\s\d{1,2}$", "", md, flags=re.MULTILINE)  # "if you 1"
    md = re.sub(r"\s+([.,;:])", r"\1", md)  # "obesity ." left by inline links
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DOCS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for src in SOURCES:
        raw_path = DOCS_RAW_DIR / f"{src.doc_id}.html"
        if args.offline:
            html = raw_path.read_text(encoding="utf-8")
        else:
            html = fetch_html(src)
            raw_path.write_text(html, encoding="utf-8")
        md = html_to_markdown(html)
        md = re.sub(rf"^# {re.escape(src.title)}\s*\n", "", md)  # page h1 duplicates our header
        header = f"# {src.title}\n\nSource: {src.publisher}, {src.url}\n\n"
        (DOCS_CLEAN_DIR / f"{src.doc_id}.md").write_text(header + md, encoding="utf-8")
        manifest.append(
            {
                **asdict(src),
                "chars": len(md),
                "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        log.info("%-30s %6d chars", src.doc_id, len(md))
    (DOCS_CLEAN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
