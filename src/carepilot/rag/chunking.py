"""Heading-aware chunking of the cleaned Markdown documents.

Why heading-aware: a passage like "- are age 35 or older" is meaningless without the
heading "What are the risk factors for type 2 diabetes?". Each chunk keeps its heading
path, and the embedded text is prefixed with "title > section" so retrieval sees context.

Algorithm: split a document into sections at Markdown headings, split each section into
blocks (paragraphs / bullet runs), then greedily pack blocks into chunks of at most
MAX_CHARS. The last block of a chunk is repeated at the start of the next one when it is
short enough (overlap), so a sentence split across a boundary is still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 900
OVERLAP_MAX_CHARS = 250
MIN_CHUNK_CHARS = 40

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    publisher: str
    url: str
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        """Text that is embedded and BM25-indexed: context header + body."""
        return (
            f"{self.title} > {self.section}\n{self.text}"
            if self.section
            else f"{self.title}\n{self.text}"
        )


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(heading path, body)], ignoring the top-level document title (h1)."""
    path: list[str] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path), text))
        body.clear()

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if not m:
            body.append(line)
            continue
        flush()
        level = len(m.group(1))
        if level == 1:
            path = []
            continue
        del path[level - 2 :]  # h2 -> depth 0, h3 -> depth 1, h4 -> depth 2
        path.append(m.group(2).strip())
    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines; consecutive bullet lines stay one block."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _split_long(block: str, limit: int) -> list[str]:
    """Break an oversized block at sentence or bullet boundaries."""
    units = re.split(r"(?<=[.!?])\s+|\n(?=- )", block)
    parts: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > limit:
            parts.append(cur)
            cur = u
        else:
            cur = f"{cur} {u}".strip() if cur else u
    if cur:
        parts.append(cur)
    return parts


def pack(blocks: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    units: list[str] = []
    for b in blocks:
        units.extend(_split_long(b, max_chars) if len(b) > max_chars else [b])
    chunks: list[str] = []
    cur: list[str] = []
    for u in units:
        if cur and sum(len(x) + 2 for x in cur) + len(u) > max_chars:
            chunks.append("\n\n".join(cur))
            tail = cur[-1]
            cur = [tail] if len(tail) <= OVERLAP_MAX_CHARS else []
        cur.append(u)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(doc_id: str, title: str, publisher: str, url: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    markdown = re.sub(r"^Source: .*$", "", markdown, flags=re.MULTILINE)  # kept as metadata
    for section, body in split_sections(markdown):
        for text in pack(split_blocks(body)):
            if len(text) < MIN_CHUNK_CHARS and not text.startswith("- "):
                continue  # stray lines carry no answerable content
            if text.lower().startswith("in this section"):
                continue  # page table of contents, duplicates the headings
            chunks.append(
                Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, publisher, url, section, text)
            )
    return chunks
