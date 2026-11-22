from carepilot.rag.chunking import MAX_CHARS, chunk_document, pack, split_sections

DOC = """# Title

Source: NIH, https://example.org

## What are the symptoms?

Common symptoms include

- thirst
- fatigue

### Type 1

Type 1 symptoms can start quickly, within weeks, and may be severe.

## Other

In this section:

- a
- b
"""


def test_sections_keep_heading_path_and_skip_h1():
    sections = dict(split_sections(DOC))
    assert "What are the symptoms? > Type 1" in sections
    assert "Title" not in " ".join(sections)


def test_chunks_carry_metadata_and_skip_source_line_and_toc():
    chunks = chunk_document("d1", "Title", "NIH", "https://example.org", DOC)
    texts = " ".join(c.text for c in chunks)
    assert "Source:" not in texts
    assert "In this section" not in texts
    assert all(c.doc_id == "d1" and c.url == "https://example.org" for c in chunks)
    assert chunks[0].chunk_id == "d1#0"


def test_embed_text_prefixes_context():
    c = chunk_document("d1", "Title", "NIH", "u", DOC)[0]
    assert c.embed_text.startswith("Title > What are the symptoms?")


def test_pack_respects_max_and_overlaps_short_tail():
    blocks = [f"Paragraph {i}. " + "word " * 40 for i in range(10)]
    chunks = pack(blocks)
    assert all(len(c) <= MAX_CHARS + 260 for c in chunks)  # max + one overlapped block
    assert len(chunks) > 1
    # last block of chunk i reappears at start of chunk i+1
    assert chunks[1].startswith(chunks[0].split("\n\n")[-1])


def test_oversized_block_split_at_sentences():
    long_block = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = pack([long_block])
    assert len(chunks) > 1
    assert all(c.rstrip().endswith(".") for c in chunks)
