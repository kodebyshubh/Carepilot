from carepilot.rag.ingest import html_to_markdown

HTML = (
    """
<html><body>
<nav>Home | About</nav>
<main>
  <h1>Diabetes</h1>
  <p>You are more likely to develop diabetes if you are overweight. 1</p>
  <p>Risk rises with <a href="#">obesity</a> .</p>
  <ul><li>are age 35 or older</li><li>have a family history 2</li></ul>
  <h2>Share</h2>
  <script>alert('x')</script>
  <p>"""
    + "Filler sentence to make the main block long enough. " * 12
    + """</p>
  <h2>References</h2>
  <p>Some citation list.</p>
</main>
<footer>Contact us</footer>
</body></html>
"""
)


def test_keeps_headings_paragraphs_and_bullets():
    md = html_to_markdown(HTML)
    assert "# Diabetes" in md
    assert "- are age 35 or older" in md


def test_drops_nav_footer_script_and_boilerplate():
    md = html_to_markdown(HTML)
    for junk in ("Home | About", "Contact us", "alert", "## Share"):
        assert junk not in md


def test_strips_footnote_markers_and_link_spacing():
    md = html_to_markdown(HTML)
    assert "overweight. 1" not in md and "overweight." in md
    assert "family history 2" not in md
    assert "obesity." in md


def test_cuts_references_section():
    md = html_to_markdown(HTML)
    assert "citation list" not in md
