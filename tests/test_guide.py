"""Every tool has a guide with interpretation help, documented options and sources."""
from fiasuite import guide
from fiasuite.analyses import online
from fiasuite.tools import TOOLS


def test_every_tool_has_a_guide():
    ids = [t.id for t in TOOLS] + [f"online:{name}" for name in online.PROVIDERS]
    assert sorted(ids) == sorted(guide.GUIDE)
    for tool_id in ids:
        entry = guide.GUIDE[tool_id]
        assert entry["what"] and entry["read"] and entry["sources"], tool_id
        assert all(url.startswith("https://") for _, url in entry["sources"])


def test_every_option_is_explained():
    for tool in TOOLS:
        documented = guide.GUIDE[tool.id].get("options", {})
        for param in tool.params:
            assert any(param.label in key or key in param.label for key in documented), (tool.id, param.label)


def test_render_links_and_escapes():
    page = guide.render("ghosts", "JPEG ghosts", ["Quality step"])
    assert '<a href="https://doi.org/10.1109/TIFS.2008.2012215">' in page and "How to read it" in page
    assert "<script" not in guide.render("x", "<script>", [])


def test_every_tool_has_example_cases_and_their_pictures_exist():
    from fiasuite.guide_cases import CASES
    for tool in TOOLS:
        assert CASES.get(tool.id), tool.id
    for tool_id, cases in CASES.items():
        for name, text, picture in cases:
            assert name and text
            assert picture is None or (guide.EXAMPLES / f"{picture}.jpg").exists(), (tool_id, picture)
    assert 'src="ela_splice.jpg"' in guide.render("ela", "Error level analysis", [])
