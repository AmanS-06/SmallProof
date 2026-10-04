from smallproof.core.types import Chunk, Citation
from smallproof.generate.citations import check_citations, extract_cited_pages


def test_extracts_common_citation_formats():
    text = "Revenue was 32.8B [p. 12]. Margin fell [pp. 14-15] and [page 20], see also (p.3) and [P 12]."
    assert extract_cited_pages(text) == [12, 14, 15, 20, 3]


def test_extracts_lists_and_en_dash_ranges():
    assert extract_cited_pages("[pages 4, 6] [pp. 8\u20139]") == [4, 6, 8, 9]


def test_ignores_malformed_citations():
    assert extract_cited_pages("[p. abc] [pp. 9-7] [pp. 1-500] no citation here") == []


def test_check_citations_splits_valid_and_invalid():
    chunks = [Chunk("c1", "docA", "x", 12, 13), Chunk("c2", "docB", "y", 20, 20)]
    result = check_citations("A [p. 12], B [p. 20], C [p. 99].", chunks)
    assert result.citations == [Citation("docA", 12), Citation("docB", 20)]
    assert result.invalid_pages == [99]
    assert not result.all_valid


def test_page_covered_by_two_documents_links_both():
    chunks = [Chunk("c1", "docB", "x", 5, 5), Chunk("c2", "docA", "y", 4, 6)]
    result = check_citations("[p. 5]", chunks)
    assert result.citations == [Citation("docA", 5), Citation("docB", 5)]
    assert result.all_valid
