from core.types import Chunk
from ingest.chunk_tagger import rule_chunk_type, tag_with_rules
from ingest.chunker import chunk_document, chunk_page
from ingest.parsers.pdf_tables import clean_table, table_to_texts
from ingest.parsers.pdf_text import clean_page_text
from ingest.vocab_builder import add_metadata, build_vocab, find_section, years_in


def words(text):
    return len(text.split())


def test_chunk_page_respects_size_overlap_and_page():
    text = "\n".join(f"Line {i} has five words." for i in range(40))  # 40 lines x 5 words
    chunks = chunk_page("doc", 7, text, words, target_tokens=50, overlap_tokens=10, min_tokens=5)
    assert len(chunks) > 1
    assert all(c.page_start == c.page_end == 7 for c in chunks)
    assert all(words(c.text) <= 50 for c in chunks)
    first, second = chunks[0].text.split("\n"), chunks[1].text.split("\n")
    assert first[-2:] == second[:2]  # 10 words of overlap = the last two lines
    assert chunks[0].id == "doc:p7:c0"


def test_chunk_document_skips_blank_and_tiny_pages():
    pages = ["", "12", "\n".join(["A real sentence with several words in it."] * 10)]
    chunks = chunk_document("d", pages, words, target_tokens=100, overlap_tokens=0, min_tokens=5)
    assert {c.page_start for c in chunks} == {3}


def test_long_line_is_split():
    line = " ".join(["word"] * 300)
    chunks = chunk_page("d", 1, line, words, target_tokens=100, overlap_tokens=0, min_tokens=5)
    assert all(words(c.text) <= 100 for c in chunks) and len(chunks) == 3


def test_clean_page_text():
    assert clean_page_text("a  b\r\n\r\n\r\n\r\nc\x00") == "a b\n\nc"


def test_tables_are_cleaned_and_split_with_header():
    table = clean_table([["Year", None, "Sales"], ["2018", "", "10"], [None, None, None], ["2017", "", "9"]])
    assert table == [["Year", "Sales"], ["2018", "10"], ["2017", "9"]]
    pieces = table_to_texts(table, max_tokens=4, count_tokens=words)
    assert pieces == ["Year | Sales\n2018 | 10", "Year | Sales\n2017 | 9"]


def test_rule_tagger():
    assert rule_chunk_type("Net sales | 2018 | 2017\nA | 1 | 2") == "table"
    assert rule_chunk_type("Free cash flow is defined as operating cash flow less capital spending.") == "definition"
    assert rule_chunk_type("Sales rose 5% to 1,200 and margin was 21.4% while EPS was 9.10 in total.") == "numeric"
    assert rule_chunk_type("We continue to invest in our people and our communities.") == "narrative"
    tagged = tag_with_rules([Chunk("a", "d", "x", 1, 1, chunk_type="table"), Chunk("b", "d", "We grow.", 1, 1)])
    assert [c.chunk_type for c in tagged] == ["table", "narrative"]


def test_years_and_sections():
    assert years_in("In FY2018 and 2017, not 12018") == ["2017", "2018"]
    patterns = [r"^\s*(item\s+\d{1,2}[a-c]?)\b"]
    from ingest.vocab_builder import compile_section_patterns

    assert find_section("text\nItem 7. Management's Discussion\nmore", compile_section_patterns(patterns)) == "item 7"


def test_metadata_carries_section_forward():
    chunks = [Chunk("c0", "d", "Item 7. MD&A\nSales in 2018 rose.", 1, 1), Chunk("c1", "d", "Costs fell.", 2, 2)]
    tagged = add_metadata(chunks, {"d": {"company": "Acme", "year": 2018}}, [r"^\s*(item\s+\d{1,2}[a-c]?)\b"])
    assert tagged[0].metadata["section"] == "item 7" and tagged[1].metadata["section"] == "item 7"
    assert tagged[0].metadata["years"] == "2018" and tagged[1].metadata["company"] == "Acme"
    vocab = build_vocab(tagged, {"d": {"company": "Acme", "year": 2018}})
    assert vocab["doc_fields"]["company"] == ["Acme"] and vocab["years"] == {"2018": 1}
