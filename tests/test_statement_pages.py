from core.config import load_config
from core.types import Chunk
from query.statement_pages import build_statement_index, question_kinds, reserve_slots

TITLES = {
    "balance_sheet": r"(consolidated\s+)?balance\s+sheets?",
    "cash_flow": r"(consolidated\s+)?statements?\s+of\s+cash\s+flows?",
}
NUMBERS = " ".join(f"{n},{n:03d}" for n in range(100, 120))  # 20 numbers, like a statement table


def _chunk(chunk_id: str, text: str, company: str = "Acme", year: str = "2022") -> Chunk:
    page = int(chunk_id.split(":p")[1].split(":")[0])
    return Chunk(id=chunk_id, doc_id=chunk_id.split(":")[0], text=text, page_start=page, page_end=page,
                 metadata={"company": company, "year": year})


def test_statement_pages_are_found_from_their_title_and_numbers():
    chunks = [
        _chunk("ACME_2022:p50:c0", f"Table of Contents\nConsolidated Balance Sheets\nAssets\n{NUMBERS}"),
        _chunk("ACME_2022:p50:c1", f"Consolidated Balance Sheets\n{NUMBERS}"),  # not the first chunk of the page
        _chunk("ACME_2022:p51:c0", f"Notes\nThe consolidated balance sheets show {NUMBERS}"),  # title not on its own line
        _chunk("ACME_2022:p52:c0", "Consolidated Statements of Cash Flows\nsee page 60"),  # too few numbers
        _chunk("ACME_2022:p60:c0", f"CONSOLIDATED STATEMENTS OF CASH FLOWS\n{NUMBERS}"),
    ]
    index = build_statement_index(chunks, TITLES, min_numbers=15)
    assert [c.id for c in index.lookup("Acme", "2022", ["balance_sheet"])] == ["ACME_2022:p50:c0"]
    assert [c.id for c in index.lookup("Acme", "2022", ["cash_flow", "balance_sheet"])] == ["ACME_2022:p60:c0", "ACME_2022:p50:c0"]
    assert index.lookup("Acme", "2021", ["balance_sheet"]) == []


def test_question_kinds_come_from_keyword_starts():
    keywords = {"balance_sheet": ["quick ratio", "inventor"], "cash_flow": ["capex"]}
    assert question_kinds("What is the FY2022 quick ratio for Acme?", keywords) == ["balance_sheet"]
    assert question_kinds("How did inventory turnover and CAPEX change?", keywords) == ["balance_sheet", "cash_flow"]
    assert question_kinds("Who is the CEO?", keywords) == []
    assert question_kinds("Is the inventory of risks long?", {"cash_flow": ["capex"]}) == []


def test_reserved_slots_keep_the_best_ranked_chunks_first():
    ranked = [_chunk(f"A:p{n}:c0", "x") for n in range(1, 8)]
    extra = [_chunk("A:p50:c0", "x"), ranked[4]]  # the second one is already ranked 5th
    final = reserve_slots(ranked, extra, keep=3, top_k=5)
    assert [c.id for c in final] == ["A:p1:c0", "A:p2:c0", "A:p3:c0", "A:p50:c0", "A:p5:c0"]
    assert [c.id for c in reserve_slots(ranked, [], keep=3, top_k=5)] == [c.id for c in ranked[:5]]


def test_financebench_pack_titles_and_keywords_load():
    settings = load_config("financebench")["statements"]
    assert set(settings["titles"]) == set(settings["question_keywords"]) == {"balance_sheet", "cash_flow", "income"}
    assert settings["keep_ranked"] + settings["slots"] <= load_config("financebench")["reranker"]["top_k"]
