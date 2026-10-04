from smallproof.core.types import Chunk
from smallproof.generate.ollama_backend import format_passages
from smallproof.generate.table_years import annotate_table_years

INCOME = """Consolidated Statements of Operations
Year Ended
December 26,
2015
December 27,
2014
(In millions)
Net revenue $ 3,991 $ 5,506
Cost of sales 2,911 3,667
Goodwill impairment charge — 233
Operating income (loss) (481) (155)
Notes due 2025 1,500 1,500"""


def test_values_get_the_year_of_their_column():
    lines = annotate_table_years(INCOME).splitlines()
    assert "Net revenue (2015: 3,991; 2014: 5,506)" in lines
    assert "Cost of sales (2015: 2,911; 2014: 3,667)" in lines
    assert "Goodwill impairment charge (2015: —; 2014: 233)" in lines
    assert "Operating income (loss) (2015: -481; 2014: -155)" in lines
    assert "Notes due 2025 (2015: 1,500; 2014: 1,500)" in lines  # a year inside the label stays put
    assert lines[:7] == INCOME.splitlines()[:7]  # the header is unchanged


def test_a_header_on_one_line_is_not_a_row():
    text = "Balance Sheets\nMay 31, 2020 May 26, 2019\nReceivables 1,615.1 1,679.7\nInventories 1,426.3 1,559.3\nGoodwill 13,923.2 13,995.8"
    assert annotate_table_years(text).splitlines()[2] == "Receivables (2020: 1,615.1; 2019: 1,679.7)"


def test_text_that_is_not_a_table_is_unchanged():
    prose = "In 2022 revenue grew 12% and in 2021 it grew 9%. We opened 30 stores."
    assert annotate_table_years(prose) == prose
    few_rows = "2022 2021\nRevenue 10 9\nThe rest is prose."
    assert annotate_table_years(few_rows) == few_rows  # fewer than three matching rows


def test_passages_are_labelled_only_when_asked():
    chunk = Chunk(id="A:p5:c0", doc_id="A", text=INCOME, page_start=5, page_end=5, metadata={"company": "AMD"})
    assert "3,991 $ 5,506" in format_passages([chunk])
    assert "(2015: 3,991; 2014: 5,506)" in format_passages([chunk], table_years=True)
