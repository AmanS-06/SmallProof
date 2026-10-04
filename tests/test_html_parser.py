from smallproof.ingest.parsers.html_text import html_pages, xbrl_facts
from smallproof.query.metadata_tags import match_company

FILING = """<html><body>
<div style="display:none"><ix:header>hidden facts 999</ix:header></div>
<p>Cover page <ix:nonNumeric name="dei:EntityRegistrantName">Example Corp, Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2023</ix:nonNumeric></p>
<div style="page-break-after:always"></div>
<table><tr><td><p>Total revenues</p></td><td>$</td><td><p>3,773.5</p></td><td>3,958.5</td></tr></table>
<hr style="page-break-after: always"/>
<p>Third page</p>
</body></html>"""


def test_pages_split_at_page_breaks_and_rows_stay_on_one_line():
    pages = html_pages(FILING)
    assert len(pages) == 3
    assert "999" not in pages[0]
    assert "Total revenues $3,773.5 3,958.5" in pages[1]
    assert pages[2] == "Third page"


def test_cover_facts():
    facts = xbrl_facts(FILING)
    assert facts["EntityRegistrantName"] == "Example Corp, Inc." and facts["DocumentFiscalYearFocus"] == "2023"


def test_aliases_and_capital_tickers():
    aliases = {"ALL": "Allstate", "Cboe Global Markets, Inc.": "Cboe Global Markets"}
    assert match_company("ALL net income in 2023", ["Allstate"], [], aliases) == ("Allstate", 1.0)
    assert match_company("Is all of it debt?", ["Allstate"], [], aliases) is None
    assert match_company("CBOE revenue", ["Cboe Global Markets"], [], {"CBOE": "Cboe Global Markets"})[0] == "Cboe Global Markets"
