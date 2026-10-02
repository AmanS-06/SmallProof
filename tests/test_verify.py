from core.types import Chunk
from generate.numbers import find_numbers, same_amount
from generate.verify import verify_answer

BALANCE = """Consolidated Balance Sheets
May 31, 2020 May 26, 2019
Total current assets 5,121.3 4,186.7
Total current liabilities 7,491.5 7,987.1
Total assets 30,806.7 30,111.2"""
CASH_FLOW = """Consolidated Statement of Cash Flows
Years ended December 31
2018
2017
2016
Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)
Proceeds from sale of PP&E 262 49 58
Net cash used in investing activities (222) (3,086) (1,403)"""


def chunk(text: str, page: int = 5) -> Chunk:
    return Chunk(id=f"D:p{page}:c0", doc_id="D", text=text, page_start=page, page_end=page)


def test_numbers_skip_years_citations_and_names():
    found = find_numbers("In FY2018 3M spent $1,577 million [p. 60] on PP&E, filed in its 10-K for 2018 [1].")
    assert [(n.raw, n.value, n.scale) for n in found] == [("$1,577 million", 1577.0, 1e6)]


def test_numbers_read_accounting_negatives_and_percents():
    found = find_numbers("(1,577) and 18.36% and 0.68")
    assert [(n.value, n.percent, n.decimals) for n in found] == [(-1577.0, False, 0), (18.36, True, 2), (0.68, False, 2)]


def test_same_amount_allows_rounding_and_scale_only():
    written = find_numbers("$1.6 billion")[0]
    assert same_amount(written, 1577)  # 1,577 million rounds to 1.6 billion
    assert not same_amount(written, 1373)
    assert not same_amount(find_numbers("6,818")[0], 5818)


def test_a_copied_number_passes():
    result = verify_answer("3M spent $1,577 million on PP&E in FY2018 [p. 5].", [chunk(CASH_FLOW)], "capex for 2018?")
    assert result.ok and result.numbers[0].status == "quoted"


def test_a_wrong_calculation_is_corrected():
    answer = "The ratio is 5.12. Current assets were $5,121.3 million and current liabilities $7,491.5 million: 5,121.3 / 7,491.5 = 5.12."
    result = verify_answer(answer, [chunk(BALANCE)], "working capital ratio for FY2020?")
    assert result.ok
    assert "0.68" in result.text and "5.12" not in result.text
    assert result.corrections


def test_a_number_from_nowhere_is_refused():
    answer = "Net working capital is $6,818 million: total current assets ($5,121.3 million) minus liabilities ($7,491.5 million)."
    result = verify_answer(answer, [chunk(BALANCE)], "net working capital?")
    assert not result.ok and any("6,818" in reason for reason in result.reasons)


def test_a_one_step_result_without_an_equals_sign_is_accepted():
    answer = "Current assets of $5,121.3 million less current liabilities of $7,491.5 million leave $2,370.2 million of negative working capital."
    assert verify_answer(answer, [chunk(BALANCE)], "working capital?").ok


def test_a_chain_of_steps_is_followed():
    text = "Capex 1,577, 1,373 and 1,420."
    answer = "Average capex = (1,577 + 1,373 + 1,420) / 3 = 4,370 / 3 = 1,456.67 million."
    assert verify_answer(answer, [chunk(text)], "average capex?").ok


def test_a_value_from_another_years_column_is_refused():
    answer = "In 2018, 3M spent $1,373 million on property, plant and equipment."
    result = verify_answer(answer, [chunk(CASH_FLOW)], "What was 3M's 2018 capex?")
    assert not result.ok and result.numbers[0].status == "wrong_year"


def test_hedged_and_cut_off_answers_are_refused():
    assert not verify_answer("Capex was $1,577 million. NOT FOUND", [chunk(CASH_FLOW)]).ok
    assert not verify_answer("Capex was $1,577 million", [chunk(CASH_FLOW)], truncated=True).ok


def test_page_words_ordinals_and_doubling():
    assert find_numbers("From page 68, assets were 1,001,425.")[0].value == 1001425.0  # 68 is a page, not an amount
    assert verify_answer("It raised dividends for 61 consecutive years.", [chunk("the 61st consecutive year")]).ok
    two_lines = chunk("Five Year Credit Agreement up to $4,200,000,000. 364 Day Credit Agreement up to $4,200,000,000.")
    assert verify_answer("Each allows $4,200,000,000, so the total is $8,400,000,000.", [two_lines]).ok


def test_answers_without_numbers_pass():
    assert verify_answer("Yes, the business is cyclical [p. 5].", [chunk("The business is cyclical.")]).ok
