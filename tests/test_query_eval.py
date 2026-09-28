import pytest

from core.config import deep_merge, get, load_config
from core.pipeline import VARIANTS
from core.types import Chunk, Entity
from eval.grading import extract_numbers, grade, numeric_grade
from query.bm25_index import tokenize
from query.metadata_tags import QueryTags, boost_factors, match_company, match_year


def test_deep_merge_and_get():
    merged = deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}, "d": 1}
    assert get(merged, "a.c") == 3 and get(merged, "x.y", None) is None
    with pytest.raises(KeyError):
        get(merged, "x")


def test_pack_config_merges_over_default():
    config = load_config("financebench")
    assert config["pack_name"] == "financebench"
    assert config["models"]["embedder"]["repo_id"] == "BAAI/bge-small-en-v1.5"
    assert "metric_lookup" in config["router"]["intents"]
    with pytest.raises(ValueError):
        load_config("no_such_pack")


def test_variants_switch_one_component_off():
    assert VARIANTS["C_no_reranker"]["reranker"] is False and VARIANTS["C_no_reranker"]["bm25"] is True
    assert VARIANTS["B_dense_rag"] == {k: k == "dense" for k in VARIANTS["C_full"]}


def test_bm25_tokenize():
    assert tokenize("What was the FY2018 capex of $1,577 million?") == ["fy2018", "capex", "1577", "million"]


def test_company_and_year_tags():
    companies = ["3M", "American Express", "Amcor"]
    assert match_company("What was 3M's FY2018 capex?", companies, []) == ("3M", 1.0)
    fuzzy = match_company("How did Amex do?", companies, [Entity("Amercan Express", "company", 0.9)])
    assert fuzzy[0] == "American Express" and fuzzy[1] > 0.8
    assert match_year("Change from FY2017 to FY2018?", ["2017", "2018", "2019"]) == ("2018", 1.0)
    assert match_year("In 1999?", ["2018"]) is None


def test_boosts_count_confident_matches_only():
    chunk = Chunk("a", "d", "x", 1, 1, chunk_type="table", metadata={"company": "3M", "year": "2018", "section": "consolidated statements of income"})
    tags = QueryTags(company=("3M", 1.0), year=("2018", 0.4), section=("income statement", 0.9), intent=("metric_lookup", 0.8))
    factors = boost_factors([chunk], tags, 0.25, 0.5, {"income statement": ["income"]}, {"metric_lookup": ["table"]})
    assert factors["a"] == pytest.approx(1.75)  # company, section, intent match; year is below the threshold


def test_numeric_grading():
    assert extract_numbers("Capex was $1,577 million in 2018") == [(1577e6, False)]
    assert numeric_grade("$1577.00", "3M spent $1.577 billion [p. 60].") is True
    assert numeric_grade("24.26%", "The margin was 0.2426.") is True
    assert numeric_grade("$1577.00", "It was $1,373 million.") is False
    assert numeric_grade("Yes, it improved.", "It improved.") is None
    assert grade("q", "$10", "NOT FOUND", refused=True) == {"correct": False, "method": "refused"}
    assert grade("q", "UNANSWERABLE", "sorry", refused=True) == {"correct": True, "method": "unanswerable"}
    assert grade("q", "Yes.", "Yes", refused=False) == {"correct": None, "method": "ungraded"}
