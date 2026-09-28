import json

import pytest

from core.types import Chunk, Citation, LabelScore, Verdict


def make_chunk(**overrides):
    fields = {"id": "c1", "doc_id": "doc", "text": "Revenue grew.", "page_start": 3, "page_end": 4}
    fields.update(overrides)
    return Chunk(**fields)


def test_chunk_pages_are_inclusive():
    assert list(make_chunk().pages) == [3, 4]


def test_chunk_rejects_bad_page_range():
    with pytest.raises(ValueError):
        make_chunk(page_start=5, page_end=4)
    with pytest.raises(ValueError):
        make_chunk(page_start=0, page_end=0)


def test_with_score_returns_copy_and_keeps_original():
    chunk = make_chunk()
    scored = chunk.with_score(0.7)
    assert scored.score == 0.7
    assert chunk.score is None
    assert scored is not chunk


def test_verdict_saves_to_json_as_text():
    assert json.dumps(Verdict.SUFFICIENT) == '"sufficient"'


def test_label_score_is_a_plain_tuple():
    label, score = LabelScore("table", 0.9)
    assert (label, score) == ("table", 0.9)


def test_citation_can_go_in_a_set():
    assert len({Citation("doc", 3), Citation("doc", 3)}) == 1
