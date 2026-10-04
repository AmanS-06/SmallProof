import pytest

from smallproof.core import registry
from smallproof.core.interfaces import Reranker


class DummyReranker(Reranker):
    def __init__(self, top=1):
        self.top = top

    def rerank(self, query, chunks, k):
        return chunks[:k]


def test_register_and_create_passes_settings():
    registry.register("reranker", "dummy_test")(DummyReranker)
    try:
        built = registry.create("reranker", "dummy_test", top=3)
        assert isinstance(built, DummyReranker)
        assert built.top == 3
        assert "dummy_test" in registry.available("reranker")
    finally:
        registry.unregister("reranker", "dummy_test")
    assert "dummy_test" not in registry.available("reranker")


def test_backend_must_subclass_its_interface():
    with pytest.raises(TypeError):
        registry.register("classifier", "wrong_test")(DummyReranker)


def test_duplicate_name_is_rejected():
    registry.register("reranker", "dup_test")(DummyReranker)
    try:
        with pytest.raises(ValueError):
            registry.register("reranker", "dup_test")(DummyReranker)
    finally:
        registry.unregister("reranker", "dup_test")


def test_unknown_kind_and_name_give_clear_errors():
    with pytest.raises(ValueError, match="Unknown stage kind"):
        registry.register("rerankr", "x")
    with pytest.raises(ValueError, match="No reranker backend named"):
        registry.get_backend("reranker", "missing")
