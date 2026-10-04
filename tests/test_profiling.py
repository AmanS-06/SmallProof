import pytest

from smallproof.core.profiling import Profiler, percentile, summarize_latencies


def test_stage_records_and_adds_up():
    profiler = Profiler()
    with profiler.stage("retrieve"):
        sum(range(1000))
    first = profiler.timings_ms["retrieve"]
    with profiler.stage("retrieve"):
        sum(range(1000))
    assert first >= 0
    assert profiler.timings_ms["retrieve"] >= first
    assert profiler.total_ms() == profiler.timings_ms["retrieve"]


def test_stage_that_raises_is_still_timed():
    profiler = Profiler()
    with pytest.raises(RuntimeError):
        with profiler.stage("gate"):
            raise RuntimeError("boom")
    assert "gate" in profiler.timings_ms


def test_percentile_matches_numpy_linear_method():
    # numpy.percentile(range(1, 21), 95) == 19.05 and numpy.percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 21)), 95) == pytest.approx(19.05)
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_summarize_latencies():
    summary = summarize_latencies([10, 20, 30, 40])
    assert summary["n"] == 4
    assert summary["median_ms"] == 25
    assert summary["min_ms"] == 10 and summary["max_ms"] == 40
    with pytest.raises(ValueError):
        summarize_latencies([])
