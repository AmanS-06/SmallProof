"""Per-stage timing, shared by the pipeline, eval and bench scripts.

Usage:
    profiler = Profiler()
    with profiler.stage("retrieve"):
        chunks = retriever.retrieve(query, None, 50)
    profiler.timings_ms  # {"retrieve": 12.3}

summarize_latencies() turns the timings of many runs into median and p95.
Memory readings (peak RAM, VRAM) are added in Phase 0, once psutil and
nvidia-ml-py are installed.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any


class Profiler:
    """Collects how long each named stage takes, in milliseconds."""

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the code inside the with-block and add it to this stage's total.

        A stage used twice (for example retrieval before and after widening)
        adds up both times.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            # In finally, so a stage that raises still gets its time recorded.
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed_ms

    def total_ms(self) -> float:
        """Sum of all stages. Only meaningful when stages do not overlap."""
        return sum(self.timings_ms.values())


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100), interpolating linearly between ranks.

    This is the same method as numpy.percentile's default, so numbers stay
    comparable if we switch to numpy later.
    """
    if not values:
        raise ValueError("Need at least one value")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(values_ms: Sequence[float]) -> dict[str, float]:
    """Count, median, p95, min and max of a list of timings in milliseconds."""
    if not values_ms:
        raise ValueError("Need at least one timing to summarize")
    return {
        "n": len(values_ms),
        "median_ms": statistics.median(values_ms),
        "p95_ms": percentile(values_ms, 95),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


class GpuMemorySampler:
    """Samples total used GPU memory in a background thread while the with-block runs.

    Windows usually cannot report GPU memory per process, so this records the
    whole GPU: the baseline before, the peak during, and the difference.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()

    def _used_mib(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._used_mib())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemorySampler:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline_mib = self._used_mib()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(self._used_mib())
        self._nvml.nvmlShutdown()

    def result(self) -> dict[str, float]:
        peak = max(self.samples)
        return {
            "baseline_mib": round(self.baseline_mib),
            "peak_mib": round(peak),
            "delta_mib": round(peak - self.baseline_mib),
            "samples": len(self.samples),
        }
