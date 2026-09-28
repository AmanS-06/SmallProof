"""Shared helpers for the Phase 0 bench scripts.

Importing this module first also:
- puts the project root on sys.path, so `import core` works,
- points HF_HOME at models/hf and turns on offline mode, so a missing model
  stops the script instead of being downloaded.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma
# Thread cap (same as runtime.torch_threads): keeps the laptop cool. Child processes inherit it.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import yaml  # noqa: E402

from core.profiling import GpuMemorySampler, summarize_latencies  # noqa: E402,F401

RESULTS_DIR = ROOT / "bench" / "results"
FIXTURES_FILE = ROOT / "bench" / "fixtures" / "fixtures.json"
# BENCH_PDF lets a test run point the scripts at another PDF.
SAMPLE_PDF = Path(os.environ.get("BENCH_PDF", str(ROOT / "data" / "samples" / "3M_2018_10K.pdf")))
WARMUP_RUNS = 2
TIMED_RUNS = 20

KEY_PACKAGES = [
    "torch", "transformers", "sentence-transformers", "gliner", "huggingface-hub", "tokenizers",
    "chromadb", "rank-bm25", "pypdfium2", "pdfplumber", "pymupdf", "ollama", "numpy",
    "psutil", "nvidia-ml-py", "pyyaml", "pytest",
]


# Section: config, models, fixtures

def load_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def load_fixtures() -> dict[str, Any]:
    return json.loads(FIXTURES_FILE.read_text(encoding="utf-8"))


def model_dir(key: str) -> Path | None:
    """Local snapshot folder of a configured model, or None if it is not downloaded."""
    spec = load_config()["models"][key]
    repo_folder = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + spec["repo_id"].replace("/", "--"))
    revision = spec["revision"]
    if revision == "main":  # a branch name: the cache records which commit it points to
        ref_file = repo_folder / "refs" / "main"
        if not ref_file.exists():
            return None
        revision = ref_file.read_text().strip()
    snapshot = repo_folder / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def require_model(key: str) -> str:
    """Path of a model as a string, or exit with a clear message if it is missing."""
    path = model_dir(key)
    if path is None:
        repo_id = load_config()["models"][key]["repo_id"]
        sys.exit(f"Model '{key}' ({repo_id}) is not downloaded yet. See docs/PLAN.md step 0.2.")
    return str(path)


def load_sample_chunks(words_per_chunk: int = 260, min_words: int = 30) -> list[tuple[int, str]]:
    """Split the sample PDF into (page, text) chunks of about words_per_chunk words.

    A rough stand-in for the real chunker (Phase 1): chunks never cross a page,
    and short leftovers are dropped. Benches record real token counts.
    """
    if not SAMPLE_PDF.exists():
        sys.exit(f"Sample PDF missing: {SAMPLE_PDF}. See docs/PLAN.md step 0.2.")
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(SAMPLE_PDF))
    chunks: list[tuple[int, str]] = []
    try:
        for index in range(len(pdf)):
            words = pdf[index].get_textpage().get_text_range().split()
            for start in range(0, len(words), words_per_chunk):
                piece = words[start : start + words_per_chunk]
                if len(piece) >= min_words:
                    chunks.append((index + 1, " ".join(piece)))
    finally:
        pdf.close()
    return chunks


# Section: timing and memory

def time_once(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn once. Returns (result, milliseconds)."""
    start = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - start) * 1000, 1)


def time_calls(fn: Callable[[], Any], warmup: int = WARMUP_RUNS, runs: int = TIMED_RUNS) -> dict[str, float]:
    """Run fn `warmup` times untimed, then `runs` times timed. Returns median, p95 and more (ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {key: round(value, 2) for key, value in summarize_latencies(times).items()}


def peak_ram_mb() -> float:
    """Peak memory of this process so far (Windows peak working set, RSS elsewhere)."""
    import psutil

    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 2**20, 1)


# Section: one process per component

def _child_entry(queue: Any, target: Callable[..., dict], args: tuple) -> None:
    try:
        result = target(*args)
        result["peak_ram_mb"] = peak_ram_mb()
        queue.put(("ok", result))
    except BaseException:  # send the error back instead of dying silently
        queue.put(("error", traceback.format_exc()))


def run_in_child(target: Callable[..., dict], *args: Any) -> dict:
    """Run target(*args) in a fresh process, so its peak RAM is measured on its own.

    target must be a top-level function that returns a dict.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child_entry, args=(queue, target, args))
    process.start()
    status, payload = queue.get(timeout=3600)  # read before join, so a big result cannot block
    process.join()
    if status == "error":
        raise RuntimeError(f"{target.__name__} failed in the child process:\n{payload}")
    return payload


# Section: setup info and results

def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def _power_plan() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        output = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\((.+)\)", output)
    return match.group(1) if match else None


def _gpu_info() -> dict[str, Any] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": pynvml.nvmlDeviceGetName(handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "total_mib": round(memory.total / 2**20),
            "used_mib_now": round(memory.used / 2**20),
        }
    finally:
        pynvml.nvmlShutdown()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def setup_info() -> dict[str, Any]:
    """Hardware, power and software details saved with every result."""
    import psutil

    battery = psutil.sensors_battery()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "on_ac_power": battery.power_plugged if battery else None,
        "power_plan": _power_plan(),
        "gpu": _gpu_info(),
        "packages": package_versions(),
    }


def save_result(name: str, data: dict[str, Any]) -> Path:
    """Write bench/results/<name>.json with the setup info added."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"setup": setup_info(), **data}, indent=2, default=str), encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")
    return path


def folder_size_bytes(path: Path) -> int | None:
    """Total size of the files under path, not following links. None if path does not exist."""
    if not path.exists():
        return None
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(folder, name)
            if not os.path.islink(file_path):
                total += os.lstat(file_path).st_size
    return total
