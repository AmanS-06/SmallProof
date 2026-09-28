"""Step 0.4: environment check. Safe to run any time; it never downloads anything.

Checks:
1. Python version and key package versions
2. torch is the CPU build and no CUDA packages are installed
3. the hf CLI is installed
4. every configured model is downloaded and loads offline
5. nothing was written to the default Hugging Face cache on C:
6. Ollama is reachable, the SLM is pulled, and it runs on the GPU

Run: .venv\\Scripts\\python.exe bench\\env_check.py
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

from common import folder_size_bytes, load_config, model_dir, package_versions, save_result

checks: list[dict] = []


def add(check: str, status: str, detail: str = "") -> None:
    """status is ok, missing, fail or info."""
    checks.append({"check": check, "status": status, "detail": detail})


def first_line(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"[:200]


def check_python_and_torch() -> None:
    add("python 3.11", "ok" if sys.version_info[:2] == (3, 11) else "fail", sys.version.split()[0])
    missing = [name for name, version in package_versions().items() if version is None]
    add("key packages installed", "fail" if missing else "ok", f"missing: {missing}" if missing else "")
    import torch

    cpu_build = torch.__version__.endswith("+cpu") and not torch.cuda.is_available()
    add("torch is CPU-only", "ok" if cpu_build else "fail", torch.__version__)
    # nvidia-ml-py only reads GPU status (NVML); it is not a CUDA runtime.
    cuda_packages = sorted(
        name for dist in metadata.distributions()
        if (name := dist.metadata["Name"] or "") and name.lower() != "nvidia-ml-py"
        and re.match(r"(nvidia-|.*cuda)", name.lower())
    )
    add("no CUDA packages", "fail" if cuda_packages else "ok", ", ".join(cuda_packages))
    hf_cli = Path(sys.executable).parent / ("hf.exe" if sys.platform == "win32" else "hf")
    add("hf CLI installed", "ok" if hf_cli.exists() else "fail", str(hf_cli))


def _load_classifier(path: str) -> None:
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=path, device=-1, model_kwargs={"reference_compile": False})
    classifier("Net sales rose 3 percent.", ["revenue", "legal"])


def _load_extractor(path: str) -> None:
    from gliner import GLiNER

    GLiNER.from_pretrained(path, local_files_only=True).predict_entities("Acme reported $4 billion in 2018.", ["company", "money"])


def _load_embedder(path: str) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(path, device="cpu").encode("test")


def _load_reranker(path: str) -> None:
    from sentence_transformers import CrossEncoder

    CrossEncoder(path, device="cpu").predict([("question", "passage")])


LOADERS = {"classifier": _load_classifier, "extractor": _load_extractor,
           "embedder": _load_embedder, "reranker": _load_reranker}


def check_models() -> None:
    for key in load_config()["models"]:
        path = model_dir(key)
        if path is None:
            add(f"model {key}", "missing", "not downloaded (docs/PLAN.md step 0.2)")
            continue
        loader = LOADERS.get(key)
        if loader is None:  # the GLiNER tokenizer is checked by loading GLiNER
            add(f"model {key}", "ok", "downloaded")
            continue
        try:
            loader(str(path))
            add(f"model {key}", "ok", "loads offline")
        except Exception as error:
            add(f"model {key}", "fail", first_line(error))


def check_default_cache() -> None:
    # Models live in hub/. The hf CLI also keeps small housekeeping files one level up; those are fine.
    default_hub = Path.home() / ".cache" / "huggingface" / "hub"
    size = folder_size_bytes(default_hub) or 0
    add("no models in C: HF cache", "ok" if size == 0 else "fail", f"{default_hub}: {size / 2**20:.1f} MB")


def check_ollama() -> None:
    try:
        import ollama

        pulled = [entry.model for entry in ollama.list().models]
    except Exception as error:
        add("Ollama reachable", "missing", first_line(error))
        return
    add("Ollama reachable", "ok")
    model = load_config()["generator"]["model"]
    if model not in pulled:
        add(f"SLM {model} pulled", "missing", f"pulled: {pulled}")
        return
    ollama.generate(model=model, prompt="Reply with OK.", options={"num_predict": 5})
    running = [entry for entry in ollama.ps().models if entry.model == model]
    fraction = running[0].size_vram / running[0].size if running else 0
    add(f"SLM {model} on GPU", "ok" if fraction == 1 else "fail", f"{fraction:.0%} of the model in VRAM")


def main() -> None:
    check_python_and_torch()
    check_models()
    check_default_cache()
    check_ollama()
    save_result("env_check", {"checks": checks})
    for row in checks:
        print(f"[{row['status']:>7}] {row['check']}" + (f" ({row['detail']})" if row["detail"] else ""))


if __name__ == "__main__":
    main()
