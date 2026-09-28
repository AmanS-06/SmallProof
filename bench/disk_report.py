"""Disk usage of everything the project stores, against the 15 GB budget (target under 10 GB).

The Ollama app and model folders are found from the OLLAMA_MODELS user
variable (read from the registry, so a fresh `setx` is seen) or the defaults.
The global pip cache is shared with other projects and not counted.

Run: .venv\\Scripts\\python.exe bench\\disk_report.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from common import ROOT, folder_size_bytes, load_config, save_result

BUDGET_GB = 15
TARGET_GB = 10


def user_env(name: str) -> str | None:
    """A user environment variable, reading the registry so values set by setx are seen."""
    if os.environ.get(name):
        return os.environ[name]
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            return None
    return None


def gb(size: int | None) -> float | None:
    return None if size is None else round(size / 2**30, 3)


def main() -> None:
    ollama = load_config()["runtime"]["ollama"]  # the project runs its own Ollama build
    ollama_app = ROOT / Path(ollama["exe"]).parent
    ollama_models = ROOT / ollama["models_dir"]
    counted = {
        ".venv": ROOT / ".venv",
        "models/hf": ROOT / "models" / "hf",
        f"Ollama models ({ollama_models})": ollama_models,
        f"Ollama app ({ollama_app})": ollama_app,
        "data": ROOT / "data",
        "bench/results": ROOT / "bench" / "results",
    }
    rows = {label: gb(folder_size_bytes(path)) for label, path in counted.items()}
    hub = ROOT / "models" / "hf" / "hub"
    per_model = {p.name.removeprefix("models--"): gb(folder_size_bytes(p)) for p in sorted(hub.glob("models--*"))}
    total = round(sum(size for size in rows.values() if size), 3)
    info = {"HF cache on C: (should stay empty)": gb(folder_size_bytes(Path.home() / ".cache" / "huggingface"))}

    save_result("disk", {"counted_gb": rows, "hf_models_gb": per_model, "total_gb": total,
                         "budget_gb": BUDGET_GB, "target_gb": TARGET_GB, "not_counted_gb": info})
    for label, size in {**rows, **{f"  hf: {k}": v for k, v in per_model.items()}}.items():
        print(f"{label:<60} {'absent' if size is None else f'{size:.2f} GB'}")
    print(f"{'TOTAL':<60} {total:.2f} GB of {BUDGET_GB} GB (target {TARGET_GB} GB)")
    for label, size in info.items():
        print(f"{label:<60} {'absent' if size is None else f'{size:.2f} GB'}")


if __name__ == "__main__":
    main()
