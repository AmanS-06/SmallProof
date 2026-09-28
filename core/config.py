"""Loads configs/default.yaml, merges a domain pack on top, and finds model files.

A pack's config.yaml only lists what differs from the default (labels,
patterns, prompts). Model files are looked up in the project's Hugging Face
cache (paths.models_dir). With offline: true nothing is ever downloaded.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = ["paths.models_dir", "paths.index_dir", "models.embedder.repo_id", "generator.model"]
_MISSING = object()


def deep_merge(base: dict, override: dict) -> dict:
    """Return base updated with override. Nested dicts are merged, everything else replaced."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get(config: dict, dotted_key: str, default: Any = _MISSING) -> Any:
    """Read a nested value, for example get(config, "retrieval.bm25_k")."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            if default is _MISSING:
                raise KeyError(f"Config key '{dotted_key}' is missing")
            return default
        value = value[part]
    return value


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(pack: str | None = None, overrides: dict | None = None, root: Path = PROJECT_ROOT) -> dict:
    """Default config, then the pack's config, then any overrides (for example from an ablation)."""
    config = _read_yaml(root / "configs" / "default.yaml")
    if pack:
        pack_dir = root / "packs" / pack
        if not pack_dir.is_dir():
            raise ValueError(f"Unknown pack '{pack}': {pack_dir} does not exist")
        config = deep_merge(config, _read_yaml(pack_dir / "config.yaml"))
        config["pack_name"] = pack
        config["pack_dir"] = str(pack_dir)
    if overrides:
        config = deep_merge(config, overrides)
    config["root"] = str(root)
    for key in REQUIRED_KEYS:
        try:
            get(config, key)
        except KeyError as error:
            raise ValueError(f"Invalid config: {error}") from None
    return config


def resolve_path(config: dict, path: str | Path) -> Path:
    """Paths in the config are relative to the project root unless absolute."""
    path = Path(path)
    return path if path.is_absolute() else Path(config["root"]) / path


def model_path(config: dict, key: str) -> str:
    """Local snapshot folder of a configured model.

    Offline, a missing model raises an error with a pointer to the download
    step. Online, the repo id is returned so the library downloads it.
    """
    spec = config["models"][key]
    repo_folder = resolve_path(config, config["paths"]["models_dir"]) / "hub" / (
        "models--" + spec["repo_id"].replace("/", "--")
    )
    revision = str(spec.get("revision", "main"))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):  # a branch name: look up its commit
        ref_file = repo_folder / "refs" / revision
        revision = ref_file.read_text().strip() if ref_file.exists() else ""
    snapshot = repo_folder / "snapshots" / revision
    if revision and snapshot.is_dir():
        return str(snapshot)
    if config.get("offline", True):
        raise FileNotFoundError(f"Model '{key}' ({spec['repo_id']}) is not downloaded. See docs/PLAN.md step 0.2.")
    return spec["repo_id"]


def apply_runtime_env(config: dict) -> None:
    """Point Hugging Face at the project cache and, when offline, block downloads."""
    os.environ.setdefault("HF_HOME", str(resolve_path(config, config["paths"]["models_dir"])))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if config.get("offline", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    threads = get(config, "runtime.torch_threads", None)
    if threads:  # thread cap keeps the CPU (and the laptop) cooler
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(threads)
        import torch

        torch.set_num_threads(int(threads))
