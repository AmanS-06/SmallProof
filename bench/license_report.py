"""Step 0.6: license table for installed libraries (package metadata) and configured models (config).

Model licenses in configs/default.yaml were read from Hugging Face repo metadata.
Model card notes on training data are checked separately (docs/PLAN.md step 0.6).

Run: .venv\\Scripts\\python.exe bench\\license_report.py
"""

from __future__ import annotations

from importlib import metadata

from common import KEY_PACKAGES, load_config, save_result


def package_license(name: str) -> str:
    """Best short license string from package metadata."""
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    text = (meta.get("License") or "").strip().splitlines()
    return text[0][:60] if text else "unknown"


def main() -> None:
    libraries = {name: package_license(name) for name in KEY_PACKAGES}
    models = {spec["repo_id"]: spec["license"] for spec in load_config()["models"].values()}
    save_result("licenses", {"libraries": libraries, "models": models})
    for name, license_name in {**libraries, **models}.items():
        print(f"{name:<45} {license_name}")


if __name__ == "__main__":
    main()
