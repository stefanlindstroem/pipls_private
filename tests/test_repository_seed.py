from __future__ import annotations

from pathlib import Path

import pipls


def test_package_imports() -> None:
    assert pipls.__version__ == "0.0.0"


def test_required_llm_contracts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = {
        ".llm/README.md",
        ".llm/project.md",
        ".llm/mathematics.md",
        ".llm/numerical_contracts.md",
        ".llm/development.md",
        ".llm/public_api.md",
        ".llm/snapshot.sh",
        ".llm/apply_patch.sh",
        ".llm/create_patch.sh",
    }
    missing = sorted(path for path in required if not (root / path).is_file())
    assert not missing, f"Missing repository contracts: {missing}"


def test_llm_layer_is_outside_installable_package() -> None:
    package_root = Path(pipls.__file__).resolve().parent
    assert ".llm" not in {part.name for part in package_root.parents}
