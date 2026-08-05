from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MERMAID_FENCE_FORMAT = "pymdownx.superfences.fence_code_format"
_MERMAID_FENCE_FORMAT_TAG = (
    "tag:yaml.org,2002:python/name:pymdownx.superfences.fence_code_format"
)


class _MkDocsLoader(yaml.SafeLoader):
    pass


def _construct_mermaid_fence_format(
    loader: yaml.SafeLoader,
    node: yaml.nodes.Node,
) -> str:
    value = loader.construct_scalar(node)
    if value:
        raise ValueError("The Mermaid fence formatter tag must not carry a scalar value")
    return MERMAID_FENCE_FORMAT


_MkDocsLoader.add_constructor(
    _MERMAID_FENCE_FORMAT_TAG,
    _construct_mermaid_fence_format,
)


def load_mkdocs_config(path: Path) -> dict[str, Any]:
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    if not isinstance(config, dict):
        raise ValueError("MkDocs configuration must be a mapping")
    return config
