"""YAML config loading."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_nested(cfg: dict, key_path: str, default: Any = None) -> Any:
    keys = key_path.split(".")
    out = cfg
    for k in keys:
        out = out.get(k, default)
        if out is default:
            return default
    return out
