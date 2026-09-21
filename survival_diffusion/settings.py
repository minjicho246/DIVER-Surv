"""Filesystem and experiment-tracking configuration."""

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Paths and W&B settings that vary between environments."""

    ucsf_csv_path: str = "data/ucsf/metadata.csv"
    ucsf_img_dir: str = "data/ucsf/images"
    upenn_csv_path: str = "data/upenn/metadata.csv"
    upenn_img_dir: str = "data/upenn/images"
    rhuh_csv_path: str = "data/rhuh/metadata.csv"
    rhuh_img_dir: str = "data/rhuh/images"
    output_root: str = "outputs"
    cache_root: str = "cache"
    wandb_entity: str | None = None
    wandb_project: str = "DIVER-Surv"


def load_runtime_config(path=None):
    """Overlay JSON values onto portable defaults."""
    if path is None:
        return RuntimeConfig()
    with Path(path).open(encoding="utf-8") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("Runtime configuration must be a JSON object.")

    allowed = {field.name for field in fields(RuntimeConfig)}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    for key, value in overrides.items():
        if key == "wandb_entity" and value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Configuration value {key!r} must be a string or null entity.")
    return RuntimeConfig(**overrides)
