"""Resolve asset specs into bounded ``Asset`` samples.

Deterministic and IO-only — no LLM. Given an ``AssetSpec`` (from the YAML) and a
base directory, this counts the source, picks spread-out sample indices, reads
just those rows, and returns an ``Asset``. The LLM transform into world_state
material happens later, in the AssetIngestionAgent.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

from agentspect.assets.base import Asset
from agentspect.assets.loaders import infer_format
from agentspect.assets.registry import loader_registry
from agentspect.assets.sampling import stratified_indices

if TYPE_CHECKING:
    from agentspect.models.yaml_input import AssetSpec


class AssetLoadError(Exception):
    pass


def _resolve_path(path: str, base_dir: str | None) -> str:
    if os.path.isabs(path) or base_dir is None:
        return path
    return os.path.join(base_dir, path)


def load_asset(spec: "AssetSpec", base_dir: str | None = None) -> Asset:
    fmt = spec.format or infer_format(spec.path)
    if not fmt:
        raise AssetLoadError(
            f"Asset '{spec.id}': cannot infer format from path '{spec.path}'; "
            f"set `format` explicitly (one of: {sorted(loader_registry.formats())})"
        )
    if fmt not in loader_registry:
        raise AssetLoadError(
            f"Asset '{spec.id}': no loader for format '{fmt}' "
            f"(known: {sorted(loader_registry.formats())})"
        )

    resolved = _resolve_path(spec.path, base_dir)
    if not os.path.exists(resolved):
        raise AssetLoadError(f"Asset '{spec.id}': file not found: {resolved}")

    loader = loader_registry.get(fmt)
    total = loader.count(resolved)
    indices = stratified_indices(total, spec.sample_size, spec.seed)
    rows = loader.read_rows(resolved, indices)

    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return Asset(
        id=spec.id,
        format=fmt,
        role=spec.role,
        rows=rows,
        total_rows=total,
        sampled_indices=indices,
        description=spec.description,
        target=spec.target,
        content_hash=digest,
    )


def load_assets(specs: list["AssetSpec"], base_dir: str | None = None) -> list[Asset]:
    return [load_asset(spec, base_dir) for spec in specs]
