"""ATA assets — sample external data sources into test material.

Two axes:

- **source × format** — how the data is read, via ``AssetLoader`` implementations
  in the ``loader_registry`` (built-in: CSV, JSONL). Add a format or a remote
  source by registering another loader.
- **role** — what a sample becomes: ``entities`` and ``data_sample`` are handled
  by the ``AssetIngestionAgent``; ``knowledge_base`` (RAG) is reserved.
"""

from agentspect.assets.base import Asset, AssetLoader
from agentspect.assets.loaders import CSVLoader, JSONLLoader, infer_format
from agentspect.assets.registry import LoaderRegistry, loader_registry, register_loader
from agentspect.assets.sampling import stratified_indices
from agentspect.assets.service import AssetLoadError, load_asset, load_assets

__all__ = [
    "Asset",
    "AssetLoader",
    "LoaderRegistry",
    "loader_registry",
    "register_loader",
    "CSVLoader",
    "JSONLLoader",
    "infer_format",
    "stratified_indices",
    "load_asset",
    "load_assets",
    "AssetLoadError",
]
