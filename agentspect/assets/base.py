"""Asset loader base class and the loaded-Asset representation.

An asset is an external data source (today: a local CSV/JSONL file; later a
Parquet export or a live warehouse query) that ATA samples from to build test
material. A loader never reads the whole thing into memory or a prompt: it
reports the total row count cheaply, then reads only the specific rows selected
for sampling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class Asset(BaseModel):
    """A bounded sample of an external source, ready to be ingested."""

    id: str
    format: str
    role: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total_rows: int = 0
    sampled_indices: list[int] = Field(default_factory=list)
    description: str | None = None
    target: str | None = None
    content_hash: str | None = None


class AssetLoader(ABC):
    """Reads a source without loading it wholesale.

    Subclasses declare the ``format`` they handle and are registered so the
    service can look them up. Both operations stream the source: ``count`` in
    one pass keeping nothing, ``read_rows`` in one pass keeping only the rows at
    the requested indices.
    """

    format: ClassVar[str]

    @abstractmethod
    def count(self, path: str) -> int:
        """Total number of records in the source (rows, excluding any header)."""

    @abstractmethod
    def read_rows(self, path: str, indices: list[int]) -> list[dict[str, Any]]:
        """Return the records at the given 0-based indices, in index order."""
