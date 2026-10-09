"""Built-in asset loaders for local files: CSV and JSONL.

Both stream the file — a single pass to count, a single pass to pull the sampled
rows — so a large file never lands in memory or a prompt. A single big JSON
array is intentionally not supported (use JSONL / newline-delimited records).
"""

from __future__ import annotations

import csv
import json
from typing import Any

from agentspect.assets.base import AssetLoader
from agentspect.assets.registry import register_loader

_EXTENSION_FORMATS = {
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def infer_format(path: str) -> str | None:
    lowered = path.lower()
    for ext, fmt in _EXTENSION_FORMATS.items():
        if lowered.endswith(ext):
            return fmt
    return None


@register_loader
class CSVLoader(AssetLoader):
    format = "csv"

    def count(self, path: str) -> int:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # header
            except StopIteration:
                return 0
            return sum(1 for _ in reader)

    def read_rows(self, path: str, indices: list[int]) -> list[dict[str, Any]]:
        wanted = set(indices)
        if not wanted:
            return []
        out: dict[int, dict[str, Any]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)  # first line is the header
            for i, row in enumerate(reader):
                if i in wanted:
                    out[i] = dict(row)
                    if len(out) == len(wanted):
                        break
        return [out[i] for i in sorted(out)]


@register_loader
class JSONLLoader(AssetLoader):
    format = "jsonl"

    def count(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def read_rows(self, path: str, indices: list[int]) -> list[dict[str, Any]]:
        wanted = set(indices)
        if not wanted:
            return []
        out: dict[int, dict[str, Any]] = {}
        with open(path, encoding="utf-8") as f:
            i = 0
            for line in f:
                if not line.strip():
                    continue
                if i in wanted:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        out[i] = record
                    else:
                        out[i] = {"value": record}
                    if len(out) == len(wanted):
                        break
                i += 1
        return [out[i] for i in sorted(out)]
