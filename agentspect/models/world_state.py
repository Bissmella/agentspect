import copy
from typing import Any

import jsonpatch
import jsonpointer


class PatchValidationError(Exception):
    pass


class WorldState:
    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._snapshots: list[dict[str, Any]] = []

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        return self._snapshots

    def snapshot(self) -> dict[str, Any]:
        snap = copy.deepcopy(self._data)
        self._snapshots.append(snap)
        return snap

    def resolve_pointer(self, path: str) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        return jsonpointer.resolve_pointer(self._data, path)

    def pointer_exists(self, path: str) -> bool:
        if not path.startswith("/"):
            path = "/" + path
        try:
            jsonpointer.resolve_pointer(self._data, path)
            return True
        except jsonpointer.JsonPointerException:
            return False

    def validate_patch(self, ops: list[dict[str, Any]]) -> list[str]:
        errors = []
        for op in ops:
            operation = op.get("op")
            path = op.get("path", "")

            if operation in ("remove", "replace", "move"):
                source_path = op.get("from", path) if operation == "move" else path
                if not self.pointer_exists(source_path):
                    errors.append(f"Path does not exist for {operation}: {source_path}")

            if operation == "replace":
                if self.pointer_exists(path):
                    existing = self.resolve_pointer(path)
                    new_value = op.get("value")
                    if type(existing) != type(new_value) and existing is not None:
                        errors.append(
                            f"Type mismatch at {path}: "
                            f"existing {type(existing).__name__}, "
                            f"new {type(new_value).__name__}"
                        )

            if operation == "move":
                from_path = op.get("from")
                if from_path and not self.pointer_exists(from_path):
                    errors.append(f"Source path does not exist for move: {from_path}")

        return errors

    def apply_patch(self, ops: list[dict[str, Any]]) -> None:
        errors = self.validate_patch(ops)
        if errors:
            raise PatchValidationError("; ".join(errors))

        patch = jsonpatch.JsonPatch(ops)
        self._data = patch.apply(self._data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        return cls(copy.deepcopy(data))
