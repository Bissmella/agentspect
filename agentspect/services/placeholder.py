import re
from typing import Any

import jsonpointer


class PlaceholderResolutionError(Exception):
    pass


PLACEHOLDER_RE = re.compile(r"\{\{(.+?)\}\}")


def resolve_placeholders(text: str, world_state: dict[str, Any]) -> str:
    def replacer(match: re.Match) -> str:
        path = match.group(1).strip()
        pointer = "/" + path.replace(".", "/")

        try:
            value = jsonpointer.resolve_pointer(world_state, pointer)
        except jsonpointer.JsonPointerException:
            raise PlaceholderResolutionError(
                f"Unresolved placeholder: {{{{{path}}}}} — "
                f"path '{pointer}' not found in current world_state"
            )

        return str(value)

    return PLACEHOLDER_RE.sub(replacer, text)


def extract_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def validate_placeholders(text: str, world_state: dict[str, Any]) -> list[str]:
    errors = []
    for path in extract_placeholders(text):
        pointer = "/" + path.strip().replace(".", "/")
        try:
            jsonpointer.resolve_pointer(world_state, pointer)
        except jsonpointer.JsonPointerException:
            errors.append(f"Invalid placeholder path: {path}")
    return errors
