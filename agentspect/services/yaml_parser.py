import hashlib

import yaml
from pydantic import ValidationError

from agentspect.models.yaml_input import YAMLInput


class YAMLValidationError(Exception):
    pass


def parse_and_validate(yaml_str: str) -> tuple[YAMLInput, str]:
    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise YAMLValidationError(f"Invalid YAML syntax: {e}")

    if not isinstance(data, dict):
        raise YAMLValidationError("YAML must be a mapping at the root level")

    if "rag" in data:
        raise YAMLValidationError(
            "The 'rag' key is reserved for a future release and must not be present"
        )

    required_keys = ["agent_under_test", "world_state", "test_config", "llm_config"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise YAMLValidationError(f"Missing required keys: {', '.join(missing)}")

    tc = data.get("test_config", {})
    total = tc.get("total", 0)
    positive = tc.get("positive", 0)
    negative = tc.get("negative", 0)

    if total != positive + negative:
        raise YAMLValidationError(
            f"test_config.total ({total}) must equal "
            f"positive ({positive}) + negative ({negative})"
        )

    try:
        parsed = YAMLInput.model_validate(data)
    except ValidationError as e:
        errors = "; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors())
        raise YAMLValidationError(f"Validation failed: {errors}")

    asset_ids = [a.id for a in parsed.assets]
    duplicates = {aid for aid in asset_ids if asset_ids.count(aid) > 1}
    if duplicates:
        raise YAMLValidationError(
            f"Duplicate asset id(s): {', '.join(sorted(duplicates))}"
        )

    yaml_hash = hashlib.sha256(yaml_str.encode()).hexdigest()

    return parsed, yaml_hash
