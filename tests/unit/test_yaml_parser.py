import pytest

from agentspect.services.yaml_parser import YAMLValidationError, parse_and_validate


def test_parse_valid_yaml(valid_yaml):
    result, yaml_hash = parse_and_validate(valid_yaml)
    assert result.agent_under_test.name == "Test Agent"
    assert result.test_config.total == 10
    assert result.test_config.positive == 7
    assert result.test_config.negative == 3
    assert len(yaml_hash) == 64


def test_invalid_yaml_syntax():
    invalid_yaml = "foo: bar: baz"
    with pytest.raises(YAMLValidationError, match="Invalid YAML syntax"):
        parse_and_validate(invalid_yaml)


_CALLABLE_YAML = """
agent_under_test:
  name: "In-process Agent"
  protocol: {protocol}
  description: "A callable agent"

world_state:
  entities: []
  catalog: {{}}
  constraints: []
  context: {{}}

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""


def test_callable_protocol_needs_no_url():
    result, _ = parse_and_validate(_CALLABLE_YAML.format(protocol="callable"))
    assert result.agent_under_test.protocol == "callable"
    assert result.agent_under_test.url is None


def test_http_protocol_requires_url():
    with pytest.raises(YAMLValidationError, match="url is required"):
        parse_and_validate(_CALLABLE_YAML.format(protocol="http"))


def test_unknown_protocol_rejected():
    with pytest.raises(YAMLValidationError):
        parse_and_validate(_CALLABLE_YAML.format(protocol="grpc"))


_ASSET_YAML = """
agent_under_test:
  name: "Data Agent"
  protocol: callable
  description: "reasons over data"

world_state:
  entities: []
  catalog: {{}}
  constraints: []
  context: {{}}

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514

assets:
{assets}
"""


def test_assets_parse():
    result, _ = parse_and_validate(_ASSET_YAML.format(
        assets="  - id: customers\n    path: ./customers.csv\n    role: entities\n"
    ))
    assert len(result.assets) == 1
    assert result.assets[0].id == "customers"
    assert result.assets[0].sample_size == 50  # default


def test_duplicate_asset_id_rejected():
    dup = (
        "  - id: customers\n    path: ./a.csv\n"
        "  - id: customers\n    path: ./b.csv\n"
    )
    with pytest.raises(YAMLValidationError, match="Duplicate asset id"):
        parse_and_validate(_ASSET_YAML.format(assets=dup))


def test_unsupported_asset_format_rejected():
    with pytest.raises(YAMLValidationError):
        parse_and_validate(_ASSET_YAML.format(
            assets="  - id: c\n    path: ./c.parquet\n    format: parquet\n"
        ))


def test_unsupported_asset_role_rejected():
    with pytest.raises(YAMLValidationError):
        parse_and_validate(_ASSET_YAML.format(
            assets="  - id: c\n    path: ./c.csv\n    role: bogus\n"
        ))


def test_rag_key_rejected():
    yaml_with_rag = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 5
  positive: 3
  negative: 2

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514

rag:
  enabled: true
"""
    with pytest.raises(YAMLValidationError, match="'rag' key is reserved"):
        parse_and_validate(yaml_with_rag)


def test_total_mismatch():
    yaml_mismatch = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 10
  positive: 5
  negative: 3

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""
    with pytest.raises(YAMLValidationError, match="total.*must equal"):
        parse_and_validate(yaml_mismatch)


def test_missing_required_keys():
    yaml_missing = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"
"""
    with pytest.raises(YAMLValidationError, match="Missing required keys"):
        parse_and_validate(yaml_missing)


def test_invalid_protocol():
    yaml_bad_protocol = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: ftp
  description: "A test agent"

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 5
  positive: 3
  negative: 2

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""
    with pytest.raises(YAMLValidationError, match="Validation failed"):
        parse_and_validate(yaml_bad_protocol)


def test_invalid_provider():
    yaml_bad_provider = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 5
  positive: 3
  negative: 2

llm_config:
  provider: azure
  model: gpt-4
"""
    with pytest.raises(YAMLValidationError, match="Validation failed"):
        parse_and_validate(yaml_bad_provider)
