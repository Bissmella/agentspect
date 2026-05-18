import pytest


@pytest.fixture
def sample_world_state():
    return {
        "entities": [
            {"id": "customer_1", "phone": "+33612345678", "name": "Isabelle Martin", "verified": True},
            {"id": "customer_2", "phone": "+33698765432", "name": "Jean Dupont", "verified": False},
        ],
        "catalog": {
            "available_slots": ["2026-05-20T10:00", "2026-05-20T14:00"],
        },
        "constraints": [
            "only verified entities can book",
            "slots must be within 09:00-18:00",
        ],
        "context": {
            "current_time": "2026-05-16T08:00",
            "language": "fr",
        },
    }


@pytest.fixture
def valid_yaml():
    return """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"
  capabilities:
    - booking
  known_limitations:
    - no rescheduling

world_state:
  entities:
    - id: customer_1
      phone: "+33612345678"
  catalog:
    available_slots:
      - "2026-05-20T10:00"
  constraints:
    - "only verified entities can book"
  context:
    current_time: "2026-05-16T08:00"

test_config:
  total: 10
  positive: 7
  negative: 3

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""
