"""Data-collection accuracy: normalization, CER, recall-probe scoring, metric."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from ata.agents.scorer import (
    ExtractedFields,
    _char_error_rate,
    _evaluate_data_collection_assertion,
    _normalize_field,
)
from ata.metrics.base import MetricContext
from ata.metrics.builtin import DataCollectionMetric
from ata.models.suite import DataCollectionAssertion, DataField, ScenarioVerdict, Verdict
from ata.models.transcript import Transcript, Turn

# ── deterministic helpers ──────────────────────────────────────────────────────

def test_normalize_phone_strips_nondigits():
    assert _normalize_field("phone", "+33 6 12-34") == "3361234"


def test_normalize_email_lowercases_and_trims():
    assert _normalize_field("email", " John@Example.COM ") == "john@example.com"


def test_char_error_rate():
    assert _char_error_rate("33612", "33612") == 0.0
    assert _char_error_rate("33612", "33610") == pytest.approx(0.2)  # 1 of 5 wrong
    assert _char_error_rate("", "") == 0.0
    assert _char_error_rate("abc", "") == 1.0


# ── recall-probe scoring (LLM extracts, compare is deterministic) ───────────────

def _recall_transcript(agent_line: str) -> Transcript:
    t = Transcript(scenario_id="probe1", session_id="x", protocol="http")
    t.add_turn(Turn(user_message="what do you have on file for me?", agent_response=agent_line))
    return t


async def test_data_collection_all_fields_match():
    assertion = DataCollectionAssertion(
        description="agent recalls the caller's details",
        fields=[
            DataField(name="email", expected="isabelle@example.com", kind="email"),
            DataField(name="phone", expected="+33612345678", kind="phone"),
        ],
    )
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=ExtractedFields(values={
            "email": "Isabelle@example.com",          # case differs — still matches
            "phone": "+33 6 12 34 56 78",             # formatting differs — still matches
        })
    )
    result, fields = await _evaluate_data_collection_assertion(
        assertion, _recall_transcript("..."), llm
    )
    assert result.satisfied is True
    assert all(f["match"] for f in fields)
    assert all(f["cer"] == 0.0 for f in fields)


async def test_data_collection_flags_wrong_digit():
    assertion = DataCollectionAssertion(
        description="d",
        fields=[DataField(name="phone", expected="+33612345678", kind="phone")],
    )
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=ExtractedFields(values={"phone": "+33612345679"})  # last digit wrong
    )
    result, fields = await _evaluate_data_collection_assertion(
        assertion, _recall_transcript("..."), llm
    )
    assert result.satisfied is False
    assert fields[0]["match"] is False
    assert fields[0]["cer"] > 0


async def test_data_collection_tolerance_allows_near_miss():
    assertion = DataCollectionAssertion(
        description="d",
        fields=[DataField(name="phone", expected="+33612345678", kind="phone")],
        max_cer=0.1,  # tolerate one digit off in an 11-digit number (~0.09)
    )
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=ExtractedFields(values={"phone": "+33612345679"})
    )
    result, fields = await _evaluate_data_collection_assertion(
        assertion, _recall_transcript("..."), llm
    )
    assert result.satisfied is True  # within tolerance
    assert fields[0]["match"] is True


# ── assertion model + metric ────────────────────────────────────────────────────

def test_data_collection_assertion_parses_in_union():
    from ata.models.suite import Scenario, ScenarioType

    s = Scenario(
        id="probe1", type=ScenarioType.POSITIVE, description="recall probe", turns=["confirm"],
        assertions=[DataCollectionAssertion(
            description="d", fields=[DataField(name="email", expected="a@b.com", kind="email")])],
    )
    assert s.assertions[0].type == "data_collection"


def test_data_collection_metric_aggregates():
    v = ScenarioVerdict(
        scenario_id="probe1", verdict=Verdict.SUCCESS, reason="",
        data_collection=[
            {"field": "email", "expected": "a@b.com", "got": "a@b.com", "cer": 0.0, "match": True},
            {"field": "phone", "expected": "123", "got": "124", "cer": 0.333, "match": False},
        ],
    )
    ctx = MetricContext(verdicts={"probe1": v})
    result = DataCollectionMetric().compute(ctx)
    assert result.fields_checked == 2
    assert result.fields_correct == 1
    assert result.accuracy == 0.5
    assert result.avg_cer == pytest.approx((0.0 + 0.333) / 2)
