import uuid

import pytest

from backend.tasks import (
    _map_depends_on_type,
    _map_scenario_type,
    _map_verdict,
    _scenario_uuid,
)
from backend.db.models import DependsOnTypeDB, ScenarioTypeDB, VerdictDB


class TestScenarioUUID:
    def test_deterministic(self):
        suite_id = uuid.UUID("12345678-1234-1234-1234-123456789012")
        u1 = _scenario_uuid(suite_id, "scenario-a")
        u2 = _scenario_uuid(suite_id, "scenario-a")
        assert u1 == u2

    def test_different_scenarios_different_uuids(self):
        suite_id = uuid.UUID("12345678-1234-1234-1234-123456789012")
        u1 = _scenario_uuid(suite_id, "scenario-a")
        u2 = _scenario_uuid(suite_id, "scenario-b")
        assert u1 != u2

    def test_different_suites_different_uuids(self):
        s1 = uuid.UUID("12345678-1234-1234-1234-123456789012")
        s2 = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        u1 = _scenario_uuid(s1, "scenario-a")
        u2 = _scenario_uuid(s2, "scenario-a")
        assert u1 != u2


class TestMapVerdict:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("success", VerdictDB.SUCCESS),
            ("success_unverified", VerdictDB.SUCCESS_UNVERIFIED),
            ("failure", VerdictDB.FAILURE),
            ("failure_corrupt", VerdictDB.FAILURE_CORRUPT),
            ("suspect", VerdictDB.SUSPECT),
            ("error", VerdictDB.ERROR),
        ],
    )
    def test_valid_verdicts(self, input_str, expected):
        assert _map_verdict(input_str) == expected

    def test_unknown_returns_none(self):
        assert _map_verdict("unknown") is None


class TestMapScenarioType:
    def test_positive(self):
        assert _map_scenario_type("positive") == ScenarioTypeDB.POSITIVE

    def test_negative(self):
        assert _map_scenario_type("negative") == ScenarioTypeDB.NEGATIVE


class TestMapDependsOnType:
    def test_probe(self):
        assert _map_depends_on_type("probe") == DependsOnTypeDB.PROBE

    def test_defensive_probe(self):
        assert _map_depends_on_type("defensive_probe") == DependsOnTypeDB.DEFENSIVE_PROBE

    def test_none(self):
        assert _map_depends_on_type(None) is None

    def test_unknown(self):
        assert _map_depends_on_type("something") is None


class TestMakeProgressCallback:
    def test_creates_callable(self):
        from backend.tasks import _make_progress_callback
        cb = _make_progress_callback("suite-123")
        assert callable(cb)
