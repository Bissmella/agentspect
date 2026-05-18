import pytest

from backend.models.suite import Scenario, ScenarioType, ScenarioVerdict, Verdict
from backend.services.dag import build_dag, cascade_skip, topological_batches


def make_scenario(id: str, depends_on: str | None = None) -> Scenario:
    return Scenario(
        id=id,
        type=ScenarioType.POSITIVE,
        description=f"Scenario {id}",
        turns=["Hello"],
        depends_on=depends_on,
    )


def test_build_dag_no_dependencies():
    scenarios = [make_scenario("a"), make_scenario("b"), make_scenario("c")]
    dag = build_dag(scenarios)
    assert dag == {"a": [], "b": [], "c": []}


def test_build_dag_with_dependencies():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
        make_scenario("c", depends_on="a"),
        make_scenario("d", depends_on="b"),
    ]
    dag = build_dag(scenarios)
    assert set(dag["a"]) == {"b", "c"}
    assert dag["b"] == ["d"]
    assert dag["c"] == []
    assert dag["d"] == []


def test_build_dag_invalid_dependency():
    scenarios = [make_scenario("a", depends_on="nonexistent")]
    with pytest.raises(ValueError, match="depends on unknown scenario"):
        build_dag(scenarios)


def test_topological_batches_no_dependencies():
    scenarios = [make_scenario("a"), make_scenario("b"), make_scenario("c")]
    batches = topological_batches(scenarios)
    assert len(batches) == 1
    assert set(batches[0]) == {"a", "b", "c"}


def test_topological_batches_linear():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
        make_scenario("c", depends_on="b"),
    ]
    batches = topological_batches(scenarios)
    assert len(batches) == 3
    assert batches[0] == ["a"]
    assert batches[1] == ["b"]
    assert batches[2] == ["c"]


def test_topological_batches_diamond():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
        make_scenario("c", depends_on="a"),
        make_scenario("d", depends_on="b"),
    ]
    batches = topological_batches(scenarios)

    assert batches[0] == ["a"]
    assert set(batches[1]) == {"b", "c"}
    assert batches[2] == ["d"]


def test_topological_batches_empty():
    batches = topological_batches([])
    assert batches == []


def test_cascade_skip():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
        make_scenario("c", depends_on="a"),
        make_scenario("d", depends_on="b"),
    ]
    dag = build_dag(scenarios)
    verdicts: dict[str, ScenarioVerdict] = {}

    skipped = cascade_skip(dag, "a", verdicts, "patch validation failed")

    assert skipped == {"b", "c", "d"}
    assert len(verdicts) == 3

    for sid in ["b", "c", "d"]:
        assert verdicts[sid].verdict == Verdict.ERROR
        assert "PATCH_FAILED cascade from a" in verdicts[sid].reason


def test_cascade_skip_partial():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
        make_scenario("c"),
        make_scenario("d", depends_on="b"),
    ]
    dag = build_dag(scenarios)
    verdicts: dict[str, ScenarioVerdict] = {}

    skipped = cascade_skip(dag, "a", verdicts, "error")

    assert skipped == {"b", "d"}
    assert "c" not in verdicts


def test_cascade_skip_leaf_node():
    scenarios = [
        make_scenario("a"),
        make_scenario("b", depends_on="a"),
    ]
    dag = build_dag(scenarios)
    verdicts: dict[str, ScenarioVerdict] = {}

    skipped = cascade_skip(dag, "b", verdicts, "error")

    assert skipped == set()
    assert len(verdicts) == 0
