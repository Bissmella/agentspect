from collections import defaultdict, deque

from backend.models.suite import Scenario, ScenarioVerdict, Verdict


def build_dag(scenarios: list[Scenario]) -> dict[str, list[str]]:
    dag: dict[str, list[str]] = defaultdict(list)

    scenario_ids = {s.id for s in scenarios}

    for scenario in scenarios:
        dag[scenario.id]
        if scenario.depends_on:
            if scenario.depends_on not in scenario_ids:
                raise ValueError(
                    f"Scenario '{scenario.id}' depends on unknown scenario "
                    f"'{scenario.depends_on}'"
                )
            dag[scenario.depends_on].append(scenario.id)

    return dict(dag)


def _compute_in_degrees(
    scenarios: list[Scenario],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    in_degree: dict[str, int] = {s.id: 0 for s in scenarios}
    dependents: dict[str, list[str]] = defaultdict(list)

    for scenario in scenarios:
        if scenario.depends_on:
            in_degree[scenario.id] += 1
            dependents[scenario.depends_on].append(scenario.id)

    return in_degree, dict(dependents)


def topological_batches(scenarios: list[Scenario]) -> list[list[str]]:
    if not scenarios:
        return []

    in_degree, dependents = _compute_in_degrees(scenarios)
    batches: list[list[str]] = []
    queue = deque([sid for sid, deg in in_degree.items() if deg == 0])

    while queue:
        batch = list(queue)
        batches.append(batch)
        queue.clear()

        for sid in batch:
            for dependent in dependents.get(sid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    total_scheduled = sum(len(b) for b in batches)
    if total_scheduled != len(scenarios):
        raise ValueError("Cycle detected in scenario dependencies")

    return batches


def cascade_skip(
    dag: dict[str, list[str]],
    failed_scenario_id: str,
    verdicts: dict[str, ScenarioVerdict],
    reason: str,
) -> set[str]:
    skipped: set[str] = set()
    queue = deque(dag.get(failed_scenario_id, []))

    while queue:
        scenario_id = queue.popleft()
        if scenario_id in skipped:
            continue

        skipped.add(scenario_id)
        verdicts[scenario_id] = ScenarioVerdict(
            scenario_id=scenario_id,
            verdict=Verdict.ERROR,
            reason=f"PATCH_FAILED cascade from {failed_scenario_id}: {reason}",
        )

        queue.extend(dag.get(scenario_id, []))

    return skipped
