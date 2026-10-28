import json

import pytest

from agentspect.agents.asset_ingestion import IngestedEntities, asset_ingestion_node
from agentspect.assets import (
    Asset,
    AssetLoadError,
    infer_format,
    load_asset,
    loader_registry,
    stratified_indices,
)
from agentspect.assets.loaders import CSVLoader, JSONLLoader
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import AssetSpec


# ── Loaders ───────────────────────────────────────────────────────────────

def _write_csv(tmp_path, n):
    p = tmp_path / "data.csv"
    lines = ["id,name,verified"]
    lines += [f"c{i},Name{i},{'true' if i % 2 == 0 else 'false'}" for i in range(n)]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def _write_jsonl(tmp_path, n):
    p = tmp_path / "data.jsonl"
    p.write_text(
        "\n".join(json.dumps({"id": f"c{i}", "name": f"Name{i}"}) for i in range(n)),
        encoding="utf-8",
    )
    return str(p)


class TestCSVLoader:
    def test_count_excludes_header(self, tmp_path):
        assert CSVLoader().count(_write_csv(tmp_path, 10)) == 10

    def test_read_rows_by_index(self, tmp_path):
        rows = CSVLoader().read_rows(_write_csv(tmp_path, 10), [0, 3, 9])
        assert [r["id"] for r in rows] == ["c0", "c3", "c9"]
        assert rows[0]["name"] == "Name0"

    def test_read_rows_empty(self, tmp_path):
        assert CSVLoader().read_rows(_write_csv(tmp_path, 5), []) == []


class TestJSONLLoader:
    def test_count(self, tmp_path):
        assert JSONLLoader().count(_write_jsonl(tmp_path, 7)) == 7

    def test_read_rows_by_index(self, tmp_path):
        rows = JSONLLoader().read_rows(_write_jsonl(tmp_path, 7), [1, 5])
        assert [r["id"] for r in rows] == ["c1", "c5"]


class TestInferFormat:
    @pytest.mark.parametrize("name,fmt", [
        ("a.csv", "csv"), ("a.jsonl", "jsonl"), ("a.ndjson", "jsonl"), ("a.txt", None),
    ])
    def test_infer(self, name, fmt):
        assert infer_format(name) == fmt

    def test_registry_has_builtin_loaders(self):
        assert "csv" in loader_registry and "jsonl" in loader_registry


# ── Sampling ──────────────────────────────────────────────────────────────

class TestSampling:
    def test_spread_and_bounded(self):
        idx = stratified_indices(1000, 10, seed=0)
        assert len(idx) <= 10
        assert idx == sorted(idx)
        assert all(0 <= i < 1000 for i in idx)
        # spread across the file, not clustered at the head
        assert max(idx) > 500

    def test_reproducible(self):
        assert stratified_indices(1000, 10, seed=42) == stratified_indices(1000, 10, seed=42)

    def test_sample_larger_than_total(self):
        assert stratified_indices(3, 10) == [0, 1, 2]

    def test_empty(self):
        assert stratified_indices(0, 5) == []


# ── Service ───────────────────────────────────────────────────────────────

class TestLoadAsset:
    def test_loads_and_samples(self, tmp_path):
        spec = AssetSpec(id="customers", path=_write_csv(tmp_path, 100), sample_size=5)
        asset = load_asset(spec)
        assert asset.format == "csv"
        assert asset.total_rows == 100
        assert 1 <= len(asset.rows) <= 5
        assert asset.content_hash is not None

    def test_relative_path_uses_base_dir(self, tmp_path):
        _write_csv(tmp_path, 10)
        spec = AssetSpec(id="c", path="data.csv", sample_size=3)
        asset = load_asset(spec, base_dir=str(tmp_path))
        assert asset.total_rows == 10

    def test_missing_file(self, tmp_path):
        spec = AssetSpec(id="c", path=str(tmp_path / "nope.csv"))
        with pytest.raises(AssetLoadError, match="file not found"):
            load_asset(spec)

    def test_uninferable_format(self, tmp_path):
        p = tmp_path / "data.dat"
        p.write_text("x", encoding="utf-8")
        spec = AssetSpec(id="c", path=str(p))
        with pytest.raises(AssetLoadError, match="cannot infer format"):
            load_asset(spec)


# ── Ingestion agent ───────────────────────────────────────────────────────

class _MockLLM:
    def __init__(self, entities=None, raise_exc=False):
        self._entities = entities or []
        self._raise = raise_exc

    async def chat_with_structured_output(self, messages, model, temperature=0.0):
        if self._raise:
            raise RuntimeError("llm down")
        return IngestedEntities(entities=self._entities)


def _state(assets):
    ws = WorldState.from_dict({"entities": [], "catalog": {}, "constraints": [], "context": {}})
    return {"assets": assets, "world_state": ws}, ws


class TestAssetIngestionAgent:
    async def test_entities_role_merges_llm_output(self):
        asset = Asset(id="customers", format="csv", role="entities",
                      rows=[{"name": "A", "phone": "1"}], total_rows=50)
        state, ws = _state([asset])
        llm = _MockLLM(entities=[{"id": "c1", "name": "A"}, {"id": "c2", "name": "B"}])
        await asset_ingestion_node(state, llm)
        assert ws.data["entities"] == [{"id": "c1", "name": "A"}, {"id": "c2", "name": "B"}]

    async def test_entities_appends_to_existing(self):
        asset = Asset(id="customers", format="csv", role="entities", rows=[{"x": 1}], total_rows=1)
        ws = WorldState.from_dict({"entities": [{"id": "existing"}], "catalog": {}})
        state = {"assets": [asset], "world_state": ws}
        await asset_ingestion_node(state, _MockLLM(entities=[{"id": "new"}]))
        assert [e["id"] for e in ws.data["entities"]] == ["existing", "new"]

    async def test_entities_fallback_on_llm_error(self):
        asset = Asset(id="customers", format="csv", role="entities",
                      rows=[{"name": "A"}, {"name": "B"}], total_rows=2)
        state, ws = _state([asset])
        await asset_ingestion_node(state, _MockLLM(raise_exc=True))
        # fell back to raw rows, with synthesized ids
        assert len(ws.data["entities"]) == 2
        assert ws.data["entities"][0]["id"] == "customers_0"

    async def test_data_sample_role_is_deterministic(self):
        rows = [{"a": 1}, {"a": 2}]
        asset = Asset(id="sales", format="jsonl", role="data_sample", rows=rows, total_rows=999)
        state, ws = _state([asset])
        await asset_ingestion_node(state, _MockLLM(raise_exc=True))  # LLM must not be needed
        assert ws.data["catalog"]["sales"] == rows

    async def test_target_override(self):
        asset = Asset(id="c", format="csv", role="entities", rows=[{"x": 1}],
                      total_rows=1, target="/catalog/people")
        state, ws = _state([asset])
        await asset_ingestion_node(state, _MockLLM(entities=[{"id": "p1"}]))
        assert ws.data["catalog"]["people"] == [{"id": "p1"}]

    async def test_knowledge_base_builds_scenarios_without_touching_world_state(self):
        rows = [
            {"question": "What are your hours?", "answer": "9 to 5, Mon-Fri."},
            {"question": "Do you ship abroad?", "answer": "Yes, worldwide."},
        ]
        asset = Asset(id="kb", format="csv", role="knowledge_base", rows=rows, total_rows=100)
        state, ws = _state([asset])
        result = await asset_ingestion_node(state, _MockLLM(raise_exc=True))  # no LLM needed
        kb = result["kb_scenarios"]
        assert len(kb) == 2
        assert kb[0].turns == ["What are your hours?"]
        assert kb[0].knowledge_base == {"question": "What are your hours?", "answer": "9 to 5, Mon-Fri."}
        assert "9 to 5" in kb[0].assertions[0].check  # reference answer embedded in the check
        assert ws.data["catalog"] == {}  # KB pairs never enter world_state

    async def test_knowledge_base_falls_back_to_first_two_columns(self):
        rows = [{"prod": "X100", "spec": "waterproof"}]  # no question/answer headers
        asset = Asset(id="kb", format="csv", role="knowledge_base", rows=rows, total_rows=1)
        state, _ = _state([asset])
        result = await asset_ingestion_node(state, _MockLLM())
        kb = result["kb_scenarios"]
        assert kb[0].knowledge_base == {"question": "X100", "answer": "waterproof"}

    async def test_knowledge_base_skips_blank_pairs(self):
        rows = [{"question": "Q1", "answer": ""}, {"question": "Q2", "answer": "A2"}]
        asset = Asset(id="kb", format="csv", role="knowledge_base", rows=rows, total_rows=2)
        state, _ = _state([asset])
        result = await asset_ingestion_node(state, _MockLLM())
        assert [s.knowledge_base["question"] for s in result["kb_scenarios"]] == ["Q2"]

    async def test_no_assets_is_noop(self):
        state, ws = _state([])
        result = await asset_ingestion_node(state, _MockLLM())
        assert result["status"] == "no_assets"
