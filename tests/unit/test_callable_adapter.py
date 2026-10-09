import pytest

from agentspect.adapters.callable_adapter import CallableAdapter, _accepts_history
from agentspect.agents.orchestrator import OrchestratorAgent

_CALLABLE_YAML = """
agent_under_test:
  name: "In-process Agent"
  protocol: callable
  description: "A callable agent"

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""


class TestAcceptsHistory:
    def test_single_arg(self):
        def agent(message):
            return message
        assert _accepts_history(agent) is False

    def test_two_args(self):
        def agent(message, history):
            return message
        assert _accepts_history(agent) is True

    def test_var_positional(self):
        def agent(*args):
            return ""
        assert _accepts_history(agent) is True

    def test_lambda_single(self):
        assert _accepts_history(lambda m: m) is False


class TestCallableAdapter:
    async def test_rejects_non_callable(self):
        with pytest.raises(TypeError):
            CallableAdapter("not callable")

    async def test_sync_agent_single_arg(self):
        adapter = CallableAdapter(lambda m: f"echo: {m}")
        sid = await adapter.start_session("s1")
        turn = await adapter.send_turn(sid, "hello")
        assert turn.agent_response == "echo: hello"
        assert turn.error is None
        assert turn.latency_ms >= 0

    async def test_async_agent(self):
        async def agent(message):
            return message.upper()
        adapter = CallableAdapter(agent)
        sid = await adapter.start_session("s1")
        turn = await adapter.send_turn(sid, "hi")
        assert turn.agent_response == "HI"
        assert turn.error is None

    async def test_history_is_passed_and_accumulates(self):
        seen: list[list[dict]] = []

        def agent(message, history):
            seen.append(list(history))
            return f"turn {len(history) + 1}"

        adapter = CallableAdapter(agent)
        sid = await adapter.start_session("s1")
        t1 = await adapter.send_turn(sid, "first")
        t2 = await adapter.send_turn(sid, "second")

        assert t1.agent_response == "turn 1"
        assert t2.agent_response == "turn 2"
        # First call saw empty history; second saw the first completed turn.
        assert seen[0] == []
        assert seen[1] == [{"user": "first", "agent": "turn 1"}]

    async def test_sessions_are_isolated(self):
        def agent(message, history):
            return str(len(history))
        adapter = CallableAdapter(agent)
        s1 = await adapter.start_session("sc1")
        s2 = await adapter.start_session("sc2")
        await adapter.send_turn(s1, "a")
        turn = await adapter.send_turn(s2, "b")
        assert turn.agent_response == "0"  # s2 has its own empty history

    async def test_unknown_session_returns_error_turn(self):
        adapter = CallableAdapter(lambda m: m)
        turn = await adapter.send_turn("nope", "hi")
        assert turn.error is not None
        assert "Session not found" in turn.error

    async def test_exception_becomes_error_turn(self):
        def agent(message):
            raise RuntimeError("boom")
        adapter = CallableAdapter(agent)
        sid = await adapter.start_session("s1")
        turn = await adapter.send_turn(sid, "hi")
        assert turn.agent_response == ""
        assert turn.error is not None
        assert "RuntimeError" in turn.error and "boom" in turn.error

    async def test_non_string_return_is_coerced(self):
        adapter = CallableAdapter(lambda m: 42)
        sid = await adapter.start_session("s1")
        turn = await adapter.send_turn(sid, "hi")
        assert turn.agent_response == "42"

    async def test_end_and_close_clear_sessions(self):
        adapter = CallableAdapter(lambda m: m)
        sid = await adapter.start_session("s1")
        await adapter.end_session(sid)
        turn = await adapter.send_turn(sid, "hi")
        assert turn.error is not None and "Session not found" in turn.error
        await adapter.close()  # no-op, should not raise


class TestOrchestratorCallableWiring:
    async def test_agent_builds_callable_adapter(self):
        orch = OrchestratorAgent(_CALLABLE_YAML, agent=lambda m: m)
        await orch.initialize()
        assert isinstance(orch._adapter, CallableAdapter)

    async def test_callable_protocol_without_agent_raises(self):
        orch = OrchestratorAgent(_CALLABLE_YAML)  # no agent supplied
        with pytest.raises(RuntimeError, match="callable"):
            await orch.initialize()
