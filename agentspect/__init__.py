"""ATA — Agent Testing Agent.

A black-box, outside-in end-to-end testing framework for conversational agents.
ATA talks to an agent the way a real user would — over HTTP or WebSocket — and
verifies its behavior across generated positive and negative scenarios, without
any access to the agent's internals.

Basic usage::

    import asyncio
    from agentspect import run_suite

    report = asyncio.run(run_suite(open("config.yaml").read()))
    print(report["verdict_counts"])
    print(report["metrics"])
"""

from agentspect.adapters import (
    AgentCallable,
    CallableAdapter,
    HTTPAdapter,
    ProtocolAdapter,
    WebSocketAdapter,
)
from agentspect.agents.orchestrator import OrchestratorAgent, run_suite
from agentspect.assets import Asset, AssetLoader, load_assets, register_loader
from agentspect.llm.client import LLMClient, LLMResponse, create_llm_client
from agentspect.metrics import (
    Metric,
    MetricContext,
    MetricEngine,
    MetricRegistry,
    QuantitativeMetrics,
    compute_all_metrics,
    compute_metrics,
    register,
    registry,
)
from agentspect.models.suite import (
    Assertion,
    Scenario,
    ScenarioType,
    ScenarioVerdict,
    Verdict,
)
from agentspect.models.transcript import Transcript, Turn
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import YAMLInput
from agentspect.services.yaml_parser import YAMLValidationError, parse_and_validate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # entry points
    "run_suite",
    "OrchestratorAgent",
    "parse_and_validate",
    "YAMLValidationError",
    # LLM
    "create_llm_client",
    "LLMClient",
    "LLMResponse",
    # adapters
    "ProtocolAdapter",
    "HTTPAdapter",
    "WebSocketAdapter",
    "CallableAdapter",
    "AgentCallable",
    # assets
    "Asset",
    "AssetLoader",
    "register_loader",
    "load_assets",
    # domain models
    "YAMLInput",
    "WorldState",
    "Scenario",
    "ScenarioType",
    "ScenarioVerdict",
    "Verdict",
    "Assertion",
    "Transcript",
    "Turn",
    # metrics
    "compute_all_metrics",
    "compute_metrics",
    "QuantitativeMetrics",
    "Metric",
    "MetricContext",
    "MetricRegistry",
    "MetricEngine",
    "registry",
    "register",
]
