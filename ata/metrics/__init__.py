"""ATA metrics.

Two ways to use this package:

- The extensible system — subclass :class:`Metric`, register it with
  :func:`register`, and run everything with :class:`MetricEngine` /
  :func:`compute_metrics`. Metrics can hook into each turn (see ``LatencyMetric``).
- The original six aggregate metrics as pure functions (``compute_task_completion``
  …) and :func:`compute_all_metrics`, kept for backward compatibility.
"""

from ata.metrics.base import Metric, MetricContext

# Importing the built-in classes runs ata.metrics.builtin, whose @register
# decorators populate the default registry.
from ata.metrics.builtin import (
    AgentSpeechDurationMetric,
    AgentSpeechDurationResult,
    BoundaryAdherence,
    ConstraintViolations,
    ConversationEfficiency,
    DeadAirMetric,
    DeadAirResult,
    GreetingRateMetric,
    GreetingRateResult,
    IntelligibilityMetric,
    IntelligibilityResult,
    LatencyMetric,
    LatencyResult,
    NoResponseRateMetric,
    NoResponseRateResult,
    PrematureDisconnectMetric,
    PrematureDisconnectResult,
    RecoveryBehavior,
    TaskCompletion,
    TimeToFirstAudioMetric,
    TimeToFirstAudioResult,
    TurnErrorRateMetric,
    TurnErrorRateResult,
    VerificationRate,
)
from ata.metrics.core import (
    BoundaryAdherenceMetric,
    ConstraintViolation,
    ConversationEfficiencyMetric,
    QuantitativeMetrics,
    RecoveryBehaviorMetric,
    TaskCompletionMetric,
    VerificationRateMetric,
    compute_all_metrics,
    compute_boundary_adherence,
    compute_constraint_violations,
    compute_conversation_efficiency,
    compute_recovery_behavior,
    compute_task_completion,
    compute_verification_rate,
)
from ata.metrics.engine import MetricEngine, compute_metrics
from ata.metrics.registry import MetricRegistry, register, registry

__all__ = [
    # extensible system
    "Metric",
    "MetricContext",
    "MetricRegistry",
    "registry",
    "register",
    "MetricEngine",
    "compute_metrics",
    # built-in metric classes
    "TaskCompletion",
    "BoundaryAdherence",
    "VerificationRate",
    "ConstraintViolations",
    "RecoveryBehavior",
    "ConversationEfficiency",
    "LatencyMetric",
    "TurnErrorRateMetric",
    "TimeToFirstAudioMetric",
    "NoResponseRateMetric",
    "GreetingRateMetric",
    "PrematureDisconnectMetric",
    "IntelligibilityMetric",
    "DeadAirMetric",
    "AgentSpeechDurationMetric",
    # result models
    "LatencyResult",
    "TurnErrorRateResult",
    "TimeToFirstAudioResult",
    "NoResponseRateResult",
    "GreetingRateResult",
    "PrematureDisconnectResult",
    "IntelligibilityResult",
    "DeadAirResult",
    "AgentSpeechDurationResult",
    "QuantitativeMetrics",
    "TaskCompletionMetric",
    "BoundaryAdherenceMetric",
    "VerificationRateMetric",
    "ConstraintViolation",
    "RecoveryBehaviorMetric",
    "ConversationEfficiencyMetric",
    # legacy pure functions
    "compute_all_metrics",
    "compute_task_completion",
    "compute_boundary_adherence",
    "compute_verification_rate",
    "compute_constraint_violations",
    "compute_recovery_behavior",
    "compute_conversation_efficiency",
]
