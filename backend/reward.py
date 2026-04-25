"""
reward.py - Centralized reward computation for CrisisWorld.

All reward logic lives here. environment.py delegates to this module.
Reward is derived strictly from state transitions (state_t -> state_t+1).

Reward formula:
  reward =
    +50 * lives_saved
    -100 * deaths
    -2  * time_penalty (per step)
    +10 * coordination_success
    -20 * resource_overload
    -15 * misinformation_spread
    +5  * trust_gain
    -5  * repeated_action
    -8  * ignored_tasks
    -3  * communication_spam
    +14 * coordinated_police_medical_sequence
    -10 * unsafe_medical_dispatch
    -12 * triage_conflict
    +8  * strategic_tradeoff
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RewardWeights:
    """Multiplicative weights applied to each reward component."""

    survival: float = 1.0
    time_penalty: float = 1.0
    coordination: float = 1.0
    resource_efficiency: float = 1.0
    misinformation: float = 1.0
    repeated_action: float = 1.0
    ignored_tasks: float = 1.0
    long_term_stability: float = 1.0
    infrastructure_recovery: float = 1.0
    coordination_efficiency: float = 1.0
    trust_recovery: float = 1.0
    triage_conflict: float = 1.0
    strategic_tradeoff: float = 1.0


def compute_components(
    *,
    lives_saved: int,
    deaths: int,
    cooperation: bool,
    balanced_allocation: bool,
    overloaded: bool,
    misinformation: bool,
    repeated_action: bool,
    ignored_tasks: bool,
    cooperation_confidence: float,
    prev_snapshot: Dict[str, float],
    curr_panic: float,
    curr_trust: float,
    curr_infra: float,
    curr_deaths: int,
    triage_conflict: bool = False,
    strategic_tradeoff: bool = False,
    coordinated_sequence: bool = False,
    unsafe_medical_dispatch: bool = False,
    communication_spam: bool = False,
) -> Dict[str, float]:
    """
    Compute all reward components from a single state transition.

    All deltas are derived from (prev_snapshot -> current values) so
    the reward is grounded in measurable state change, not heuristics.
    """
    panic_delta = prev_snapshot["panic_level"] - curr_panic
    trust_delta = curr_trust - prev_snapshot["trust_score"]
    infra_delta = curr_infra - prev_snapshot["infrastructure_health"]
    death_delta = prev_snapshot["deaths"] - curr_deaths

    long_term_stability = 0.8 * panic_delta + 1.2 * death_delta
    infra_recovery = 0.5 * infra_delta
    coordination_eff = (
        (10.0 * cooperation_confidence)
        if cooperation
        else (-2.0 if curr_trust < 35 else 0.0)
    )
    trust_recovery = 0.4 * trust_delta
    trust_gain = 5.0 * max(0.0, trust_delta)

    return {
        "survival_reward": float(50 * lives_saved - 100 * deaths),
        "time_penalty": -2.0,
        "coordination_reward": float(10.0 * cooperation_confidence) if cooperation else 0.0,
        "resource_efficiency": 10.0 if balanced_allocation else (-20.0 if overloaded else 0.0),
        "misinformation_penalty": -15.0 if misinformation else 0.0,
        "repeated_action_penalty": -5.0 if repeated_action else 0.0,
        "ignored_task_penalty": -8.0 if ignored_tasks else 0.0,
        "communication_spam_penalty": -3.0 if communication_spam else 0.0,
        "long_term_stability_reward": float(long_term_stability),
        "infrastructure_recovery_reward": float(infra_recovery),
        "coordination_efficiency_reward": float(coordination_eff),
        "trust_recovery_reward": float(trust_recovery),
        "trust_gain_reward": float(trust_gain),
        "triage_conflict_penalty": -12.0 if triage_conflict else 0.0,
        "strategic_tradeoff_reward": 8.0 if strategic_tradeoff else 0.0,
        "coordinated_sequence_reward": 14.0 if coordinated_sequence else 0.0,
        "unsafe_medical_dispatch_penalty": -10.0 if unsafe_medical_dispatch else 0.0,
    }


def weighted_total(components: Dict[str, float], weights: RewardWeights) -> float:
    """Combine all reward components into a single scalar using weights."""
    w = weights
    return (
        w.survival * components["survival_reward"]
        + w.time_penalty * components["time_penalty"]
        + w.coordination * components["coordination_reward"]
        + w.resource_efficiency * components["resource_efficiency"]
        + w.misinformation * components["misinformation_penalty"]
        + w.repeated_action * components["repeated_action_penalty"]
        + w.ignored_tasks * components["ignored_task_penalty"]
        + components["communication_spam_penalty"]
        + w.long_term_stability * components["long_term_stability_reward"]
        + w.infrastructure_recovery * components["infrastructure_recovery_reward"]
        + w.coordination_efficiency * components["coordination_efficiency_reward"]
        + w.trust_recovery * components["trust_recovery_reward"]
        + components["trust_gain_reward"]
        + w.triage_conflict * components["triage_conflict_penalty"]
        + w.strategic_tradeoff * components["strategic_tradeoff_reward"]
        + components["coordinated_sequence_reward"]
        + components["unsafe_medical_dispatch_penalty"]
    )


def update_agent_trust(
    agent_trust: Dict[str, float],
    agent_id: str,
    correct_decision: bool,
    coordination_success: bool,
    misinformation: bool,
) -> None:
    """
    Update per-agent trust in-place based on decision quality.

    Trust rises for correct or coordinated decisions and falls for
    misinformation. Used both for conflict resolution and as an
    RL signal that agents can observe about other agents.
    """
    current = agent_trust.get(agent_id, 50.0)
    if coordination_success:
        current = min(100.0, current + 2.0)
    elif correct_decision:
        current = min(100.0, current + 0.5)
    if misinformation:
        current = max(0.0, current - 3.0)
    agent_trust[agent_id] = current


_COMPONENT_LABELS: Dict[str, str] = {
    "survival_reward": "medical-outcome",
    "time_penalty": "temporal-delay",
    "coordination_reward": "inter-agent-coordination",
    "resource_efficiency": "resource-network",
    "misinformation_penalty": "information-integrity",
    "repeated_action_penalty": "repeated-action",
    "ignored_task_penalty": "ignored-tasks",
    "communication_spam_penalty": "broadcast-spam",
    "long_term_stability_reward": "system-stability",
    "infrastructure_recovery_reward": "infrastructure-state",
    "coordination_efficiency_reward": "trust-weighted-communication",
    "trust_recovery_reward": "public-trust-dynamics",
    "trust_gain_reward": "trust-improvement",
    "triage_conflict_penalty": "capacity-triage-conflict",
    "strategic_tradeoff_reward": "strategic-prioritization",
    "coordinated_sequence_reward": "police-medical-sequence",
    "unsafe_medical_dispatch_penalty": "unsafe-dispatch",
}


def explain_dominant(components: Dict[str, float]) -> str:
    """Return human-readable label for the largest-magnitude component."""
    name, _ = max(components.items(), key=lambda kv: abs(kv[1]))
    return _COMPONENT_LABELS.get(name, name)


def reward_change_reason(components: Dict[str, float]) -> str:
    """Return one-line explanation of the dominant reward driver."""
    name, value = max(components.items(), key=lambda kv: abs(kv[1]))
    label = _COMPONENT_LABELS.get(name, name)
    if value >= 0:
        return f"Reward increased mostly due to {label}."
    return f"Reward dropped mostly due to {label}."
