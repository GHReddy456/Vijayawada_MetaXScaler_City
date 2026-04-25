"""
models.py - Strict Pydantic schemas for CrisisWorld.

All action/observation/communication/evaluation data structures are defined here.
environment.py, server.py, and train_grpo.py import from this module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 10
AGENT_IDS = (
    "medical_agent",
    "police_agent",
    "logistics_agent",
    "communication_agent",
    "commander_agent",
)
AGENT_ALIASES = {
    "medical": "medical_agent",
    "police": "police_agent",
    "logistics": "logistics_agent",
    "communication": "communication_agent",
    "commander": "commander_agent",
}
# 'secure' added: police/commander can secure a zone before medical dispatch
ACTION_TYPES = ("dispatch", "block", "route", "allocate", "broadcast", "secure")

SCENARIOS = ("earthquake", "flood", "blackout")

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class ActionModel(BaseModel):
    """A single agent action. Validated before entering the environment."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    action_type: Literal["dispatch", "block", "route", "allocate", "broadcast", "secure"]
    target: Tuple[int, int]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        normalized = AGENT_ALIASES.get(value, value)
        if normalized not in AGENT_IDS:
            raise ValueError(
                f"agent_id must be one of {AGENT_IDS} or aliases {tuple(AGENT_ALIASES)}"
            )
        return normalized

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        if len(value) != 2:
            raise ValueError("target must be [x, y]")
        x, y = value
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError(f"target {value} out of bounds for {GRID_SIZE}x{GRID_SIZE} grid")
        return value


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class ObservationModel(BaseModel):
    """Per-agent observation returned by environment.step / reset."""

    model_config = ConfigDict(extra="forbid")

    visible_events: List[Dict[str, Any]]
    resource_status: Dict[str, Any]
    agent_status: Dict[str, Any]
    messages: List[Dict[str, Any]]
    decision_context: Dict[str, Any]
    uncertainty: Dict[str, float]  # per-source uncertainty values in [0,1]
    time_step: int


# ---------------------------------------------------------------------------
# Structured inter-agent communication
# ---------------------------------------------------------------------------


class CommunicationMessage(BaseModel):
    """Structured message exchanged between agents.

    Used for reward shaping: correct usage → +10, ignored useful → -10,
    wrong info causing damage → -20.
    """

    model_config = ConfigDict(extra="allow")

    message_id: str
    sender: str
    recipient: Optional[str] = None  # None = broadcast to all
    msg_type: Literal["request", "advisory", "alert", "coordination", "route_advisory"]
    confidence: float = Field(ge=0.0, le=1.0)
    truth_score: float = Field(default=1.0, ge=0.0, le=1.0)
    content: Dict[str, Any] = Field(default_factory=dict)
    target: Tuple[int, int] = (0, 0)
    text: str = ""
    time_step: int = 0
    used: bool = False


# ---------------------------------------------------------------------------
# Per-agent trust snapshot
# ---------------------------------------------------------------------------


class AgentTrustSnapshot(BaseModel):
    """Snapshot of per-agent trust scores for logging and frontend display."""

    model_config = ConfigDict(extra="forbid")

    step: int
    trust: Dict[str, float]  # agent_id -> trust score [0, 100]


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


class StepResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Reset request
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal[1, 2, 3] = 1
    seed: int = 42
    scenario: Optional[Literal["earthquake", "flood", "blackout"]] = None


# ---------------------------------------------------------------------------
# Step log (for training trajectory)
# ---------------------------------------------------------------------------


class StepLogModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    observation: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    outcome: Dict[str, Any]


# ---------------------------------------------------------------------------
# Evaluation / benchmark result
# ---------------------------------------------------------------------------


class EpisodeResult(BaseModel):
    """Result of one evaluation episode."""

    model_config = ConfigDict(extra="allow")

    policy: str
    scenario: str
    level: int
    seed: int
    total_reward: float
    deaths: int
    panic_level: float
    trust_score: float
    coordination_score: float
    rescue_success_rate: float
    steps: int
    reward_per_step: List[float]
    trust_per_step: List[float]
    coordination_hits_per_step: List[int]


class BenchmarkResult(BaseModel):
    """Side-by-side comparison of baseline vs trained for one scenario."""

    model_config = ConfigDict(extra="allow")

    scenario: str
    level: int
    seeds: List[int]
    baseline: EpisodeResult
    trained: EpisodeResult
    delta: Dict[str, float]  # improvement metrics
