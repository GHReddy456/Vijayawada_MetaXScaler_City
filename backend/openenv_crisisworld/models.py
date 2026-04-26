from __future__ import annotations

from typing import Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CrisisWorldAction(Action):
    """Single-agent action compatible with CrisisWorld `ActionModel` validation."""

    agent_id: str = Field(..., description="One of the five CrisisWorld agent IDs")
    action_type: str = Field(
        ...,
        description="dispatch | route | block | allocate | broadcast | secure",
    )
    target: Tuple[int, int] = Field(
        default=(0, 0),
        description="Grid target [x, y] on the 10x10 map",
    )


class CrisisWorldObservation(Observation):
    """Observation; full payloads live in `metadata` for WebSocket/HTTP clients."""

    pass
