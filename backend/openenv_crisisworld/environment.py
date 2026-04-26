from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from environment import CrisisWorldEnv
from models import ActionModel
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from .models import CrisisWorldAction, CrisisWorldObservation


class CrisisWorldOpenEnvironment(Environment[CrisisWorldAction, CrisisWorldObservation, State]):
    """
    OpenEnv `Environment` adapter over the existing CrisisWorld simulator.

    Each session owns a fresh `CrisisWorldEnv` instance. Use this entrypoint for
    Hugging Face Spaces, TRL/OpenEnv clients, and hackathon judging.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__(transform=None, rubric=None)
        self._cw = CrisisWorldEnv()
        self._episode = State(episode_id=str(uuid4()), step_count=0)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="crisisworld",
            description=(
                "Multi-agent disaster response on a 10x10 grid: medical, police, "
                "logistics, communication, and commander agents with partial observability."
            ),
            version="1.0.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CrisisWorldObservation:
        level = int(kwargs.get("level", 1))
        scenario = kwargs.get("scenario")
        s = 42 if seed is None else int(seed)
        obs_map = self._cw.reset(level=level, seed=s, scenario=scenario)
        self._episode = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return self._apply_transform(
            CrisisWorldObservation(
                done=False,
                reward=None,
                metadata={
                    "observations": obs_map,
                    "level": level,
                    "scenario": self._cw.scenario,
                    "grid_size": self._cw.grid_size,
                    "message": "CrisisWorld episode started",
                },
            )
        )

    def step(
        self,
        action: CrisisWorldAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CrisisWorldObservation:
        del timeout_s, kwargs
        am = ActionModel(
            agent_id=action.agent_id,
            action_type=action.action_type,  # type: ignore[arg-type]
            target=action.target,
            metadata=dict(action.metadata or {}),
        )
        raw = self._cw.step(am)
        self._episode = State(
            episode_id=self._episode.episode_id,
            step_count=self._cw.step_count,
        )
        return self._apply_transform(
            CrisisWorldObservation(
                done=bool(raw["done"]),
                reward=float(raw["reward"]),
                metadata={
                    "observations": raw["observations"],
                    "info": raw["info"],
                },
            )
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode.episode_id,
            step_count=self._cw.step_count,
        )
