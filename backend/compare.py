from __future__ import annotations

from typing import Any, Dict, Optional

from evaluate import evaluate_episode
from policies import baseline_policy, improved_policy


def compare_policies(
    *,
    level: int,
    seed: int,
    max_steps: Optional[int] = None,
    include_trajectory: bool = False,
    include_ollama: bool = False,
    ollama_model: str = "llama3",
) -> Dict[str, Any]:
    baseline = evaluate_episode(
        baseline_policy,
        level=level,
        seed=seed,
        max_steps=max_steps,
        include_trajectory=include_trajectory,
    )
    improved = evaluate_episode(
        improved_policy,
        level=level,
        seed=seed,
        max_steps=max_steps,
        include_trajectory=include_trajectory,
    )
    result: Dict[str, Any] = {
        "baseline": baseline,
        "improved": improved,
        "difference": {
            "reward_gain": float(improved["total_reward"] - baseline["total_reward"]),
            "deaths_reduced": float(baseline["deaths"] - improved["deaths"]),
            "panic_reduction": float(baseline["panic_level"] - improved["panic_level"]),
        },
    }
    if include_ollama:
        from ollama_policy import OllamaPolicy

        ollama = OllamaPolicy(model=ollama_model)

        def ollama_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
            agent_id = obs.get("agent_status", {}).get("self", {}).get("id", "medical_agent")
            return ollama.generate_action(obs, agent_id)

        ollama_eval = evaluate_episode(
            ollama_policy,
            level=level,
            seed=seed,
            max_steps=max_steps,
            include_trajectory=include_trajectory,
        )
        result["ollama"] = ollama_eval
        result["difference"]["ollama_reward_gain_vs_baseline"] = float(
            ollama_eval["total_reward"] - baseline["total_reward"]
        )
        result["difference"]["ollama_deaths_reduced_vs_baseline"] = float(
            baseline["deaths"] - ollama_eval["deaths"]
        )
    return result
