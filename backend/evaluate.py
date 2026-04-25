"""
evaluate.py - Benchmark evaluation pipeline for CrisisWorld.

Runs baseline and trained policies on identical scenarios (same seeds) and
produces a structured comparison JSON.  Both policies use step_multi so all
5 agents act simultaneously per step — same conditions as training.

Tracked per episode:
  - total_reward
  - deaths
  - panic_level
  - trust_score
  - coordination_score
  - rescue_success_rate
  - reward_per_step  (for reward curve)
  - trust_per_step   (for trust evolution curve)
  - coordination_hits_per_step

Output: BenchmarkResult (from models.py) which is also serialisable as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from environment import CrisisWorldEnv
from models import AGENT_IDS, BenchmarkResult, EpisodeResult

PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Single-episode runner (multi-agent, step_multi)
# ---------------------------------------------------------------------------


def _fallback_action(agent_id: str, pos: List[int]) -> Dict[str, Any]:
    return {
        "agent_id": agent_id,
        "action_type": "route",
        "target": [int(pos[0]), int(pos[1])],
        "metadata": {"reason": "evaluation fallback"},
    }


def evaluate_episode(
    policy_fn: PolicyFn,
    *,
    level: int,
    seed: int,
    max_steps: Optional[int] = None,
    include_trajectory: bool = False,
    policy_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Run one episode with `policy_fn` applied to ALL agents simultaneously.

    policy_fn receives a single-agent observation dict and returns a single
    action dict.  The episode advances via step_multi each tick.
    """
    env = CrisisWorldEnv(seed=seed, max_steps=max_steps or 100)
    obs_all = env.reset(level=level, seed=seed)

    reward_per_step: List[float] = []
    trust_per_step: List[float] = []
    coordination_hits_per_step: List[int] = []
    done = False
    step_index = 0

    while not done:
        joint: Dict[str, Any] = {}
        for aid in AGENT_IDS:
            obs = obs_all.get(aid, {})
            try:
                joint[aid] = policy_fn(obs)
            except Exception:
                pos = obs.get("agent_status", {}).get("self", {}).get("pos", [0, 0])
                joint[aid] = _fallback_action(aid, pos)

        try:
            result = env.step_multi(joint)
        except Exception:
            break

        obs_all = result["observations"]
        reward_per_step.append(float(result.get("reward", 0.0)))
        trust_per_step.append(float(env.trust_score))
        coordination_hits_per_step.append(int(env.coordination_hits))
        done = bool(result.get("done", False))
        step_index += 1

        if max_steps is not None and step_index >= max_steps:
            break

    metrics = env.metrics()
    total_cas = max(1, len(env.casualties))
    treated = sum(1 for c in env.casualties if c.get("status") == "treated")

    summary: Dict[str, Any] = {
        "policy": policy_name,
        "scenario": env.scenario,
        "level": level,
        "seed": seed,
        "total_reward": float(metrics["total_reward"]),
        "deaths": int(metrics["deaths"]),
        "panic_level": float(metrics["panic_level"]),
        "trust_score": float(metrics["trust_score"]),
        "coordination_score": float(metrics["coordination_score"]),
        "rescue_success_rate": float(treated / total_cas * 100),
        "steps": step_index,
        "reward_per_step": reward_per_step,
        "trust_per_step": trust_per_step,
        "coordination_hits_per_step": coordination_hits_per_step,
    }
    if include_trajectory:
        summary["trajectory"] = env.trajectory
    return summary


# ---------------------------------------------------------------------------
# Multi-seed benchmark: baseline vs trained / improved
# ---------------------------------------------------------------------------


def run_benchmark(
    baseline_policy: PolicyFn,
    trained_policy: PolicyFn,
    *,
    level: int = 2,
    seeds: Optional[List[int]] = None,
    max_steps: int = 60,
    baseline_name: str = "baseline",
    trained_name: str = "trained",
) -> Dict[str, Any]:
    """
    Run both policies on each seed and return averaged comparison results.

    Averaging over seeds removes seed variance from the comparison,
    making the delta attributable to policy differences only.
    """
    if seeds is None:
        seeds = [42, 100, 200, 300, 400]

    baseline_results: List[Dict[str, Any]] = []
    trained_results: List[Dict[str, Any]] = []

    for seed in seeds:
        br = evaluate_episode(
            baseline_policy,
            level=level,
            seed=seed,
            max_steps=max_steps,
            policy_name=baseline_name,
        )
        tr = evaluate_episode(
            trained_policy,
            level=level,
            seed=seed,
            max_steps=max_steps,
            policy_name=trained_name,
        )
        baseline_results.append(br)
        trained_results.append(tr)

    def _avg(results: List[Dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def _concat_curves(results: List[Dict[str, Any]], key: str) -> List[float]:
        out: List[float] = []
        for r in results:
            out.extend([float(v) for v in r.get(key, [])])
        return out

    scenario = baseline_results[0]["scenario"] if baseline_results else "unknown"

    # Use last seed's result for EpisodeResult (representative)
    last_b = baseline_results[-1] if baseline_results else {}
    last_t = trained_results[-1] if trained_results else {}

    delta = {
        "reward_gain": _avg(trained_results, "total_reward") - _avg(baseline_results, "total_reward"),
        "deaths_reduced": _avg(baseline_results, "deaths") - _avg(trained_results, "deaths"),
        "panic_reduction": _avg(baseline_results, "panic_level") - _avg(trained_results, "panic_level"),
        "trust_improvement": _avg(trained_results, "trust_score") - _avg(baseline_results, "trust_score"),
        "coordination_gain": _avg(trained_results, "coordination_score") - _avg(baseline_results, "coordination_score"),
        "rescue_rate_gain": _avg(trained_results, "rescue_success_rate") - _avg(baseline_results, "rescue_success_rate"),
    }

    return {
        "scenario": scenario,
        "level": level,
        "seeds": seeds,
        "baseline_avg": {
            "total_reward": _avg(baseline_results, "total_reward"),
            "deaths": _avg(baseline_results, "deaths"),
            "panic_level": _avg(baseline_results, "panic_level"),
            "trust_score": _avg(baseline_results, "trust_score"),
            "coordination_score": _avg(baseline_results, "coordination_score"),
            "rescue_success_rate": _avg(baseline_results, "rescue_success_rate"),
            "reward_curve": _concat_curves(baseline_results, "reward_per_step"),
            "trust_curve": _concat_curves(baseline_results, "trust_per_step"),
            "coordination_curve": _concat_curves(baseline_results, "coordination_hits_per_step"),
        },
        "trained_avg": {
            "total_reward": _avg(trained_results, "total_reward"),
            "deaths": _avg(trained_results, "deaths"),
            "panic_level": _avg(trained_results, "panic_level"),
            "trust_score": _avg(trained_results, "trust_score"),
            "coordination_score": _avg(trained_results, "coordination_score"),
            "rescue_success_rate": _avg(trained_results, "rescue_success_rate"),
            "reward_curve": _concat_curves(trained_results, "reward_per_step"),
            "trust_curve": _concat_curves(trained_results, "trust_per_step"),
            "coordination_curve": _concat_curves(trained_results, "coordination_hits_per_step"),
        },
        "delta": delta,
        "per_seed": {
            "baseline": baseline_results,
            "trained": trained_results,
        },
    }


# ---------------------------------------------------------------------------
# Evaluation CLI
# ---------------------------------------------------------------------------


def run_and_save(
    output_path: str = "checkpoints/crisisworld-grpo/benchmark_results.json",
    level: int = 2,
    seeds: Optional[List[int]] = None,
    max_steps: int = 60,
) -> Dict[str, Any]:
    """
    Run baseline vs improved policy benchmark and save results to JSON.
    Called from server.py to populate baseline_metrics on startup.
    """
    from policies import baseline_policy, improved_policy

    results = run_benchmark(
        baseline_policy=baseline_policy,
        trained_policy=improved_policy,
        level=level,
        seeds=seeds or [42, 100, 200],
        max_steps=max_steps,
        baseline_name="baseline",
        trained_name="improved",
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def load_benchmark_results(
    path: str = "checkpoints/crisisworld-grpo/benchmark_results.json",
) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    results = run_and_save()
    print(json.dumps(results["delta"], indent=2))
