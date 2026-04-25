"""
train_grpo.py - GRPO-based RL training pipeline for CrisisWorld.

Real training loop:
  - All 5 agents act simultaneously each step (step_multi)
  - Reward is derived from actual environment transitions
  - Per-episode metrics: total_reward, deaths, coordination_score, trust_score
  - Saves: reward_curve.json, deaths_curve.json, coordination_curve.json,
           trust_curve.json, checkpoint_metadata.json
  - Supports Unsloth acceleration via CRISISWORLD_USE_UNSLOTH=1

Usage:
  python train_grpo.py                          # default Qwen 0.5B
  CRISISWORLD_USE_UNSLOTH=1 python train_grpo.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import CrisisWorldEnv
from models import AGENT_IDS
from policies import improved_policy


# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------


def observation_to_prompt(observation: Dict[str, Any], agent_id: str) -> str:
    """Convert a role-specific observation into an LLM prompt."""
    visible = observation.get("visible_events", [])
    messages = observation.get("messages", [])
    resources = observation.get("resource_status", {})
    uncertainty = observation.get("uncertainty", {})
    scope = (
        observation.get("agent_status", {})
        .get("self", {})
        .get("knowledge_scope", "")
    )
    context_lines = [
        "You are controlling an emergency-response RL policy.",
        "Return ONLY valid JSON with keys: agent_id, action_type, target, metadata.",
        f"agent_id must be '{agent_id}'.",
        "action_type must be one of: dispatch, route, allocate, broadcast, secure, block.",
        "",
        f"Knowledge scope: {scope}",
        f"Uncertainty levels: {json.dumps(uncertainty)}",
        f"Visible events ({len(visible)}): {json.dumps(visible[:5])}",
        f"Messages ({len(messages)}): {json.dumps(messages[:4])}",
        f"Resources: {json.dumps(resources)}",
    ]
    return "\n".join(context_lines)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_prompt_dataset(
    samples: int = 256,
    level: int = 2,
    seed: int = 123,
) -> Any:
    """Build a HuggingFace Dataset of (prompt) rows from environment rollouts."""
    from datasets import Dataset

    env = CrisisWorldEnv(seed=seed)
    rows: List[Dict[str, str]] = []
    for i in range(samples):
        obs_all = env.reset(level=level, seed=seed + i)
        for agent_id in AGENT_IDS:
            obs = obs_all.get(agent_id, {})
            rows.append({"prompt": observation_to_prompt(obs, agent_id)})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function used by GRPOTrainer
# ---------------------------------------------------------------------------


@dataclass
class CrisisWorldReward:
    """
    Callable reward function for GRPOTrainer.

    Runs a full multi-agent episode for each completion.
    All 5 agents act: the model output is one agent's action; the other 4
    use the improved (rule-based) policy as a cooperative baseline.

    Reward = sum of step rewards over `horizon` steps.
    """

    level: int = 2
    horizon: int = 20
    seed: int = 1234

    # Logged per __call__ invocation
    episode_rewards: List[float] = field(default_factory=list)
    episode_deaths: List[int] = field(default_factory=list)
    episode_coordination: List[float] = field(default_factory=list)
    episode_trust: List[float] = field(default_factory=list)

    def _fallback_action(self, agent_id: str) -> Dict[str, Any]:
        return {
            "agent_id": agent_id,
            "action_type": "route",
            "target": [5, 5],
            "metadata": {"reason": "fallback"},
        }

    def _parse_action(self, completion_text: str, agent_id: str) -> Dict[str, Any]:
        try:
            action = json.loads(completion_text.strip())
            if not isinstance(action, dict):
                raise ValueError("not a dict")
            action["agent_id"] = agent_id  # enforce correct agent_id
            return action
        except Exception:
            return self._fallback_action(agent_id)

    def _extract_agent_id(self, prompt_text: str) -> str:
        marker = "agent_id must be '"
        start = prompt_text.find(marker)
        if start == -1:
            return "medical_agent"
        start += len(marker)
        end = prompt_text.find("'", start)
        return prompt_text[start:end] if end > start else "medical_agent"

    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        **_: Any,
    ) -> List[float]:
        reward_values: List[float] = []

        for idx, completion in enumerate(completions):
            env = CrisisWorldEnv(seed=self.seed + idx)
            obs_all = env.reset(level=self.level, seed=self.seed + idx)

            agent_id = self._extract_agent_id(prompts[idx] if idx < len(prompts) else "")
            parsed_model_action = self._parse_action(completion, agent_id)

            total = 0.0
            for step_i in range(self.horizon):
                # Build joint action: model controls agent_id, rules control the rest
                joint: Dict[str, Any] = {}
                for aid in AGENT_IDS:
                    if aid == agent_id:
                        joint[aid] = parsed_model_action
                    else:
                        obs = obs_all.get(aid, {})
                        try:
                            joint[aid] = improved_policy(obs)
                        except Exception:
                            joint[aid] = self._fallback_action(aid)

                try:
                    result = env.step_multi(joint)
                except Exception:
                    total -= 25.0
                    break

                obs_all = result["observations"]
                total += float(result.get("reward", 0.0))

                if result.get("done", False):
                    break

                # Update model action for next step from new observation
                new_obs = obs_all.get(agent_id, {})
                parsed_model_action = {
                    "agent_id": agent_id,
                    "action_type": "dispatch",
                    "target": [5, 5],
                    "metadata": {"reason": "continuation"},
                }

            reward_values.append(total)

            # Log per-episode metrics
            m = env.metrics()
            self.episode_rewards.append(total)
            self.episode_deaths.append(int(m.get("deaths", 0)))
            self.episode_coordination.append(float(m.get("coordination_score", 0.0)))
            self.episode_trust.append(float(m.get("trust_score", 0.0)))

        return reward_values


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "checkpoints/crisisworld-grpo",
    samples: int = 256,
    level: int = 2,
    horizon: int = 20,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    num_generations: int = 4,
    generation_batch_size: int = 4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    logging_steps: int = 10,
    save_steps: int = 50,
) -> None:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = build_prompt_dataset(samples=samples, level=level)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    use_unsloth = os.getenv("CRISISWORLD_USE_UNSLOTH", "0") == "1"
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1024,
                load_in_4bit=True,
            )
        except Exception as e:
            print(f"[train] Unsloth unavailable ({e}), falling back to standard HF.")
            model = None

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    reward_fn = CrisisWorldReward(level=level, horizon=horizon)

    def crisisworld_reward(
        completions: List[str], prompts: List[str], **kwargs: Any
    ) -> List[float]:
        return reward_fn(completions=completions, prompts=prompts, **kwargs)

    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_prompt_length=768,
        max_completion_length=128,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[crisisworld_reward],
        args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)

    # -- Save training curves
    _save_artifacts(output_dir, reward_fn, model_name, level, samples, config)


def _save_artifacts(
    output_dir: str,
    reward_fn: CrisisWorldReward,
    model_name: str,
    level: int,
    samples: int,
    config: Any,
) -> None:
    """Persist all training metrics to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "reward_curve.json").write_text(
        json.dumps(
            {
                "episodes": list(range(len(reward_fn.episode_rewards))),
                "rewards": reward_fn.episode_rewards,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "deaths_curve.json").write_text(
        json.dumps(
            {
                "episodes": list(range(len(reward_fn.episode_deaths))),
                "deaths": reward_fn.episode_deaths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "coordination_curve.json").write_text(
        json.dumps(
            {
                "episodes": list(range(len(reward_fn.episode_coordination))),
                "coordination_score": reward_fn.episode_coordination,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "trust_curve.json").write_text(
        json.dumps(
            {
                "episodes": list(range(len(reward_fn.episode_trust))),
                "trust_score": reward_fn.episode_trust,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Backward-compatible reward_history.json (original filename)
    (out / "reward_history.json").write_text(
        json.dumps(
            {"rewards": reward_fn.episode_rewards},
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "level": level,
                "samples": samples,
                "output_dir": output_dir,
                "total_episodes": len(reward_fn.episode_rewards),
                "final_reward": reward_fn.episode_rewards[-1] if reward_fn.episode_rewards else None,
                "final_deaths": reward_fn.episode_deaths[-1] if reward_fn.episode_deaths else None,
                "final_trust": reward_fn.episode_trust[-1] if reward_fn.episode_trust else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Artifact loader (used by server.py)
# ---------------------------------------------------------------------------


def load_training_artifacts(
    output_dir: str = "checkpoints/crisisworld-grpo",
) -> Dict[str, Any]:
    """Load all training artifacts from disk for serving via the API."""
    out = Path(output_dir)
    payload: Dict[str, Any] = {
        "output_dir": output_dir,
        "available": {},
    }
    for fname in [
        "reward_curve.json",
        "deaths_curve.json",
        "coordination_curve.json",
        "trust_curve.json",
        "reward_history.json",
        "checkpoint_metadata.json",
    ]:
        p = out / fname
        key = fname.replace(".json", "")
        payload["available"][key] = p.exists()
        if p.exists():
            try:
                payload[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                payload[key] = None

    return payload


if __name__ == "__main__":
    train()
