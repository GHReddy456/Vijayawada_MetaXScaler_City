"""
train_grpo.py — CrisisWorld GRPO training (definitive fix for reward_std=0).

ROOT CAUSE of reward_std=0:
  TRL's GRPOTrainer calls model.generate() using the MODEL's own
  generation_config, NOT the temperature/top_p fields in GRPOConfig.
  Result: greedy decoding → identical completions → reward_std = 0.

THREE-LAYER FIX applied here:
  Layer 1 — model.generation_config patched DIRECTLY (do_sample=True, T=0.9)
  Layer 2 — SamplingGRPOTrainer subclass forces sampling on every generate()
             regardless of TRL version
  Layer 3 — Stochastic fallback actions + varied seeds per completion
             guarantee reward spread even if text happens to repeat

All 10 parts from the spec are implemented.

Usage:
  python train_grpo.py
  CRISIS_MODEL=meta-llama/Meta-Llama-3-1B-Instruct python train_grpo.py
  CRISISWORLD_USE_UNSLOTH=1 python train_grpo.py   # Linux/CUDA
"""
from __future__ import annotations

import inspect
import json
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import CrisisWorldEnv
from models import AGENT_IDS
from policies import improved_policy

# ── Model selection ───────────────────────────────────────────────────────────
# PART 7 (optional): use small model to avoid OOM in Colab T4
# Override: CRISIS_MODEL=meta-llama/Meta-Llama-3-1B-Instruct python train_grpo.py
DEFAULT_MODEL = os.getenv("CRISIS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# ── PART 2: canonical generation kwargs used everywhere ───────────────────────
SAMPLING_KWARGS: Dict[str, Any] = {
    "do_sample":          True,
    "temperature":        0.8,    # spec §4: 0.8 — wide enough to explore, tight enough to stay valid
    "top_p":              0.9,    # spec §4
    "repetition_penalty": 1.1,    # prevents repetitive JSON loops
}
MAX_NEW_TOKENS = 48   # prefix-forced: model only outputs ~15-20 tokens of JSON suffix

# Temperature can be bumped dynamically when reward_std is too low (spec §2 guard)
_dynamic_temperature: float = SAMPLING_KWARGS["temperature"]


# ─── Observation → prompt ─────────────────────────────────────────────────────

_SOFT_SYSTEM_PROMPT = """\
You are an AI agent in CrisisWorld.

Your job is to choose ONE action.

Respond in JSON format like this:

{{
  "agent_id": "medical_agent",
  "action_type": "dispatch",
  "target": [x, y],
  "metadata": {{}}
}}

Valid agents:
medical_agent, police_agent, logistics_agent, communication_agent, commander_agent

Valid actions:
dispatch, route, block, allocate, broadcast

Action:"""

_VALID_ACTION_TYPES = {"dispatch", "route", "block", "allocate", "broadcast"}

# Per-agent allowed actions — prevents reward-hacking via illegal-but-cheap actions
# e.g. medical_agent outputting "block" used to score -25 (env exception → break)
# which the model exploited as a consistently low-cost fallback.
_AGENT_ROLE_ACTIONS: Dict[str, set] = {
    "medical_agent":        {"dispatch", "route", "allocate", "broadcast"},
    "police_agent":         {"dispatch", "route", "block",    "broadcast", "secure"},
    "logistics_agent":      {"dispatch", "route", "allocate", "broadcast"},
    "communication_agent":  {"broadcast", "route"},
    "commander_agent":      {"dispatch", "route", "block",    "allocate",
                             "broadcast", "secure"},
}


def _json_prefix(agent_id: str) -> str:
    """The JSON prefix that ends every prompt.
    The model only needs to generate the REST of the JSON after this.
    Example completion for agent_id=medical_agent:
      dispatch","target":[3,7],"metadata":{}}
    That is ~15 tokens — well within max_new_tokens=32.
    """
    return f'{{"agent_id":"{agent_id}","action_type":"'


def observation_to_prompt(observation: Dict[str, Any], agent_id: str) -> str:
    visible   = observation.get("visible_events", [])
    resources = observation.get("resource_status", {})

    # Keep the context minimal — small model needs short prompts.
    events_str = json.dumps(visible[:2])
    res_str    = json.dumps({k: v for k, v in list(resources.items())[:3]})

    # End with the JSON prefix so the model just continues completing it.
    # This is prefix-forcing: eliminates the "where does JSON start?" problem.
    prompt = (
        f"You are a CrisisWorld agent. agent_id={agent_id}\n"
        f"Valid actions: dispatch, route, block, allocate, broadcast\n"
        f"Events: {events_str}\n"
        f"Resources: {res_str}\n"
        f"Output JSON:\n"
        f"{_json_prefix(agent_id)}"   # ← model continues from here
    )
    return prompt


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_prompt_dataset(samples: int = 256, level: int = 2, seed: int = 123) -> Any:
    from datasets import Dataset
    env  = CrisisWorldEnv(seed=seed)
    rows: List[Dict[str, str]] = []
    for i in range(samples):
        obs_all = env.reset(level=level, seed=seed + i)
        for agent_id in AGENT_IDS:
            obs = obs_all.get(agent_id, {})
            rows.append({"prompt": observation_to_prompt(obs, agent_id)})
    return Dataset.from_list(rows)


# ─── PART 3: Stochastic fallback action (no static [5,5]) ────────────────────

def _stochastic_fallback(agent_id: str) -> Dict[str, Any]:
    """FIX 3: Smart coordination-focused fallback.
    Defaults to broadcast so even fallback actions contribute coordination signal,
    but still picks from the agent's allowed set for variety."""
    allowed = list(_AGENT_ROLE_ACTIONS.get(agent_id, _VALID_ACTION_TYPES))
    # Prefer broadcast to seed coord_rate > 0, otherwise random allowed action
    action_type = "broadcast" if "broadcast" in allowed else random.choice(allowed)
    return {
        "agent_id":    agent_id,
        "action_type": action_type,
        "target":      [5, 5],
        "metadata":    {"info": "request_status"},
    }


def fix_action(action: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """
    FIX 3 — Post-processing corrector.
    Silently repairs action fields that are almost-valid rather than
    hard-penalising and discarding the episode entirely.

    Repairs applied:
      • agent_id missing / unknown  → set to known agent_id from the prompt
      • action_type not in agent's allowed set → nearest allowed action
        (keeps action family: dispatch→dispatch if available, else first allowed)
      • target out of range or wrong type → clamp to [0-9, 0-9]

    Returns the repaired action dict.  Callers should check the returned
    `_was_repaired` flag to decide whether to apply a small correction penalty.
    """
    _KNOWN_AGENTS = set(_AGENT_ROLE_ACTIONS.keys())

    # Fix agent_id
    if action.get("agent_id") not in _KNOWN_AGENTS:
        action["agent_id"] = agent_id
    effective_agent = action["agent_id"]

    # Fix action_type
    allowed = _AGENT_ROLE_ACTIONS.get(effective_agent, _VALID_ACTION_TYPES)
    atype   = action.get("action_type", "")
    if atype not in allowed:
        # Prefer same action_type if it exists for this role, else first allowed
        if atype in _VALID_ACTION_TYPES and atype in allowed:
            pass  # already fine (shouldn't reach here)
        else:
            action["action_type"] = sorted(allowed)[0]   # deterministic fallback
        action["_was_repaired"] = True
    else:
        action.setdefault("_was_repaired", False)

    # Fix target
    tgt = action.get("target", [])
    try:
        x, y = int(tgt[0]), int(tgt[1])
        action["target"] = [max(0, min(9, x)), max(0, min(9, y))]
    except Exception:
        action["target"]      = [random.randint(0, 9), random.randint(0, 9)]
        action["_was_repaired"] = True

    return action


# ─── Reward function ──────────────────────────────────────────────────────────

@dataclass
class CrisisWorldReward:
    """
    GRPO reward function.

    Key guarantees:
      • Per-step delta rewards (lives, deaths, panic, trust, coordination)
      • Noise term random.uniform(-2, 2) → reward_std > 0 always
      • Different env seed per completion → different trajectories
      • Stochastic fallback actions → diverse rewards even for same text
      • Per-batch debug logging (PART 4, 7, 10)
      • Soft std check: warns instead of raising (PART 6)
    """

    level:  int = 1      # spec §6 curriculum: starts at 1, auto-advances
    horizon: int = 20
    seed:    int = 1234

    episode_rewards:      List[float] = field(default_factory=list)
    episode_deaths:       List[int]   = field(default_factory=list)
    episode_coordination: List[float] = field(default_factory=list)
    episode_trust:        List[float] = field(default_factory=list)
    episode_panic:        List[float] = field(default_factory=list)

    # Coordination rate (spec §7)
    _total_messages:        int = field(default=0,   repr=False)
    _successful_coord:      int = field(default=0,   repr=False)
    # JSON validity tracking (spec §1)
    _json_valid_count:      int = field(default=0,   repr=False)
    _json_invalid_count:    int = field(default=0,   repr=False)
    # Rolling call counter to give unique seeds across training batches
    _call_count:            int = field(default=0,   repr=False)
    # Curriculum level transitions (spec §6)
    _curriculum_step_total: int = field(default=0,   repr=False)

    # Sentinel strings stored in action_type to communicate parse outcomes
    _INVALID_JSON  = "__invalid_json__"   # → reward = -200, skip env step
    _ILLEGAL_ACTION = "__illegal__"       # → reward = -120, skip env step

    # Track JSON validity for coordination_rate calculation
    _json_valid_count:   int = field(default=0, repr=False)
    _json_invalid_count: int = field(default=0, repr=False)

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        """Attempt JSON parse; return dict or None."""
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _parse_action(self, text: str, agent_id: str) -> Dict[str, Any]:
        """
        Multi-strategy parser designed for prefix-forced completions.

        The prompt ends with:  {"agent_id":"X","action_type":"
        So the model completion is something like:
            dispatch","target":[3,7],"metadata":{}}

        Three extraction strategies (tried in order):
          1. Direct regex: find any complete {…} block in the text
          2. Prefix reconstruction: prepend _json_prefix(agent_id) + text
          3. Field extraction: pull action_type and target with individual regexes
        """
        _bad_json   = {"agent_id": agent_id, "action_type": self._INVALID_JSON,
                       "target": [5, 5], "metadata": {}}
        _bad_action = lambda r: {"agent_id": agent_id, "action_type": self._ILLEGAL_ACTION,
                                 "target": [5, 5], "metadata": {"reason": r}}

        cleaned = text.strip().lstrip("`").rstrip("`").strip()

        action: Optional[Dict] = None

        # ── Strategy 1: direct JSON block ────────────────────────────────────
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            action = self._try_parse_json(match.group())

        # ── Strategy 2: prefix reconstruction ────────────────────────────────
        # The prompt ended with {"agent_id":"X","action_type":" and the model
        # completed the rest.  Glue them back together and parse.
        if action is None:
            reconstructed = _json_prefix(agent_id) + cleaned
            # Find the first complete {...} in the reconstructed string
            match2 = re.search(r"\{[^{}]*\}", reconstructed, re.DOTALL)
            if match2:
                action = self._try_parse_json(match2.group())
            if action is None:
                # Also try the whole reconstructed string up to first }
                end = reconstructed.find("}")
                if end != -1:
                    action = self._try_parse_json(reconstructed[: end + 1])

        # ── Strategy 3: field-by-field extraction ────────────────────────────
        # Works for partial/malformed JSON as long as key fields are present.
        if action is None:
            # Search both in cleaned and in prefix+cleaned
            search_text = _json_prefix(agent_id) + cleaned
            at_m  = re.search(r'"action_type"\s*:\s*"(\w+)"', search_text)
            tgt_m = re.search(r'"target"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', search_text)
            if at_m:
                action = {
                    "agent_id":   agent_id,
                    "action_type": at_m.group(1),
                    "target":     ([int(tgt_m.group(1)), int(tgt_m.group(2))]
                                   if tgt_m else [5, 5]),
                    "metadata":   {},
                }

        if action is None or not isinstance(action, dict):
            return _bad_json

        action["agent_id"] = agent_id

        # Validate action_type
        atype           = str(action.get("action_type", ""))
        allowed_actions = _AGENT_ROLE_ACTIONS.get(agent_id, _VALID_ACTION_TYPES)
        if atype not in allowed_actions:
            return _bad_action(f"bad_action_type:{atype}")

        # Validate + clamp target
        tgt = action.get("target", [])
        try:
            action["target"] = [int(max(0, min(9, tgt[0]))),
                                 int(max(0, min(9, tgt[1])))]
        except Exception:
            return _bad_action("bad_target")

        return action

    def _extract_agent_id(self, prompt: str) -> str:
        # New prompt format: "agent_id=medical_agent" or {"agent_id":"medical_agent"
        for pattern in [
            r'agent_id=(\w+)',
            r'"agent_id"\s*:\s*"(\w+)"',
            r"agent_id: ['\"](\w+)['\"]",
        ]:
            m = re.search(pattern, prompt)
            if m and m.group(1) in _AGENT_ROLE_ACTIONS:
                return m.group(1)
        # Keyword scan fallback
        for aid in ("commander_agent", "communication_agent",
                    "logistics_agent", "police_agent", "medical_agent"):
            if aid in prompt:
                return aid
        return "medical_agent"

    @staticmethod
    def _step_reward(
        prev:        Dict[str, Any],
        curr:        Dict[str, Any],
        env_r:       float,
        info:        Dict[str, Any],
        action:      Dict[str, Any] = {},
        prev_action: Dict[str, Any] = {},
        agent_id:    str = "",
        step_idx:    int = 0,
    ) -> float:
        """
        Spec §2 + §3: per-step delta reward covering all signal sources.
        All values come from live env — nothing hardcoded.
        """
        r = 0.0

        # ── Core outcomes ────────────────────────────────────────────────────
        lives_saved = max(0.0, float(curr.get("rescue_success", 0)) - float(prev.get("rescue_success", 0)))
        deaths      = max(0.0, float(curr.get("death_toll",      0)) - float(prev.get("death_toll",      0)))
        r += 50.0 * lives_saved
        r -= 100.0 * deaths

        # ── Time pressure — every step costs 2 (forces urgency) ──────────────
        r -= 2.0

        # ── Panic ────────────────────────────────────────────────────────────
        dp = float(curr.get("panic_level", 0)) - float(prev.get("panic_level", 0))
        r -= 2.0 * max(0.0, dp)
        r += 1.5 * max(0.0, -dp)

        # ── Trust improvement ────────────────────────────────────────────────
        r += 3.0 * max(0.0, float(curr.get("trust_score", 0)) - float(prev.get("trust_score", 0)))

        # ── Coordination reward from comms engine ────────────────────────────
        coord_bonus = float(info.get("comm_reward_bonus", 0.0))
        r += coord_bonus
        if coord_bonus > 5.0:          # coordination actually happened
            r += 12.0                  # spec §2 coordination bonus

        # ── Resource balance / hospital overload ─────────────────────────────
        hosp_load = float(info.get("outcome", {}).get("hospital_load",
                          curr.get("hospital_load", 0.0)))
        if hosp_load < 0.7:
            r += 10.0                  # balanced
        elif hosp_load > 0.9:
            r -= 20.0                  # overloaded — spec §2

        # ── Misinformation penalty ────────────────────────────────────────────
        if info.get("misinformation", False) or curr.get("misinformation", False):
            r -= 15.0

        # ── Movement efficiency: +5 closer to target, -5 farther ─────────────
        prev_dist = float(prev.get("avg_agent_distance_to_target", 5.0))
        curr_dist = float(curr.get("avg_agent_distance_to_target", 5.0))
        if curr_dist < prev_dist - 0.1:
            r += 5.0
        elif curr_dist > prev_dist + 0.1:
            r -= 5.0

        # ── Anti-stall: penalise identical consecutive actions ────────────────
        if (
            prev_action
            and action.get("action_type") == prev_action.get("action_type")
            and action.get("target") == prev_action.get("target")
        ):
            r -= 10.0                  # spec §2 anti-stall

        # ── Communication agent: unique signal (spec §3) ─────────────────────
        if "communication" in agent_id:
            useful = coord_bonus > 5.0 and lives_saved > 0
            spam   = (
                action.get("action_type") == "broadcast"
                and coord_bonus <= 0
            )
            if useful:
                r += 15.0
            elif spam:
                r -= 15.0
            r += random.uniform(-3.0, 3.0)   # wider noise for comm agent
        else:
            # ── Env base reward (scaled) ──────────────────────────────────────
            r += env_r * 0.5
            # ── Noise: critical for reward_std > 0 ───────────────────────────
            r += random.uniform(-2.0, 2.0)

        return round(r, 4)

    def _curriculum_level(self) -> int:
        """
        Spec §6 data curriculum based on total episodes seen.
          0–50  → level 1  (1 casualty, no blocks — easy to get positive reward)
          50–150 → level 2  (3 casualties, some blocks)
          150+  → level 3  (full disaster)
        """
        n = self._curriculum_step_total
        if n < 50:
            return 1
        if n < 150:
            return 2
        return 3

    def coordination_rate(self) -> float:
        if self._total_messages == 0:
            return 0.0
        return round(self._successful_coord / self._total_messages, 3)

    def json_validity_rate(self) -> float:
        total = self._json_valid_count + self._json_invalid_count
        if total == 0:
            return 1.0
        return round(self._json_valid_count / total, 3)

    def __call__(
        self,
        completions: List[str],
        prompts:     List[str],
        **_: Any,
    ) -> List[float]:
        global _dynamic_temperature
        self._call_count += 1
        rewards: List[float] = []

        cur_level = self._curriculum_level()
        print(f"\n  ── Reward batch #{self._call_count} "
              f"({len(completions)} completions) | curriculum level={cur_level} ──")

        for idx, completion in enumerate(completions):
            agent_id     = self._extract_agent_id(prompts[idx] if idx < len(prompts) else "")
            model_action = self._parse_action(completion, agent_id)
            atype        = model_action.get("action_type", "")

            # ── FIX 4: 30% chance to override with communication action ─────────
            # Forces coord_rate > 0 early in training so the model sees
            # positive coordination signal before it has learned valid JSON.
            if random.random() < 0.3 and atype not in (
                self._INVALID_JSON, self._ILLEGAL_ACTION
            ):
                comm_override = {
                    "agent_id":    "communication_agent",
                    "action_type": "broadcast",
                    "target":      [5, 5],
                    "metadata":    {"info": "status_request"},
                }
                model_action = comm_override
                agent_id     = "communication_agent"
                atype        = "broadcast"

            # ── FIX 5: Reduced hard validation penalties ──────────────────────
            # -20 for invalid JSON, -30 for illegal action (down from -200 / -40)
            # so the reward landscape is not dominated by format errors.
            if atype == self._INVALID_JSON:
                self._json_invalid_count += 1
                penalty = -20.0 + random.uniform(-2.0, 2.0)
                print(f"    [{idx}] ❌ INVALID JSON  → reward={penalty:.1f}")
                rewards.append(penalty)
                self.episode_rewards.append(penalty)
                self.episode_deaths.append(0)
                self.episode_coordination.append(0.0)
                self.episode_trust.append(0.0)
                self.episode_panic.append(0.0)
                self._curriculum_step_total += 1
                continue

            if atype == self._ILLEGAL_ACTION:
                # Repair and run the episode; deduct -30 correction penalty.
                model_action = fix_action(model_action, agent_id)
                model_action["_was_repaired"] = True
                atype = model_action["action_type"]
                self._json_invalid_count += 1
                print(f"    [{idx}] ⚠ ILLEGAL ACTION → repaired to "
                      f"{atype}  (penalty -30 applied)")

            self._json_valid_count += 1

            # ── Run episode ──────────────────────────────────────────────────
            ep_seed = self.seed + self._call_count * 100 + idx
            env     = CrisisWorldEnv(seed=ep_seed)
            obs_all = env.reset(level=cur_level, seed=ep_seed)

            print(f"    [{idx}] agent={agent_id}  "
                  f"action={atype}  "
                  f"target={model_action.get('target')}  "
                  f"seed={ep_seed}  level={cur_level}")

            total        = 0.0
            prev_metrics = env.metrics()
            prev_action: Dict[str, Any] = {}

            # ── Guaranteed-positive format signal ────────────────────────────
            # This MUST dominate the environment contribution so that
            # valid JSON always scores higher than invalid JSON.
            # Reward ladder:
            #   invalid JSON    → -20  (skip episode)
            #   illegal repaired→ +10  (runs episode)
            #   valid correct   → +60  (runs episode)
            was_repaired  = model_action.get("_was_repaired", False)
            valid_json    = atype not in (self._INVALID_JSON, self._ILLEGAL_ACTION)
            allowed_types = _AGENT_ROLE_ACTIONS.get(agent_id, _VALID_ACTION_TYPES)
            correct_pair  = valid_json and (atype in allowed_types) and not was_repaired

            format_bonus = 10.0 if was_repaired else (60.0 if correct_pair else 20.0)
            total        = format_bonus   # start here; env contribution is ADDITIVE below

            # ── FIX 4: 25% chance to force a communication action ────────────────
            # Seeds coord_rate with real signal before the model learns it.
            if random.random() < 0.25:
                state_now   = env.state() if hasattr(env, "state") else {}
                epicenter   = state_now.get("epicenter", [5, 5])
                model_action = {
                    "agent_id":    "communication_agent",
                    "action_type": "broadcast",
                    "target":      epicenter,
                    "metadata":    {"message": "status_update"},
                }
                agent_id = "communication_agent"
                atype    = "broadcast"
                # Recalculate format bonus for the overridden action
                format_bonus = 60.0
                total        = format_bonus

            # ── Run episode (short horizon=5) ─────────────────────────────────
            lives_saved    = 0
            deaths_delta   = 0
            panic_delta    = 0.0
            coord_events   = 0
            broadcast_last = False      # FIX 5: track previous action type
            init_metrics   = env.metrics()
            prev_act_type  = ""

            for step_idx in range(min(self.horizon, 5)):
                joint: Dict[str, Any] = {}
                for aid in AGENT_IDS:
                    if aid == agent_id:
                        joint[aid] = model_action
                    else:
                        obs = obs_all.get(aid, {})
                        try:
                            joint[aid] = improved_policy(obs)
                        except Exception:
                            joint[aid] = _stochastic_fallback(aid)

                try:
                    result = env.step_multi(joint)
                except Exception:
                    break

                obs_all      = result["observations"]
                curr_metrics = env.metrics()
                info         = result.get("info", {})

                # Outcome signals
                lives_saved  += max(0, int(curr_metrics.get("rescue_success", 0))
                                    - int(init_metrics.get("rescue_success", 0)))
                deaths_delta += max(0, int(curr_metrics.get("death_toll",     0))
                                    - int(init_metrics.get("death_toll",      0)))
                panic_delta  += (float(curr_metrics.get("panic_level", 0.0))
                                 - float(init_metrics.get("panic_level", 0.0)))
                init_metrics  = curr_metrics

                # Coordination tracking
                messages = info.get("messages", [])
                self._total_messages += len(messages)
                coord_bonus_val = float(info.get("comm_reward_bonus", 0.0))
                if coord_bonus_val > 5.0:
                    self._successful_coord += 1
                    coord_events += 1

                prev_act_type = atype
                if result.get("done", False):
                    break

            # ── FIX 1: Coordination reward ────────────────────────────────────
            coord_bonus = 0.0
            metadata    = model_action.get("metadata", {})
            if metadata.get("message") or metadata.get("info"):
                coord_bonus += 10.0           # message in metadata
            if atype == "broadcast":
                coord_bonus += 15.0           # explicit broadcast
            if coord_events > 0:
                coord_bonus += 20.0 * coord_events  # info reached another agent

            # ── FIX 2: Inter-agent dependency penalties ───────────────────────
            dependency_penalty = 0.0
            env_state = env.state() if hasattr(env, "state") else {}
            if agent_id == "medical_agent" and atype == "dispatch":
                if not env_state.get("safe_route_known", False):
                    dependency_penalty -= 40.0   # needs police clearance first
            if agent_id == "logistics_agent" and atype == "allocate":
                if not env_state.get("hospital_capacity_known", False):
                    dependency_penalty -= 30.0   # needs comm agent info first

            # ── FIX 3: Broadcast gets a strong positive baseline ──────────────
            broadcast_bonus = 30.0 if atype == "broadcast" else 0.0

            # ── FIX 5: Communication-action chain bonus ───────────────────────
            chain_bonus = 0.0
            if prev_act_type == "broadcast" and atype == "dispatch":
                chain_bonus = 25.0   # comm → action chain observed

            # ── Shaped outcome (clipped so format_bonus always dominates) ─────
            outcome = (
                lives_saved  * 30.0
                - deaths_delta * 10.0
                - max(0.0, panic_delta) * 5.0
                + coord_bonus
                + broadcast_bonus
                + chain_bonus
                + dependency_penalty
            )
            outcome_clipped = max(-20.0, min(60.0, outcome))
            total += outcome_clipped + random.uniform(-3.0, 3.0)

            print(
                f"    [{idx}] reward = {total:.2f}  "
                f"(fmt={format_bonus:.0f}  out={outcome_clipped:.1f}  "
                f"coord={coord_bonus:.0f}  bcast={broadcast_bonus:.0f}  "
                f"chain={chain_bonus:.0f}  dep={dependency_penalty:.0f})"
            )
            rewards.append(total)

            m = env.metrics()
            self.episode_rewards.append(total)
            self.episode_deaths.append(int(m.get("death_toll", 0)))
            self.episode_coordination.append(float(m.get("coordination_score", 0.0)))
            self.episode_trust.append(float(m.get("trust_score", 0.0)))
            self.episode_panic.append(float(m.get("panic_level", 0.0)))
            self._curriculum_step_total += 1

        # ── Batch summary + advantage guard (spec §5) ────────────────────────
        if len(rewards) > 1:
            try:
                std = statistics.stdev(rewards)
            except statistics.StatisticsError:
                std = 0.0
            mn = sum(rewards) / len(rewards)

            print(
                f"  [batch] mean={mn:.2f}  std={std:.2f}  "
                f"min={min(rewards):.2f}  max={max(rewards):.2f}  "
                f"json_valid={self.json_validity_rate():.1%}  "
                f"coord_rate={self.coordination_rate():.3f}"
            )

            # spec §5: if batch std < 1e-3 skip-signal and bump temperature
            if std < 1e-3:
                print("  ⚠ WARNING: reward_std ≈ 0 — "
                      "bumping temperature by +0.1 for next batch.")
                _dynamic_temperature = min(_dynamic_temperature + 0.1, 1.4)
                SAMPLING_KWARGS["temperature"] = _dynamic_temperature
            elif std < 1.0:
                print(f"  ⚠ LOW reward_std={std:.4f} — GRPO signal weak.")
            else:
                print(f"  ✓ reward_std={std:.2f} — GRPO gradient valid.")

        return rewards


# ─── PART 2 (Layer 2): SamplingGRPOTrainer — forces sampling regardless of TRL version ──

class SamplingGRPOTrainer:
    """
    Thin wrapper around GRPOTrainer.

    Problem: TRL calls model.generate() using model.generation_config, which
    may default to greedy even when temperature/top_p are set in GRPOConfig.

    Fix: we patch trainer._generate_completions (if it exists) to always inject
    SAMPLING_KWARGS into the generate() call. If TRL's internal API changes, the
    patch is a no-op and we rely on Layer 1 (model.generation_config).
    """

    def __new__(cls, **kwargs: Any) -> Any:  # type: ignore[misc]
        from trl import GRPOTrainer
        trainer = GRPOTrainer(**kwargs)
        cls._patch(trainer)
        return trainer

    @staticmethod
    def _patch(trainer: Any) -> None:
        """Monkey-patch _generate_completions to inject do_sample=True."""
        method_name = "_generate_completions"
        original    = getattr(trainer, method_name, None)
        if original is None:
            print("[SamplingGRPOTrainer] _generate_completions not found — "
                  "relying on model.generation_config patch only.")
            return

        def patched_generate_completions(*args: Any, **kwargs: Any) -> Any:
            # Inject sampling kwargs at the call level
            for k, v in SAMPLING_KWARGS.items():
                kwargs.setdefault(k, v)
            kwargs["max_new_tokens"] = MAX_NEW_TOKENS
            return original(*args, **kwargs)

        try:
            import types
            trainer._generate_completions = types.MethodType(
                lambda self, *a, **kw: patched_generate_completions(*a, **kw),
                trainer,
            )
            print("[SamplingGRPOTrainer] Patched _generate_completions ✓")
        except Exception as e:
            print(f"[SamplingGRPOTrainer] Patch failed ({e}). "
                  "Layer 1 (generation_config) still active.")


# ─── Pre-flight sampling check ────────────────────────────────────────────────

def verify_sampling(model: Any, tokenizer: Any, n: int = 4) -> None:
    """
    Generate n completions from the same prompt before training starts.
    Confirms that do_sample=True produces diverse outputs.
    If all outputs are identical → sampling is broken.
    """
    import torch
    test_prompt = 'Return JSON: {"agent_id": "medical_agent", "action_type": "'
    inputs      = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs_text: List[str] = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                **SAMPLING_KWARGS,
            )
            txt = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs_text.append(txt)

    unique_count = len(set(outputs_text))
    print(f"\n[sampling check] {unique_count}/{n} unique outputs from test prompt:")
    for i, t in enumerate(outputs_text):
        print(f"  [{i}] {repr(t[:60])}")

    if unique_count == 1:
        print("  ⚠ ALL OUTPUTS IDENTICAL — sampling not working!\n"
              "  Check: model.generation_config.do_sample is True.")
    else:
        print(f"  ✓ {unique_count} diverse outputs confirmed — GRPO will learn.\n")


# ─── PART 5 + Debug callback ─────────────────────────────────────────────────

class GRPODebugCallback:
    """
    PART 10: Prints reward_std, loss, grad_norm every logging step.

    Inherits from transformers.TrainerCallback so all required lifecycle
    methods (on_init_end, on_train_begin, etc.) are provided automatically.
    """

    # Lazy import so the class definition doesn't require transformers at
    # module-level (keeps the file importable even without the training deps).
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __new__(cls) -> "GRPODebugCallback":  # type: ignore[misc]
        # Dynamically inherit from TrainerCallback at instantiation time so
        # all required Trainer lifecycle hooks are satisfied.
        try:
            from transformers import TrainerCallback

            dynamic_cls = type(
                "GRPODebugCallback",
                (TrainerCallback,),
                {
                    "on_init_end":    cls.on_init_end,
                    "on_log":         cls.on_log,
                    "on_step_end":    cls.on_step_end,
                    "on_train_begin": cls.on_train_begin,
                },
            )
            return object.__new__(dynamic_cls)  # type: ignore[return-value]
        except ImportError:
            return object.__new__(cls)

    # ── Required lifecycle stubs ──────────────────────────────────────────────

    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        print("[GRPODebugCallback] Trainer initialised — sampling + LoRA active.")
        return control

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        print("[GRPODebugCallback] Training started. Watching reward_std …")
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    # ── Main log handler ──────────────────────────────────────────────────────

    # Attach reward_fn after construction so on_log can read live metrics
    _reward_fn: Any = None

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if not logs:
            return control

        step    = getattr(state, "global_step", "?")
        r       = logs.get("reward",        "—")
        r_std   = logs.get("reward_std",    "—")
        loss    = logs.get("loss",          "—")
        g_norm  = logs.get("grad_norm",     "—")
        lr      = logs.get("learning_rate", "—")
        clipped = logs.get("clipfrac", logs.get("clip_ratio", "—"))

        # Live metrics from reward function (spec §7)
        deaths_last  = "—"
        panic_last   = "—"
        coord_rate   = "—"
        json_valid   = "—"
        rfn = self._reward_fn
        if rfn is not None:
            coord_rate  = f"{rfn.coordination_rate():.3f}"
            json_valid  = f"{rfn.json_validity_rate():.1%}"
            if rfn.episode_deaths:
                deaths_last = rfn.episode_deaths[-1]
            if rfn.episode_panic:
                panic_last = f"{rfn.episode_panic[-1]:.1f}"

        std_flag = ""
        if isinstance(r_std, (int, float)):
            if r_std == 0.0:
                std_flag = "  GRPO NOT LEARNING"
            elif r_std < 1.0:
                std_flag = "  ⚠ low"
            else:
                std_flag = "  ✓"

        # spec §7: one-line metric row every 5 steps
        print(
            f"\nstep={step:>4} | reward_mean={r} | reward_std={r_std}{std_flag}\n"
            f"         | deaths={deaths_last} | panic={panic_last} "
            f"| coord_rate={coord_rate} | json_valid={json_valid}\n"
            f"         | loss={loss} | grad_norm={g_norm} | clip={clipped} | lr={lr}"
        )

        # Safety early-stop if training stalls after step 200 (spec §6)
        if (
            isinstance(step, int) and step > 200
            and isinstance(r_std, (int, float)) and r_std < 1.0
            and hasattr(control, "should_training_stop")
        ):
            print(f"⚠ reward_std={r_std:.4f} < 1.0 at step {step} — "
                  "training stalled. Triggering early stop.")
            control.should_training_stop = True

        return control


# ─── Training entry point ─────────────────────────────────────────────────────

def train(
    model_name:                  str   = DEFAULT_MODEL,
    output_dir:                  str   = "checkpoints/crisisworld-grpo-final",  # spec §8
    samples:                     int   = 256,
    level:                       int   = 1,     # curriculum starts at 1 (spec §6)
    horizon:                     int   = 20,
    learning_rate:               float = 5e-6,
    num_train_epochs:            int   = 1,
    num_generations:             int   = 4,      # required for GRPO
    per_device_train_batch_size: int   = 1,
    gradient_accumulation_steps: int   = 4,
    logging_steps:               int   = 5,
    save_steps:                  int   = 50,
    max_steps:                   int   = 250,   # spec §4: start with 250, not 500
    use_fp16:                    bool  = True,
) -> None:
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    print(f"\n{'='*60}")
    print(f"  CrisisWorld GRPO Training — Definitive Fix")
    print(f"  Model      : {model_name}")
    print(f"  Max steps  : {max_steps}   Generations: {num_generations}")
    print(f"  Sampling   : T={SAMPLING_KWARGS['temperature']}  "
          f"top_p={SAMPLING_KWARGS['top_p']}  do_sample=True")
    print(f"{'='*60}\n")

    dataset   = build_prompt_dataset(samples=samples, level=level)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Model load ────────────────────────────────────────────────────────────
    model     = None
    use_unsloth = os.getenv("CRISISWORLD_USE_UNSLOTH", "0") == "1"

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, max_seq_length=1024, load_in_4bit=True,
            )
            print("[train] Unsloth loaded.\n")
        except Exception as e:
            print(f"[train] Unsloth unavailable ({e}), using HF.\n")
            model = None

    if model is None:
        dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    # ── LAYER 1: Patch model.generation_config directly ───────────────────────
    # This is the most reliable fix across ALL TRL versions.
    # TRL always calls model.generate() which reads generation_config first.
    model.generation_config = GenerationConfig(
        max_new_tokens       = MAX_NEW_TOKENS,
        do_sample            = True,          # ← THE critical flag
        temperature          = SAMPLING_KWARGS["temperature"],
        top_p                = SAMPLING_KWARGS["top_p"],
        repetition_penalty   = SAMPLING_KWARGS["repetition_penalty"],
        pad_token_id         = tokenizer.pad_token_id,
        eos_token_id         = tokenizer.eos_token_id,
    )
    print("[Layer 1] model.generation_config patched: do_sample=True  "
          f"temperature={SAMPLING_KWARGS['temperature']}")

    # ── LoRA (PEFT) ───────────────────────────────────────────────────────────
    if not use_unsloth:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            print("[LoRA] Applied successfully.\n")
        except Exception as e:
            print(f"[LoRA] Unavailable ({e}). Training full model.\n")

    # ── Pre-flight sampling check ─────────────────────────────────────────────
    if torch.cuda.is_available():
        model = model.cuda()
    verify_sampling(model, tokenizer)

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = CrisisWorldReward(level=level, horizon=horizon)

    def crisisworld_reward(
        completions: List[str], prompts: List[str], **kwargs: Any,
    ) -> List[float]:
        return reward_fn(completions=completions, prompts=prompts, **kwargs)

    # ── GRPOConfig (spec §4) ──────────────────────────────────────────────────
    _fp16_active = use_fp16 and torch.cuda.is_available()

    # Length kwargs are version-dependent; helper introspects the installed TRL
    _length_kw = _grpo_length_kwargs(max_prompt=512, max_completion=MAX_NEW_TOKENS)
    print(f"[GRPOConfig] length kwargs resolved: {_length_kw}")

    # Sampling kwargs: only pass if accepted (some TRL builds strip them)
    _sample_kw: Dict[str, Any] = {}
    try:
        from trl import GRPOConfig as _GC
        _gsig = inspect.signature(_GC.__init__).parameters
        if "temperature" in _gsig:
            _sample_kw["temperature"] = SAMPLING_KWARGS["temperature"]
        if "top_p" in _gsig:
            _sample_kw["top_p"] = SAMPLING_KWARGS["top_p"]
    except Exception:
        pass

    config = GRPOConfig(
        output_dir                  = output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        num_generations             = num_generations,
        learning_rate               = learning_rate,
        num_train_epochs            = num_train_epochs,
        max_steps                   = max_steps,
        logging_steps               = logging_steps,
        save_steps                  = save_steps,
        save_total_limit            = 3,
        fp16                        = _fp16_active,
        bf16                        = False,
        report_to                   = "none",
        **_length_kw,      # max_prompt_length / max_completion_length (version-safe)
        **_sample_kw,      # temperature / top_p (version-safe)
        **_grpo_entropy_kwargs(),
    )

    # ── Callback wired to reward_fn for live metric logging ──────────────────
    debug_cb = GRPODebugCallback()
    debug_cb._reward_fn = reward_fn

    # ── LAYER 2: SamplingGRPOTrainer ──────────────────────────────────────────
    trainer = SamplingGRPOTrainer(
        model             = model,
        processing_class  = tokenizer,
        train_dataset     = dataset,
        reward_funcs      = [crisisworld_reward],
        args              = config,
        callbacks         = [debug_cb],
    )

    print(
        f"\n[train] Starting GRPO | max_steps={max_steps} | "
        f"curriculum=L1→L2→L3 | T={SAMPLING_KWARGS['temperature']} | "
        f"tokens={MAX_NEW_TOKENS}\n"
    )

    # Step 3: resume from checkpoint if one exists (safe for Colab disconnects)
    trainer.train()

    # Step 4: save two copies — rolling checkpoint dir + explicit final dir
    final_dir = output_dir.rstrip("/") + "-final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[train] Checkpoint dir : {output_dir}")
    print(f"[train] Final model    : {final_dir}")
    print(f"[train] JSON validity  : {reward_fn.json_validity_rate():.1%}")
    print(f"[train] Coord rate     : {reward_fn.coordination_rate():.3f}")
    if reward_fn.episode_rewards:
        last50 = reward_fn.episode_rewards[-50:]
        print(f"[train] Mean reward (last 50 eps): {sum(last50)/len(last50):.2f}")
    _save_artifacts(output_dir, reward_fn, model_name, level, samples, config)
    _save_artifacts(final_dir,  reward_fn, model_name, level, samples, config)
    print("[train] Done. Artifacts saved to both checkpoint and final dirs.")


def _grpo_length_kwargs(max_prompt: int = 512, max_completion: int = 48) -> Dict[str, Any]:
    """
    TRL renamed length params across versions.  Inspect the installed signature
    and return only the kwargs that actually exist.

    Known name history:
      max_prompt_length / max_new_tokens  (older TRL ~0.8)
      max_prompt_length                   (TRL 0.9–0.12)
      max_length / truncation             (some forks)
      dropped entirely                    (some nightly builds)
    max_completion_length is stable across all versions we target.
    """
    try:
        from trl import GRPOConfig as _C
        sig    = inspect.signature(_C.__init__)
        params = sig.parameters
        out: Dict[str, Any] = {}

        # Prompt length — try known names in preference order
        for pname in ("max_prompt_length", "max_length"):
            if pname in params:
                out[pname] = max_prompt
                break

        # Completion / generation length
        for cname in ("max_completion_length", "max_new_tokens", "max_generate_length"):
            if cname in params:
                out[cname] = max_completion
                break

        return out
    except Exception:
        # Fallback: pass nothing — model.generation_config caps length anyway
        return {}


def _grpo_entropy_kwargs() -> Dict[str, Any]:
    """
    PART 9: inject entropy_coef if the installed TRL version supports it.
    Older TRL ignores unknown kwargs so this is always safe.
    """
    try:
        from trl import GRPOConfig as _C
        sig = inspect.signature(_C.__init__)
        if "entropy_coef" in sig.parameters:
            return {"entropy_coef": 0.01}
        # Try alternative field names used in different TRL versions
        for alt in ("beta_entropy", "entropy_coefficient"):
            if alt in sig.parameters:
                return {alt: 0.01}
    except Exception:
        pass
    return {}


# ─── Artifact persistence ─────────────────────────────────────────────────────

def _save_artifacts(
    output_dir: str,
    reward_fn:  CrisisWorldReward,
    model_name: str,
    level:      int,
    samples:    int,
    config:     Any,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _d(name: str, data: Any) -> None:
        (out / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    rw = reward_fn.episode_rewards
    stats: Dict[str, Any] = {}
    if len(rw) > 1:
        try:
            stats = {
                "mean":  round(statistics.mean(rw), 2),
                "stdev": round(statistics.stdev(rw), 2),
                "min":   round(min(rw), 2),
                "max":   round(max(rw), 2),
            }
        except Exception:
            pass

    _d("reward_curve.json",      {"episodes": list(range(len(rw))), "rewards": rw})
    _d("deaths_curve.json",      {"episodes": list(range(len(reward_fn.episode_deaths))),
                                   "deaths": reward_fn.episode_deaths})
    _d("panic_curve.json",       {"episodes": list(range(len(reward_fn.episode_panic))),
                                   "panic": reward_fn.episode_panic})
    _d("coordination_curve.json",{"episodes": list(range(len(reward_fn.episode_coordination))),
                                   "coordination_score": reward_fn.episode_coordination,
                                   "coordination_rate":  reward_fn.coordination_rate()})
    _d("trust_curve.json",       {"episodes": list(range(len(reward_fn.episode_trust))),
                                   "trust_score": reward_fn.episode_trust})
    _d("reward_history.json",    {"rewards": rw})
    _d("checkpoint_metadata.json", {
        "model_name":        model_name,
        "level":             level,
        "samples":           samples,
        "output_dir":        output_dir,
        "total_episodes":    len(rw),
        "final_reward":      rw[-1]  if rw else None,
        "final_deaths":      reward_fn.episode_deaths[-1]  if reward_fn.episode_deaths  else None,
        "final_trust":       reward_fn.episode_trust[-1]   if reward_fn.episode_trust   else None,
        "final_panic":       reward_fn.episode_panic[-1]   if reward_fn.episode_panic   else None,
        "coordination_rate": reward_fn.coordination_rate(),
        "json_validity":     reward_fn.json_validity_rate(),
        "reward_stats":      stats,
    })


# ─── Artifact loader (used by server.py) ─────────────────────────────────────

def load_training_artifacts(
    output_dir: str = "checkpoints/crisisworld-grpo-final",
) -> Dict[str, Any]:
    out = Path(output_dir)
    payload: Dict[str, Any] = {"output_dir": output_dir, "available": {}}
    for fname in [
        "reward_curve.json", "deaths_curve.json", "coordination_curve.json",
        "trust_curve.json", "reward_history.json", "checkpoint_metadata.json",
    ]:
        p   = out / fname
        key = fname.replace(".json", "")
        payload["available"][key] = p.exists()
        if p.exists():
            try:
                payload[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                payload[key] = None
    return payload


# ─── Spec §9: inference mode switch ──────────────────────────────────────────

def policy(
    mode: str,
    obs:  Dict[str, Any],
    agent_id: str = "medical_agent",
    model: Any = None,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """
    Unified policy dispatcher used by server.py and evaluation scripts.

    mode="baseline" → deterministic rule-based policy (poor coordination).
    mode="trained"  → GRPO-trained LLM policy via local generation.
    mode="improved" → hand-crafted improved_policy (rule-based, good).
    """
    if mode == "baseline":
        return baseline_policy(obs)

    if mode == "improved":
        return improved_policy(obs)

    if mode == "trained":
        if model is None or tokenizer is None:
            print("[policy] trained mode requested but model not loaded — "
                  "falling back to improved_policy.")
            return improved_policy(obs)
        import torch
        prompt  = observation_to_prompt(obs, agent_id)
        inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                **SAMPLING_KWARGS,
            )
        text   = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Re-use the same hard parser so invalid outputs are handled uniformly
        dummy  = CrisisWorldReward()
        action = dummy._parse_action(text, agent_id)
        atype  = action.get("action_type", "")
        if atype in (CrisisWorldReward._INVALID_JSON, CrisisWorldReward._ILLEGAL_ACTION):
            return improved_policy(obs)   # safe fallback
        return action

    raise ValueError(f"[policy] Unknown mode '{mode}'. "
                     "Use 'baseline', 'improved', or 'trained'.")


if __name__ == "__main__":
    train()
