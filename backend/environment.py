from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import engine
import reward as _reward
from engine import (
    SCENARIO_PROFILES,
    advance_world_dynamics,
    apply_time_effects,
    build_causal_chain,
    compute_uncertainty,
    generate_dynamic_events,
    hospital_overload_ratio,
    step_towards as _step_towards_fn,
)
from comms import (
    evaluate_comm_outcomes,
    generate_advisory_responses,
    generate_step_comms,
)
from models import ACTION_TYPES, AGENT_IDS, ActionModel, ObservationModel, StepLogModel
from reward import RewardWeights, compute_components, reward_change_reason, update_agent_trust, weighted_total


class CrisisWorldEnv:
    # Scenario profiles delegated to engine module (single source of truth)
    SCENARIO_PROFILES = SCENARIO_PROFILES

    ROLE_ACTIONS = {
        "medical_agent": {"dispatch", "route", "allocate", "broadcast"},
        # police can now 'secure' a zone before medical dispatch
        "police_agent": {"dispatch", "route", "block", "broadcast", "secure"},
        "logistics_agent": {"dispatch", "route", "allocate", "broadcast"},
        "communication_agent": {"broadcast", "route"},
        # commander has full action set including secure
        "commander_agent": set(ACTION_TYPES),
    }

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        panic_threshold: float = 100.0,
        seed: int = 42,
        reward_weights: Optional[RewardWeights] = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.panic_threshold = panic_threshold
        self.rng = random.Random(seed)
        self.reward_weights = reward_weights or RewardWeights()
        self.level = 1
        self.reset(level=1, seed=seed)

    # OpenEnv-style API
    def reset(
        self, level: int = 1, seed: Optional[int] = None, scenario: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        if level not in (1, 2, 3):
            raise ValueError("level must be 1, 2, or 3")
        if seed is not None:
            self.rng.seed(seed)
        self.level = level
        self.step_count = 0
        self.total_reward = 0.0
        self.dead_count = 0
        self.saved_count = 0
        self.panic_level = 5.0 * level
        self.trust_score = 80.0
        self.no_progress_steps = 0
        self.trajectory: List[Dict[str, Any]] = []
        self.scenario = scenario or {1: "earthquake", 2: "flood", 3: "blackout"}[level]
        if self.scenario not in self.SCENARIO_PROFILES:
            raise ValueError("scenario must be one of earthquake, flood, blackout")
        self.scenario_profile = copy.deepcopy(self.SCENARIO_PROFILES[self.scenario])
        self.message_ttl = 6
        self.active_effects: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.system_state = {
            "infrastructure_health": 100.0,
            "communication_integrity": 100.0,
            "mobility_index": 1.0,
        }
        self.fuel_level = 100.0
        self.district_panic: Dict[Tuple[int, int], float] = {}
        self.response_time_samples: List[float] = []
        self.coordination_hits = 0
        self.message_count = 0
        self.misinformation_impact = 0.0
        self.available_ambulances = 1 + level
        self.ambulance_cooldowns: List[int] = []
        self.simultaneous_resolution_buffer: List[Dict[str, Any]] = []
        self.prev_step_snapshot = {
            "panic_level": self.panic_level,
            "trust_score": self.trust_score,
            "infrastructure_health": self.system_state["infrastructure_health"],
            "deaths": self.dead_count,
        }

        self.hospitals = [
            {"id": "h1", "pos": (1, 1), "capacity": 4 + level, "occupancy": 0},
            {"id": "h2", "pos": (8, 2), "capacity": 3 + level, "occupancy": 0},
        ]
        self.shelters = [
            {"id": "s1", "pos": (2, 8), "capacity": 6 + level, "occupancy": 0},
            {"id": "s2", "pos": (7, 7), "capacity": 6 + level, "occupancy": 0},
        ]

        self.blocked_roads = set()
        if level >= 2:
            self.blocked_roads.update({(4, 4), (4, 5), (5, 5)})
        if level == 3:
            self.blocked_roads.update({(3, 6), (6, 3), (6, 6), (2, 5), (5, 2)})

        casualty_count = {1: 1, 2: 3, 3: 6}[level]
        self.casualties = []
        for idx in range(casualty_count):
            pos = (self.rng.randint(0, 9), self.rng.randint(0, 9))
            while pos in self.blocked_roads:
                pos = (self.rng.randint(0, 9), self.rng.randint(0, 9))
            self.casualties.append(
                {
                    "id": f"c{idx}",
                    "pos": pos,
                    "severity": self.rng.randint(1, 3),
                    "status": "waiting",
                    "time_waiting": 0,
                }
            )

        self.events = [
            {"type": self.scenario, "location": (5, 5), "intensity": level + 1},
            {"type": "casualties", "count": casualty_count},
        ]

        self.agents = {
            aid: {
                "id": aid,
                "pos": self._spawn_point(aid),
                "last_action_key": None,
                "repeated_count": 0,
                "assigned_task": None,
            }
            for aid in AGENT_IDS
        }
        self.messages: List[Dict[str, Any]] = []
        self.message_counter = 0
        # Coordination memory for sequence rewards (police-secure -> medical-treat).
        self.zone_security: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.medical_entry_log: Dict[Tuple[int, int], int] = {}

        # Per-agent trust scores (0–100). Used for conflict resolution and
        # as an observable signal. Initialised at 50 (neutral).
        self.agent_trust: Dict[str, float] = {aid: 50.0 for aid in AGENT_IDS}

        # Broadcast-per-step counter for spam detection (reset each step).
        self._broadcast_this_step: Dict[str, int] = {aid: 0 for aid in AGENT_IDS}

        # Per-step coordination hit log for metrics streaming
        self.coordination_hits_per_step: List[int] = []
        self.trust_per_step: List[float] = []

        # ── Selective communication tracking (comms.py) ───────────────────────
        # last step each agent sent a comm message (anti-spam cooldown)
        self._comm_last_sent: Dict[str, int] = {aid: -99 for aid in AGENT_IDS}
        # messages sent by each agent in the current step (reset each step)
        self._comm_sent_count: Dict[str, int] = {aid: 0 for aid in AGENT_IDS}
        # all comm_links generated this episode (for metrics / training)
        self.comm_links: List[Dict[str, Any]] = []
        # last step's targeted comm messages (streamed to frontend via WS)
        self._last_comm_links: List[Dict[str, Any]] = []

        return self._all_observations()

    def state(self) -> Dict[str, Any]:
        return copy.deepcopy(
            {
                "level": self.level,
                "grid_size": self.grid_size,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "hospitals": self.hospitals,
                "shelters": self.shelters,
                "blocked_roads": list(self.blocked_roads),
                "casualties": self.casualties,
                "events": self.events,
                "agents": self.agents,
                "messages": self.messages,
                "scenario": self.scenario,
                "active_effects": self.active_effects,
                "system_state": copy.deepcopy(self.system_state),
                "metrics": self.metrics(),
                "comm_links": list(getattr(self, "_last_comm_links", [])),
            }
        )

    def step(self, action: Dict[str, Any] | ActionModel) -> Dict[str, Any]:
        self.prev_step_snapshot = {
            "panic_level": self.panic_level,
            "trust_score": self.trust_score,
            "infrastructure_health": self.system_state["infrastructure_health"],
            "deaths": self.dead_count,
        }
        action_model = self._validate_action(action)
        actor = self.agents[action_model.agent_id]
        action_key = (
            action_model.action_type,
            action_model.target,
            json.dumps(action_model.metadata, sort_keys=True),
        )

        repeated_action = action_key == actor["last_action_key"]
        if repeated_action:
            actor["repeated_count"] += 1
        else:
            actor["repeated_count"] = 0
        actor["last_action_key"] = action_key

        outcome = self._apply_action(action_model)
        generated_events = self.generate_dynamic_events()
        time_effects = self.apply_global_time_effects()
        outcome["panic_delta"] += time_effects["panic_delta"]
        outcome["trust_delta"] += time_effects["trust_delta"]
        outcome["deaths"] += time_effects["deaths"]
        outcome["system_effects"] = time_effects["effects"]
        outcome["generated_events"] = generated_events
        dynamics = self._advance_world_dynamics()
        outcome["deaths"] += dynamics["deaths"]
        outcome["lives_saved"] += dynamics["lives_saved"]

        ignored_tasks = self._ignored_task_penalty(outcome["progress"])
        components = self._reward_components(
            lives_saved=outcome["lives_saved"],
            deaths=outcome["deaths"],
            cooperation=outcome["cooperation"],
            balanced_allocation=outcome["balanced_allocation"],
            overloaded=outcome["overloaded"],
            misinformation=outcome["misinformation"],
            repeated_action=repeated_action,
            ignored_tasks=ignored_tasks,
            cooperation_confidence=float(outcome.get("cooperation_confidence", 0.0)),
            prev_snapshot=self.prev_step_snapshot,
            triage_conflict=bool(outcome.get("triage_conflict", False)),
            strategic_tradeoff=bool(outcome.get("strategic_tradeoff", False)),
        )
        reward = self._weighted_reward(components)

        self.step_count += 1
        self.total_reward += reward
        self.saved_count += outcome["lives_saved"]
        self.dead_count += outcome["deaths"]
        self.panic_level = max(0.0, self.panic_level + outcome["panic_delta"])
        self.trust_score = min(100.0, max(0.0, self.trust_score + outcome["trust_delta"]))
        self.system_state["communication_integrity"] = min(
            100.0, max(0.0, self.system_state["communication_integrity"])
        )
        self.system_state["infrastructure_health"] = min(
            100.0, max(0.0, self.system_state["infrastructure_health"])
        )
        self.system_state["mobility_index"] = min(1.0, max(0.1, self.system_state["mobility_index"]))

        done = (
            self._all_casualties_handled()
            or self.step_count >= self.max_steps
            or self.panic_level >= self.panic_threshold
        )
        observations = self._all_observations()
        coordination_details = outcome.get("coordination_details")
        info = {
            "reward_components": components,
            "outcome": outcome,
            "metrics": self.metrics(),
            "reason": self._explain_action(action_model, outcome),
            "coordination": bool(coordination_details),
            "used_message": coordination_details["message_id"] if coordination_details else None,
            "sender_agent_id": coordination_details["sender_agent_id"] if coordination_details else None,
            "receiver_agent_id": coordination_details["receiver_agent_id"] if coordination_details else None,
            "message_id": coordination_details["message_id"] if coordination_details else None,
            "reward_breakdown": {
                "survival": components["survival_reward"],
                "time": components["time_penalty"],
                "coordination": components["coordination_reward"],
                "resource": components["resource_efficiency"],
                "misinfo": components["misinformation_penalty"],
            },
            "cause_effect": "; ".join(outcome.get("system_effects", [])) if outcome.get("system_effects") else "Local action impact only.",
            "reward_change_reason": self._reward_change_reason(components),
            "dominant_system_factor": self._dominant_system_factor(components),
            "validation_feedback": {"valid": True, "violations": []},
        }

        self.trajectory.append(
            StepLogModel(
                step=self.step_count,
                observation=observations,
                action=action_model.model_dump(),
                reward=reward,
                outcome=info,
            ).model_dump()
        )

        return {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def step_multi(self, actions: Dict[str, Dict[str, Any] | ActionModel]) -> Dict[str, Any]:
        if not actions:
            raise ValueError("actions must include at least one agent action")
        self.prev_step_snapshot = {
            "panic_level": self.panic_level,
            "trust_score": self.trust_score,
            "infrastructure_health": self.system_state["infrastructure_health"],
            "deaths": self.dead_count,
        }

        action_models: Dict[str, ActionModel] = {}
        repeated_flags: Dict[str, bool] = {}
        per_agent_outcomes: Dict[str, Dict[str, Any]] = {}
        for agent_id in sorted(actions):
            action_model = self._validate_action(actions[agent_id])
            if action_model.agent_id != agent_id:
                raise ValueError(f"action key {agent_id} must match action.agent_id {action_model.agent_id}")
            action_models[agent_id] = action_model
            actor = self.agents[agent_id]
            action_key = (
                action_model.action_type,
                action_model.target,
                json.dumps(action_model.metadata, sort_keys=True),
            )
            repeated_flags[agent_id] = action_key == actor["last_action_key"]
            actor["last_action_key"] = action_key
            actor["repeated_count"] = actor["repeated_count"] + 1 if repeated_flags[agent_id] else 0

        for agent_id in sorted(action_models):
            outcome = self._apply_action(action_models[agent_id], detect_cooperation=False)
            per_agent_outcomes[agent_id] = outcome

        for agent_id in sorted(action_models):
            if action_models[agent_id].action_type == "broadcast":
                continue
            coordination_details = self._is_cooperation(action_models[agent_id])
            per_agent_outcomes[agent_id]["cooperation"] = bool(coordination_details)
            per_agent_outcomes[agent_id]["coordination_details"] = coordination_details
            if coordination_details:
                self.coordination_hits += 1
                per_agent_outcomes[agent_id]["cooperation_confidence"] = float(
                    coordination_details["effective_weight"]
                )

        generated_events = self.generate_dynamic_events()
        time_effects = self.apply_global_time_effects()
        aggregate = {
            "lives_saved": 0,
            "deaths": time_effects["deaths"],
            "cooperation": False,
            "balanced_allocation": False,
            "overloaded": False,
            "misinformation": False,
            "panic_delta": time_effects["panic_delta"],
            "trust_delta": time_effects["trust_delta"],
            "progress": False,
            "coordination_details": None,
            "cooperation_confidence": 0.0,
            "triage_conflict": False,
            "strategic_tradeoff": False,
            "coordinated_sequence": False,
            "unsafe_medical_dispatch": False,
            "system_effects": time_effects["effects"],
            "generated_events": generated_events,
        }
        for outcome in per_agent_outcomes.values():
            aggregate["lives_saved"] += outcome["lives_saved"]
            aggregate["deaths"] += outcome["deaths"]
            aggregate["panic_delta"] += outcome["panic_delta"]
            aggregate["trust_delta"] += outcome["trust_delta"]
            aggregate["cooperation"] = aggregate["cooperation"] or outcome["cooperation"]
            aggregate["balanced_allocation"] = aggregate["balanced_allocation"] or outcome["balanced_allocation"]
            aggregate["overloaded"] = aggregate["overloaded"] or outcome["overloaded"]
            aggregate["misinformation"] = aggregate["misinformation"] or outcome["misinformation"]
            aggregate["progress"] = aggregate["progress"] or outcome["progress"]
            aggregate["triage_conflict"] = aggregate["triage_conflict"] or outcome.get("triage_conflict", False)
            aggregate["strategic_tradeoff"] = aggregate["strategic_tradeoff"] or outcome.get(
                "strategic_tradeoff", False
            )
            aggregate["coordinated_sequence"] = aggregate["coordinated_sequence"] or outcome.get(
                "coordinated_sequence", False
            )
            aggregate["unsafe_medical_dispatch"] = aggregate["unsafe_medical_dispatch"] or outcome.get(
                "unsafe_medical_dispatch", False
            )
            if outcome.get("coordination_details") and aggregate["coordination_details"] is None:
                aggregate["coordination_details"] = outcome["coordination_details"]
            aggregate["cooperation_confidence"] = max(
                aggregate["cooperation_confidence"],
                float(outcome.get("cooperation_confidence", 0.0)),
            )

        dynamics = self._advance_world_dynamics()
        aggregate["deaths"] += dynamics["deaths"]
        aggregate["lives_saved"] += dynamics["lives_saved"]

        # -- Commander override: if commander assigns a target, override other agents
        if "commander_agent" in action_models:
            cmd = action_models["commander_agent"]
            assigned_to = cmd.metadata.get("assign_to")
            if assigned_to and assigned_to in action_models:
                cmd_target = cmd.target
                override_action = action_models[assigned_to]
                # Override only if commander's trust > assigned agent's trust
                if self.agent_trust.get("commander_agent", 50) >= self.agent_trust.get(assigned_to, 50):
                    action_models[assigned_to] = ActionModel.model_validate({
                        "agent_id": assigned_to,
                        "action_type": override_action.action_type,
                        "target": list(cmd_target),
                        "metadata": {**override_action.metadata, "overridden_by": "commander_agent"},
                    })

        # -- Communication spam detection (more than 2 broadcasts per agent per step)
        broadcast_counts: Dict[str, int] = {}
        for aid, am in action_models.items():
            if am.action_type == "broadcast":
                broadcast_counts[aid] = broadcast_counts.get(aid, 0) + 1
        comm_spam = any(v > 2 for v in broadcast_counts.values())

        # ── Selective targeted communication (comms.py) ───────────────────────
        # Reset per-step sent counters
        self._comm_sent_count = {aid: 0 for aid in AGENT_IDS}

        # Build plain-dict action map for comms module
        plain_actions: Dict[str, Any] = {
            aid: am.model_dump() for aid, am in action_models.items()
        }

        # Generate targeted messages for agents that need to communicate
        step_comm_links = generate_step_comms(
            agent_actions=plain_actions,
            observations=self._all_observations(),
            agent_last_sent=self._comm_last_sent,
            agent_sent_count=self._comm_sent_count,
            current_step=self.step_count,
        )

        # Generate advisory responses (receiver → sender)
        advisory_responses = generate_advisory_responses(
            incoming=step_comm_links,
            agent_actions=plain_actions,
            observations=self._all_observations(),
            current_step=self.step_count,
        )
        step_comm_links.extend(advisory_responses)

        # Persist into env message store (for _is_cooperation lookups)
        for cl in step_comm_links:
            env_msg = {
                "message_id": cl["message_id"],
                "sender": cl["from"],
                "recipient": cl["to"],
                "text": cl.get("text", ""),
                "target": list(plain_actions.get(cl["from"], {}).get("target", [5, 5])),
                "intent": cl.get("reason", "coordination"),
                "confidence": cl.get("confidence", 0.8),
                "truth_score": cl.get("confidence", 0.8),
                "timestamp": self.step_count,
                "time_step": self.step_count,
                "used": False,
            }
            self.messages.append(env_msg)

        # Evaluate outcomes (post-step reward deltas)
        comm_reward_deltas = evaluate_comm_outcomes(
            comm_messages=step_comm_links,
            per_agent_outcomes=per_agent_outcomes,
            agent_actions=plain_actions,
        )
        # Accumulate comm reward deltas into aggregate
        comm_reward_bonus = sum(comm_reward_deltas.values())

        self._last_comm_links = step_comm_links
        self.comm_links.extend(step_comm_links)

        ignored_tasks = self._ignored_task_penalty(aggregate["progress"])
        repeated_any = any(repeated_flags.values())
        components = self._reward_components(
            lives_saved=aggregate["lives_saved"],
            deaths=aggregate["deaths"],
            cooperation=aggregate["cooperation"],
            balanced_allocation=aggregate["balanced_allocation"],
            overloaded=aggregate["overloaded"],
            misinformation=aggregate["misinformation"],
            repeated_action=repeated_any,
            ignored_tasks=ignored_tasks,
            cooperation_confidence=float(aggregate.get("cooperation_confidence", 0.0)),
            prev_snapshot=self.prev_step_snapshot,
            triage_conflict=aggregate["triage_conflict"],
            strategic_tradeoff=aggregate["strategic_tradeoff"],
            coordinated_sequence=aggregate["coordinated_sequence"],
            unsafe_medical_dispatch=aggregate["unsafe_medical_dispatch"],
            communication_spam=comm_spam,
        )
        reward = self._weighted_reward(components) + comm_reward_bonus

        self.step_count += 1
        self.total_reward += reward
        self.saved_count += aggregate["lives_saved"]
        self.dead_count += aggregate["deaths"]
        self.panic_level = max(0.0, self.panic_level + aggregate["panic_delta"])
        self.trust_score = min(100.0, max(0.0, self.trust_score + aggregate["trust_delta"]))

        # -- Per-agent trust update
        for aid in action_models:
            ao = per_agent_outcomes.get(aid, {})
            update_agent_trust(
                self.agent_trust,
                aid,
                correct_decision=bool(ao.get("progress")),
                coordination_success=bool(ao.get("coordination_details")),
                misinformation=bool(ao.get("misinformation")),
            )

        # -- Track per-step metrics for training curves
        self.coordination_hits_per_step.append(self.coordination_hits)
        self.trust_per_step.append(self.trust_score)

        done = (
            self._all_casualties_handled()
            or self.step_count >= self.max_steps
            or self.panic_level >= self.panic_threshold
        )
        self._emit_police_route_advisories()
        observations = self._all_observations()
        coordination_details = aggregate.get("coordination_details")

        # -- Build causal chain from engine module (full multi-step context)
        causal_chain = build_causal_chain(
            aggregate,
            components,
            step_count=self.step_count,
            panic_level=self.panic_level,
            overload_ratio=self._hospital_overload_ratio(),
            waiting_casualties=sum(1 for c in self.casualties if c.get("status") == "waiting"),
        )

        # -- Per-agent correctness signals (for frontend green/red)
        per_agent_correctness: Dict[str, bool] = {
            aid: bool(per_agent_outcomes.get(aid, {}).get("progress") or
                      per_agent_outcomes.get(aid, {}).get("coordination_details") or
                      per_agent_outcomes.get(aid, {}).get("lives_saved", 0) > 0)
            for aid in action_models
        }

        info = {
            "reward_components": components,
            "outcome": aggregate,
            "metrics": self.metrics(),
            "reason": "Simultaneous multi-agent step executed.",
            "coordination": bool(coordination_details),
            "used_message": coordination_details["message_id"] if coordination_details else None,
            "sender_agent_id": coordination_details["sender_agent_id"] if coordination_details else None,
            "receiver_agent_id": coordination_details["receiver_agent_id"] if coordination_details else None,
            "message_id": coordination_details["message_id"] if coordination_details else None,
            "reward_breakdown": {
                "survival": components["survival_reward"],
                "time": components["time_penalty"],
                "coordination": components["coordination_reward"],
                "resource": components["resource_efficiency"],
                "misinfo": components["misinformation_penalty"],
            },
            "cause_effect": "; ".join(aggregate.get("system_effects", []))
            if aggregate.get("system_effects")
            else "Multi-agent local effects only.",
            "causal_chain": causal_chain,
            "reward_change_reason": self._reward_change_reason(components),
            "dominant_system_factor": self._dominant_system_factor(components),
            "per_agent_outcomes": per_agent_outcomes,
            "per_agent_correctness": per_agent_correctness,
            "agent_trust": dict(self.agent_trust),
            "comm_links": self._last_comm_links,
            "comm_reward_deltas": comm_reward_deltas,
            "validation_feedback": {"valid": True, "violations": []},
        }

        self.trajectory.append(
            StepLogModel(
                step=self.step_count,
                observation=observations,
                action={aid: model.model_dump() for aid, model in action_models.items()},
                reward=reward,
                outcome=info,
            ).model_dump()
        )
        return {"observations": observations, "reward": reward, "done": done, "info": info}

    def save_trajectory(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for step in self.trajectory:
                f.write(json.dumps(step) + "\n")

    def metrics(self) -> Dict[str, Any]:
        response_avg = sum(self.response_time_samples) / max(1, len(self.response_time_samples))
        coordination_score = (100.0 * self.coordination_hits) / max(1, self.message_count)
        infra_damage = 100.0 - self.system_state["infrastructure_health"]
        total_cas = max(1, len(self.casualties))
        treated = sum(1 for c in self.casualties if c.get("status") == "treated")
        rescue_success_rate = (treated / total_cas) * 100.0
        return {
            "deaths": float(self.dead_count),
            "panic_level": float(self.panic_level),
            "trust_score": float(self.trust_score),
            "total_reward": float(self.total_reward),
            "response_time_avg": float(response_avg),
            "coordination_score": float(coordination_score),
            "misinformation_impact": float(self.misinformation_impact),
            "infrastructure_damage_index": float(infra_damage),
            "rescue_success_rate": float(rescue_success_rate),
            "agent_trust": {k: round(v, 2) for k, v in self.agent_trust.items()},
        }

    # Internal helpers
    def _validate_action(self, action: Dict[str, Any] | ActionModel) -> ActionModel:
        model = action if isinstance(action, ActionModel) else ActionModel.model_validate(action)
        if model.action_type not in self.ROLE_ACTIONS[model.agent_id]:
            raise ValueError(f"{model.agent_id} cannot perform action_type={model.action_type}")
        # Dispatch/route into blocked cell is forbidden; secure is allowed (police clear it)
        if tuple(model.target) in self.blocked_roads and model.action_type in {"dispatch", "route"}:
            raise ValueError("cannot dispatch/route directly into a blocked road cell")
        return model

    def _apply_action(self, action: ActionModel, detect_cooperation: bool = True) -> Dict[str, Any]:
        outcome = {
            "lives_saved": 0,
            "deaths": 0,
            "cooperation": False,
            "balanced_allocation": False,
            "overloaded": False,
            "misinformation": False,
            "panic_delta": 0.0,
            "trust_delta": 0.0,
            "progress": False,
            "coordination_details": None,
            "cooperation_confidence": 0.0,
            "response_time_delta": 0.0,
            "triage_conflict": False,
            "strategic_tradeoff": False,
            "coordinated_sequence": False,
            "unsafe_medical_dispatch": False,
        }
        actor = self.agents[action.agent_id]
        old_pos = tuple(actor["pos"])

        if action.action_type in {"dispatch", "route"}:
            actor["pos"] = self._step_towards(actor["pos"], action.target)
            response_delta = (
                abs(action.target[0] - old_pos[0]) + abs(action.target[1] - old_pos[1])
            ) * (1.0 + (len(self.blocked_roads) / max(1, self.grid_size)))
            response_delta += (1.0 - self.system_state["mobility_index"]) * 4.0
            self.response_time_samples.append(float(response_delta))
            outcome["response_time_delta"] = float(response_delta)

        if action.action_type == "dispatch":
            if action.agent_id == "police_agent":
                casualty_at_target = self._casualty_at(tuple(action.target))
                if casualty_at_target and casualty_at_target["status"] == "waiting":
                    self.zone_security[tuple(action.target)] = {
                        "secured_by": "police_agent",
                        "time_step": self.step_count,
                        "casualty_id": casualty_at_target["id"],
                    }
                    outcome["progress"] = True
                    outcome["trust_delta"] += 0.3
                    outcome["panic_delta"] -= 0.6

            if action.agent_id in {"medical_agent", "commander_agent"}:
                sec = self.zone_security.get(tuple(action.target))
                is_secured = bool(
                    sec
                    and sec.get("secured_by") == "police_agent"
                    and (self.step_count - int(sec.get("time_step", -100))) <= 4
                )
                if is_secured:
                    outcome["coordinated_sequence"] = True
                    outcome["trust_delta"] += 0.5
                    outcome["panic_delta"] -= 0.5
                else:
                    outcome["unsafe_medical_dispatch"] = True
                    outcome["trust_delta"] -= 0.6
                    outcome["panic_delta"] += 0.6

            casualty = self._casualty_at(actor["pos"])
            if casualty and casualty["status"] == "waiting" and action.agent_id in {"medical_agent", "commander_agent"}:
                self.medical_entry_log[tuple(actor["pos"])] = self.step_count
                sec_at_entry = self.zone_security.get(tuple(actor["pos"]))
                is_secured_at_entry = bool(
                    sec_at_entry
                    and sec_at_entry.get("secured_by") == "police_agent"
                    and (self.step_count - int(sec_at_entry.get("time_step", -100))) <= 4
                )
                if is_secured_at_entry:
                    outcome["coordinated_sequence"] = True
                    outcome["trust_delta"] += 0.7
                    outcome["panic_delta"] -= 0.7

                if action.agent_id == "medical_agent" and self.available_ambulances <= 0:
                    outcome["triage_conflict"] = True
                    outcome["panic_delta"] += 0.8
                    outcome["trust_delta"] -= 0.5
                else:
                    casualty["status"] = "treated"
                    outcome["lives_saved"] += 1
                    outcome["panic_delta"] -= 3.0
                    outcome["trust_delta"] += 2.0
                    outcome["progress"] = True
                    if action.agent_id == "medical_agent":
                        self.available_ambulances = max(0, self.available_ambulances - 1)
                        self.ambulance_cooldowns.append(2)
                    waiting = [c for c in self.casualties if c["status"] == "waiting"]
                    if waiting:
                        top_severity = max(int(c["severity"]) for c in waiting)
                        if int(casualty["severity"]) >= top_severity:
                            outcome["strategic_tradeoff"] = True
            if action.agent_id == "commander_agent":
                target_agent = action.metadata.get("assign_to")
                if target_agent in self.agents:
                    self.agents[target_agent]["assigned_task"] = {"target": action.target, "by": "commander_agent"}
                    outcome["progress"] = True

        elif action.action_type == "secure":
            # Police/commander explicitly secures a zone. Stronger than dispatch
            # for zone_security: valid for 6 steps (vs 4 for dispatch).
            if action.agent_id in {"police_agent", "commander_agent"}:
                self.zone_security[tuple(action.target)] = {
                    "secured_by": action.agent_id,
                    "time_step": self.step_count,
                    "casualty_id": None,
                    "strong": True,  # secure is a stronger signal than dispatch
                }
                outcome["progress"] = True
                outcome["trust_delta"] += 0.6
                outcome["panic_delta"] -= 0.8
                outcome["coordinated_sequence"] = True
                actor["pos"] = self._step_towards(actor["pos"], action.target)

        elif action.action_type == "block":
            state = action.metadata.get("state", "blocked")
            if state == "open":
                self.blocked_roads.discard(tuple(action.target))
                outcome["progress"] = True
            else:
                self.blocked_roads.add(tuple(action.target))
                outcome["panic_delta"] += 0.5

        elif action.action_type == "allocate":
            amount = int(action.metadata.get("amount", 1))
            entity_type = action.metadata.get("entity", "hospital")
            target_entity = self._entity_at(action.target, entity_type)
            if target_entity is not None:
                target_entity["occupancy"] += max(0, amount)
                if target_entity["occupancy"] > target_entity["capacity"]:
                    outcome["overloaded"] = True
                    outcome["panic_delta"] += 2.0
                    outcome["trust_delta"] -= 1.5
                else:
                    cap = target_entity["capacity"]
                    fill = target_entity["occupancy"] / max(1, cap)
                    if 0.3 <= fill <= 0.8:
                        outcome["balanced_allocation"] = True
                        outcome["trust_delta"] += 1.0
                    outcome["progress"] = True

        elif action.action_type == "broadcast":
            text = str(action.metadata.get("text", "")).strip()
            truth = float(action.metadata.get("truth_score", 1.0))
            confidence = float(action.metadata.get("confidence", truth))
            message_id = f"m{self.message_counter}"
            self.message_counter += 1
            self.message_count += 1
            message = {
                "message_id": message_id,
                "sender": action.agent_id,
                "text": text,
                "target": tuple(action.target),
                "intent": str(action.metadata.get("intent", "coordination")),
                "confidence": max(0.0, min(1.0, confidence)),
                "timestamp": self.step_count,
                "truth_score": truth,
                "time_step": self.step_count,
                "used": False,
            }
            comm_failure_rate = float(self.scenario_profile.get("communication_failure_rate", 0.0))
            comm_integrity_factor = self.system_state["communication_integrity"] / 100.0
            if self.rng.random() <= comm_failure_rate * (1.2 - comm_integrity_factor):
                outcome["misinformation"] = True
                outcome["panic_delta"] += 1.5
                outcome["trust_delta"] -= 1.0
                outcome["progress"] = False
            else:
                self.messages.append(message)
            if truth < 0.5:
                outcome["misinformation"] = True
                outcome["panic_delta"] += 3.0
                outcome["trust_delta"] -= 3.0
                self.misinformation_impact += 1.0
            else:
                outcome["trust_delta"] += 0.5
                outcome["progress"] = True

        coordination_details = self._is_cooperation(action) if detect_cooperation else None
        outcome["cooperation"] = bool(coordination_details)
        outcome["coordination_details"] = coordination_details
        if coordination_details:
            self.coordination_hits += 1
            outcome["cooperation_confidence"] = float(coordination_details["effective_weight"])
        return outcome

    def _is_cooperation(self, action: ActionModel) -> Optional[Dict[str, Any]]:
        if action.action_type == "broadcast":
            return None
        trust_weight = self.trust_score / 100.0
        for message in reversed(self.messages[-10:]):
            if message["sender"] == action.agent_id:
                continue
            if self.step_count - message["time_step"] > 3:
                continue
            confidence = float(message.get("confidence", message.get("truth_score", 1.0)))
            effective_weight = confidence * trust_weight
            if tuple(message["target"]) == tuple(action.target) and effective_weight >= 0.45:
                message["used"] = True
                return {
                    "sender_agent_id": message["sender"],
                    "receiver_agent_id": action.agent_id,
                    "message_id": message.get("message_id", "unknown"),
                    "effective_weight": effective_weight,
                }
        return None

    def _explain_action(self, action: ActionModel, outcome: Dict[str, Any]) -> str:
        explicit_reason = str(action.metadata.get("reason", "")).strip()
        if explicit_reason:
            return explicit_reason
        if outcome.get("cooperation"):
            return "Action follows a recently verified coordination message."
        if action.action_type == "dispatch":
            casualty = self._casualty_at(tuple(action.target))
            if casualty and casualty.get("status") == "waiting":
                if int(casualty.get("severity", 0)) >= 3:
                    return "High severity casualty detected nearby."
                return "Casualty detected and dispatch initiated."
            return "Dispatching to probable incident location."
        if action.action_type == "broadcast":
            truth = float(action.metadata.get("truth_score", 1.0))
            if truth < 0.5:
                return "Broadcast had low confidence and increased panic."
            return "Broadcasting verified coordination signal."
        if action.action_type == "allocate":
            if outcome.get("overloaded"):
                return "Resource allocation exceeded capacity."
            if outcome.get("balanced_allocation"):
                return "Balanced resource allocation performed."
            return "Adjusting resource distribution."
        if action.action_type == "block":
            state = action.metadata.get("state", "blocked")
            return "Road opened to restore mobility." if state == "open" else "Road blocked to contain risk."
        return "Routing toward high-priority operational area."

    def generate_dynamic_events(self) -> List[Dict[str, Any]]:
        """Delegate to engine.py."""
        return generate_dynamic_events(
            rng=self.rng,
            scenario=self.scenario,
            scenario_profile=self.scenario_profile,
            step_count=self.step_count,
            grid_size=self.grid_size,
            active_effects=self.active_effects,
            event_log=self.event_log,
        )

    def apply_global_time_effects(self) -> Dict[str, Any]:
        """Delegate to engine.py. Apply results back to env state."""
        # Ambulance cooldown tick (handled locally as it mutates self)
        if self.ambulance_cooldowns:
            next_cooldowns: List[int] = []
            for remaining in self.ambulance_cooldowns:
                rem = remaining - 1
                if rem <= 0:
                    self.available_ambulances += 1
                else:
                    next_cooldowns.append(rem)
            self.ambulance_cooldowns = next_cooldowns

        result = apply_time_effects(
            rng=self.rng,
            scenario_profile=self.scenario_profile,
            casualties=self.casualties,
            hospitals=self.hospitals,
            messages=self.messages,
            active_effects=self.active_effects,
            system_state=self.system_state,
            district_panic=self.district_panic,
            blocked_roads=self.blocked_roads,
            fuel_level=self.fuel_level,
            grid_size=self.grid_size,
            step_count=self.step_count,
            message_ttl=self.message_ttl,
            misinformation_impact=self.misinformation_impact,
        )
        # Apply returned state back to env
        self.fuel_level = result["new_fuel"]
        self.active_effects = result["active_effects"]
        self.messages = result["messages"]
        self.misinformation_impact = result["misinformation_impact"]

        return {
            "panic_delta": result["panic_delta"],
            "trust_delta": result["trust_delta"],
            "deaths": result["deaths"],
            "effects": result["effects"],
        }

    def _hospital_overload_ratio(self) -> float:
        """Delegate to engine.py."""
        return hospital_overload_ratio(self.hospitals)

    def _reward_change_reason(self, components: Dict[str, float]) -> str:
        """Delegate to reward.py."""
        return reward_change_reason(components)

    def _dominant_system_factor(self, components: Dict[str, float]) -> str:
        """Delegate to reward.py."""
        from reward import explain_dominant
        return explain_dominant(components)

    def _advance_world_dynamics(self) -> Dict[str, int]:
        """Delegate to engine.py."""
        return advance_world_dynamics(self.casualties, self.hospitals, self.system_state)

    def _ignored_task_penalty(self, progress: bool) -> bool:
        open_tasks = any(c["status"] == "waiting" for c in self.casualties)
        if open_tasks and not progress:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        return self.no_progress_steps >= 2

    def _reward_components(
        self,
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
        triage_conflict: bool = False,
        strategic_tradeoff: bool = False,
        coordinated_sequence: bool = False,
        unsafe_medical_dispatch: bool = False,
        communication_spam: bool = False,
    ) -> Dict[str, float]:
        """Delegate to reward.py for single-source reward logic."""
        return compute_components(
            lives_saved=lives_saved,
            deaths=deaths,
            cooperation=cooperation,
            balanced_allocation=balanced_allocation,
            overloaded=overloaded,
            misinformation=misinformation,
            repeated_action=repeated_action,
            ignored_tasks=ignored_tasks,
            cooperation_confidence=cooperation_confidence,
            prev_snapshot=prev_snapshot,
            curr_panic=self.panic_level,
            curr_trust=self.trust_score,
            curr_infra=self.system_state["infrastructure_health"],
            curr_deaths=self.dead_count,
            triage_conflict=triage_conflict,
            strategic_tradeoff=strategic_tradeoff,
            coordinated_sequence=coordinated_sequence,
            unsafe_medical_dispatch=unsafe_medical_dispatch,
            communication_spam=communication_spam,
        )

    def _weighted_reward(self, components: Dict[str, float]) -> float:
        """Delegate to reward.py."""
        return weighted_total(components, self.reward_weights)

    def _all_casualties_handled(self) -> bool:
        return all(c["status"] in {"treated", "dead"} for c in self.casualties)

    def _emit_police_route_advisories(self) -> None:
        if not self.blocked_roads:
            return
        if self.step_count % 4 != 0:
            return
        road = next(iter(self.blocked_roads))
        message_id = f"m{self.message_counter}"
        self.message_counter += 1
        self.message_count += 1
        self.messages.append(
            {
                "message_id": message_id,
                "sender": "police_agent",
                "text": (
                    f"Route advisory for medical units: cell {list(road)} is blocked or unsafe "
                    "(possible bridge/road damage). Use an alternate approach; do not route dispatch through this cell."
                ),
                "target": tuple(road),
                "intent": "route_advisory",
                "confidence": 0.92,
                "timestamp": self.step_count,
                "truth_score": 1.0,
                "time_step": self.step_count,
                "used": False,
            }
        )

    def _all_observations(self) -> Dict[str, Dict[str, Any]]:
        # Compute per-step uncertainty (same for all agents; role observation filters differ)
        uncertainty = compute_uncertainty(
            blocked_roads=self.blocked_roads,
            communication_integrity=self.system_state["communication_integrity"],
            misinformation_impact=self.misinformation_impact,
            panic_level=self.panic_level,
        )
        observations = {aid: self._observation_for(aid) for aid in AGENT_IDS}
        for obs in observations.values():
            obs["uncertainty"] = uncertainty
            ObservationModel.model_validate(obs)
        return observations

    def _observation_for(self, agent_id: str, radius: int = 3) -> Dict[str, Any]:
        if agent_id == "commander_agent":
            return self._observation_commander()
        if agent_id == "medical_agent":
            return self._observation_medical()
        if agent_id == "police_agent":
            return self._observation_police()
        if agent_id == "logistics_agent":
            return self._observation_logistics()
        if agent_id == "communication_agent":
            return self._observation_communication()
        return self._observation_local(agent_id, radius)

    def _observation_local(self, agent_id: str, radius: int) -> Dict[str, Any]:
        ax, ay = self.agents[agent_id]["pos"]

        def visible(pos: Tuple[int, int]) -> bool:
            return abs(pos[0] - ax) + abs(pos[1] - ay) <= radius

        visible_events = []
        for event in self.events:
            if event.get("location") and visible(event["location"]):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting" and visible(casualty["pos"]):
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            if visible(road):
                visible_events.append({"type": "blocked_road", "pos": road})

        local_hospitals = [h for h in self.hospitals if visible(h["pos"])]
        local_shelters = [s for s in self.shelters if visible(s["pos"])]
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != agent_id and visible(data["pos"])
        }
        messages = self._filter_messages_for_agent(agent_id, ax, ay, visible)
        decision_context = self._decision_context_for(agent_id, visible_events, messages)
        scope = "Local grid view: only what you can see within a few cells."
        self_obj = copy.deepcopy(self.agents[agent_id])
        self_obj["knowledge_scope"] = scope
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": copy.deepcopy(local_hospitals),
                "shelters": copy.deepcopy(local_shelters),
                "local_blocked_roads": [r for r in self.blocked_roads if visible(r)],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _fmt_msg(self, m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "message_id": m.get("message_id"),
            "sender": m["sender"],
            "text": m["text"],
            "target": m["target"],
            "intent": m.get("intent", "coordination"),
            "confidence": m.get("confidence", m.get("truth_score", 1.0)),
            "timestamp": m.get("timestamp", m.get("time_step", 0)),
            "time_step": m["time_step"],
        }

    def _filter_messages_for_agent(
        self, agent_id: str, ax: int, ay: int, visible: Any
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for m in self.messages[-12:]:
            key = (m.get("message_id"), m.get("time_step"), m.get("sender"), m.get("text"))
            if key in seen:
                continue
            if m["sender"] == "commander_agent" or visible(tuple(m["target"])):
                seen.add(key)
                out.append(self._fmt_msg(m))
        return out[-8:]

    def _observation_commander(self) -> Dict[str, Any]:
        visible_events: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("location"):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting":
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            visible_events.append({"type": "blocked_road", "pos": road})

        messages = [self._fmt_msg(m) for m in self.messages[-12:]]
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != "commander_agent"
        }
        decision_context = self._decision_context_for("commander_agent", visible_events, messages)
        self_obj = copy.deepcopy(self.agents["commander_agent"])
        self_obj["knowledge_scope"] = (
            "Commander: full situational awareness — every calamity zone, all waiting casualties, "
            "all blocked roads/bridges, hospital and shelter capacity, and all unit positions."
        )
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": copy.deepcopy(self.hospitals),
                "shelters": copy.deepcopy(self.shelters),
                "local_blocked_roads": [tuple(r) for r in self.blocked_roads],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _observation_medical(self) -> Dict[str, Any]:
        agent_id = "medical_agent"
        radius = 4
        ax, ay = self.agents[agent_id]["pos"]

        def visible(pos: Tuple[int, int]) -> bool:
            return abs(pos[0] - ax) + abs(pos[1] - ay) <= radius

        visible_events: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("location") and visible(event["location"]):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting" and visible(casualty["pos"]):
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            if visible(road):
                visible_events.append({"type": "blocked_road", "pos": road})

        local_hospitals = [h for h in self.hospitals if visible(h["pos"])]
        if not local_hospitals:
            local_hospitals = copy.deepcopy(self.hospitals)
        local_shelters = [s for s in self.shelters if visible(s["pos"])]
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != agent_id and visible(data["pos"])
        }
        messages: List[Dict[str, Any]] = []
        seen: set = set()
        for m in self.messages[-16:]:
            key = (m.get("message_id"), m.get("time_step"), m.get("text"))
            if key in seen:
                continue
            sender = m["sender"]
            if sender in {"police_agent", "commander_agent", "communication_agent"}:
                seen.add(key)
                messages.append(self._fmt_msg(m))
            elif visible(tuple(m["target"])):
                seen.add(key)
                messages.append(self._fmt_msg(m))
        messages = messages[-10:]

        decision_context = self._decision_context_for(agent_id, visible_events, messages)
        self_obj = copy.deepcopy(self.agents[agent_id])
        self_obj["knowledge_scope"] = (
            "Medical / ambulance: nearby casualties and hospitals you can reach; "
            "you always receive police route advisories and commander orders."
        )
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": copy.deepcopy(local_hospitals),
                "shelters": copy.deepcopy(local_shelters),
                "local_blocked_roads": [r for r in self.blocked_roads if visible(r)],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _observation_police(self) -> Dict[str, Any]:
        agent_id = "police_agent"
        radius = 4
        ax, ay = self.agents[agent_id]["pos"]

        def visible(pos: Tuple[int, int]) -> bool:
            return abs(pos[0] - ax) + abs(pos[1] - ay) <= radius

        visible_events: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("location") and visible(event["location"]):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting" and visible(casualty["pos"]):
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            visible_events.append({"type": "blocked_road", "pos": road})

        local_hospitals = [h for h in self.hospitals if visible(h["pos"])]
        local_shelters = [s for s in self.shelters if visible(s["pos"])]
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != agent_id and visible(data["pos"])
        }
        messages = self._filter_messages_for_agent(agent_id, ax, ay, visible)
        decision_context = self._decision_context_for(agent_id, visible_events, messages)
        self_obj = copy.deepcopy(self.agents[agent_id])
        self_obj["knowledge_scope"] = (
            "Police: local incidents plus citywide road/bridge closures so you can advise ambulances on safe detours."
        )
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": copy.deepcopy(local_hospitals),
                "shelters": copy.deepcopy(local_shelters),
                "local_blocked_roads": [tuple(r) for r in self.blocked_roads],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _observation_logistics(self) -> Dict[str, Any]:
        agent_id = "logistics_agent"
        radius = 4
        ax, ay = self.agents[agent_id]["pos"]

        def visible(pos: Tuple[int, int]) -> bool:
            return abs(pos[0] - ax) + abs(pos[1] - ay) <= radius

        visible_events: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("location") and visible(event["location"]):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting" and visible(casualty["pos"]):
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            if visible(road):
                visible_events.append({"type": "blocked_road", "pos": road})

        local_hospitals = copy.deepcopy(self.hospitals)
        local_shelters = copy.deepcopy(self.shelters)
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != agent_id and visible(data["pos"])
        }
        messages: List[Dict[str, Any]] = []
        seen: set = set()
        for m in self.messages[-12:]:
            key = (m.get("message_id"), m.get("time_step"), m.get("text"))
            if key in seen:
                continue
            if m["sender"] in {"commander_agent", "logistics_agent"} or visible(tuple(m["target"])):
                seen.add(key)
                messages.append(self._fmt_msg(m))
        messages = messages[-8:]

        decision_context = self._decision_context_for(agent_id, visible_events, messages)
        self_obj = copy.deepcopy(self.agents[agent_id])
        self_obj["knowledge_scope"] = (
            "Logistics: shelter and hospital load across the city; commander logistics orders; local traffic cues."
        )
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": local_hospitals,
                "shelters": local_shelters,
                "local_blocked_roads": [r for r in self.blocked_roads if visible(r)],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _observation_communication(self) -> Dict[str, Any]:
        agent_id = "communication_agent"
        radius = 4
        ax, ay = self.agents[agent_id]["pos"]

        def visible(pos: Tuple[int, int]) -> bool:
            return abs(pos[0] - ax) + abs(pos[1] - ay) <= radius

        visible_events: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("location") and visible(event["location"]):
                visible_events.append(copy.deepcopy(event))
        for casualty in self.casualties:
            if casualty["status"] == "waiting" and visible(casualty["pos"]):
                visible_events.append(
                    {
                        "type": "casualty",
                        "id": casualty["id"],
                        "pos": casualty["pos"],
                        "severity": casualty["severity"],
                    }
                )
        for road in self.blocked_roads:
            if visible(road):
                visible_events.append({"type": "blocked_road", "pos": road})

        local_hospitals = [h for h in self.hospitals if visible(h["pos"])]
        local_shelters = [s for s in self.shelters if visible(s["pos"])]
        nearby_agents = {
            aid: {"pos": data["pos"], "assigned_task": data["assigned_task"]}
            for aid, data in self.agents.items()
            if aid != agent_id and visible(data["pos"])
        }
        messages = [self._fmt_msg(m) for m in self.messages[-10:]]
        decision_context = self._decision_context_for(agent_id, visible_events, messages)
        self_obj = copy.deepcopy(self.agents[agent_id])
        self_obj["knowledge_scope"] = (
            "Communications: recent public/coordination traffic only; no full commander picture."
        )
        return {
            "visible_events": visible_events,
            "resource_status": {
                "hospitals": copy.deepcopy(local_hospitals),
                "shelters": copy.deepcopy(local_shelters),
                "local_blocked_roads": [r for r in self.blocked_roads if visible(r)],
            },
            "agent_status": {
                "self": self_obj,
                "nearby_agents": nearby_agents,
                "panic_level_estimate": self.panic_level,
                "mobility_index": self.system_state["mobility_index"],
                "communication_integrity": self.system_state["communication_integrity"],
            },
            "messages": messages,
            "decision_context": decision_context,
            "time_step": self.step_count,
        }

    def _spawn_point(self, agent_id: str) -> Tuple[int, int]:
        mapping = {
            # Spread starts across interior city cells so markers are visible at load.
            "medical_agent": (3, 4),
            "police_agent": (4, 6),
            "logistics_agent": (6, 4),
            "communication_agent": (7, 6),
            "commander_agent": (5, 5),
        }
        return mapping[agent_id]

    def _step_towards(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
        if self.fuel_level <= 0:
            return src
        sx, sy = src
        tx, ty = dst
        nx, ny = sx, sy
        if sx < tx:
            nx += 1
        elif sx > tx:
            nx -= 1
        elif sy < ty:
            ny += 1
        elif sy > ty:
            ny -= 1
        upper = self.grid_size - 1
        candidate = (max(0, min(upper, nx)), max(0, min(upper, ny)))
        mobility_index = self.system_state["mobility_index"]
        if self.rng.random() > mobility_index:
            return src
        self.fuel_level = max(0.0, self.fuel_level - (1.0 + (0.4 if self.fuel_level < 20 else 0.0)))
        return src if candidate in self.blocked_roads else candidate

    def _casualty_at(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        for casualty in self.casualties:
            if tuple(casualty["pos"]) == tuple(pos):
                return casualty
        return None

    def _entity_at(self, pos: Tuple[int, int], entity_type: str) -> Optional[Dict[str, Any]]:
        entities = self.hospitals if entity_type == "hospital" else self.shelters
        for entity in entities:
            if tuple(entity["pos"]) == tuple(pos):
                return entity
        return None

    def _decision_context_for(
        self, agent_id: str, visible_events: List[Dict[str, Any]], messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        waiting_casualties = [e for e in visible_events if e.get("type") == "casualty"]
        waiting_casualties = sorted(
            waiting_casualties,
            key=lambda e: (-int(e.get("severity", 0)), e["pos"][0], e["pos"][1]),
        )
        critical_zones = [c["pos"] for c in waiting_casualties[:3]]
        resource_gaps = []
        for hospital in self.hospitals:
            gap = float(hospital["occupancy"]) - float(hospital["capacity"])
            if gap > 0:
                resource_gaps.append(
                    {"entity": "hospital", "id": hospital["id"], "pos": hospital["pos"], "overload": gap}
                )
        recommended_actions: List[Dict[str, Any]] = []
        if critical_zones:
            best_target = critical_zones[0]
            recommended_actions.append(
                {
                    "agent_id": agent_id,
                    "action_type": "dispatch" if agent_id in {"medical_agent", "commander_agent"} else "route",
                    "target": list(best_target),
                    "why": "highest_severity_first",
                }
            )
        if agent_id == "communication_agent" and critical_zones:
            recommended_actions.append(
                {
                    "agent_id": agent_id,
                    "action_type": "broadcast",
                    "target": list(critical_zones[0]),
                    "why": "coordination_signal",
                }
            )
        if resource_gaps and agent_id in {"logistics_agent", "commander_agent"}:
            recommended_actions.append(
                {
                    "agent_id": agent_id,
                    "action_type": "allocate",
                    "target": list(resource_gaps[0]["pos"]),
                    "why": "resource_gap",
                }
            )
        risk_zones = [event["pos"] for event in visible_events if event.get("type") == "blocked_road"]
        return {
            "critical_zones": critical_zones,
            "risk_zones": risk_zones,
            "resource_gaps": resource_gaps,
            "recommended_actions": recommended_actions,
            "message_pressure": len(messages),
            "available_ambulances": self.available_ambulances,
            "fuel_level": self.fuel_level,
        }
