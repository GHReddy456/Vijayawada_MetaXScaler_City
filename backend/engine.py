"""
engine.py - Pure world simulation engine for CrisisWorld.

All functions here are stateless: they take explicit arguments and return
explicit results. No reward logic, no observations, no API concerns live here.

This module owns:
  - Scenario profiles
  - Agent movement
  - Dynamic event generation
  - Time-based world effects (panic, trust, fuel, hospital overload)
  - Casualty deterioration
  - Causal chain construction
  - Per-step uncertainty quantification
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Scenario profiles (single source of truth)
# ---------------------------------------------------------------------------

SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "earthquake": {
        "panic_growth_rate": 2.5,
        "aftershock_probability": 0.1,
        "road_block_probability": 0.2,
        "hospital_overload_rate": 1.8,
    },
    "flood": {
        "panic_growth_rate": 1.8,
        "water_spread_rate": 0.3,
        "mobility_penalty": 0.5,
    },
    "blackout": {
        "panic_growth_rate": 2.0,
        "communication_failure_rate": 0.25,
        "trust_decay_rate": 1.5,
    },
}

# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def step_towards(
    pos: Tuple[int, int],
    target: Tuple[int, int],
    grid_size: int = 10,
) -> Tuple[int, int]:
    """Move one Manhattan step from pos toward target, clamped to grid."""
    x, y = pos
    tx, ty = target
    if x != tx:
        x += 1 if tx > x else -1
    elif y != ty:
        y += 1 if ty > y else -1
    return (max(0, min(grid_size - 1, x)), max(0, min(grid_size - 1, y)))


# ---------------------------------------------------------------------------
# Hospital/shelter utilities
# ---------------------------------------------------------------------------


def hospital_overload_ratio(hospitals: List[Dict[str, Any]]) -> float:
    """Return average overload ratio across all hospitals (0.0 = no overload)."""
    if not hospitals:
        return 0.0
    overload = 0.0
    for h in hospitals:
        cap = max(1.0, float(h["capacity"]))
        overload += max(0.0, float(h.get("occupancy", 0)) - cap) / cap
    return overload / len(hospitals)


# ---------------------------------------------------------------------------
# Dynamic event generation
# ---------------------------------------------------------------------------


def generate_dynamic_events(
    rng: random.Random,
    scenario: str,
    scenario_profile: Dict[str, Any],
    step_count: int,
    grid_size: int,
    active_effects: List[Dict[str, Any]],
    event_log: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate stochastic scenario events for the current step.

    Mutates `active_effects` and `event_log` (caller-owned lists).
    Returns the list of events generated this step.
    """
    profile = scenario_profile
    generated: List[Dict[str, Any]] = []

    if scenario == "earthquake" and rng.random() < float(
        profile.get("aftershock_probability", 0.0)
    ):
        ev: Dict[str, Any] = {
            "type": "aftershock",
            "location": (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)),
            "remaining": 3,
        }
        active_effects.append(ev)
        generated.append(ev)

    if rng.random() < float(profile.get("road_block_probability", 0.0)) * 0.4:
        ev = {
            "type": "bridge_collapse",
            "location": (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)),
            "remaining": 4,
        }
        active_effects.append(ev)
        generated.append(ev)

    if scenario == "blackout" and rng.random() < float(
        profile.get("communication_failure_rate", 0.0)
    ):
        ev = {"type": "communication_failure", "remaining": 3}
        active_effects.append(ev)
        generated.append(ev)

    if rng.random() < 0.06:
        ev = {"type": "misinformation_burst", "remaining": 2}
        active_effects.append(ev)
        generated.append(ev)

    if rng.random() < 0.05:
        ev = {"type": "fuel_shortage_spike", "remaining": 3}
        active_effects.append(ev)
        generated.append(ev)

    if generated:
        event_log.extend([dict(e, step=step_count) for e in generated])

    return generated


# ---------------------------------------------------------------------------
# Time-based world effects
# ---------------------------------------------------------------------------


def apply_time_effects(
    *,
    rng: random.Random,
    scenario_profile: Dict[str, Any],
    casualties: List[Dict[str, Any]],
    hospitals: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    active_effects: List[Dict[str, Any]],
    system_state: Dict[str, float],
    district_panic: Dict[Tuple[int, int], float],
    blocked_roads: Set[Tuple[int, int]],
    fuel_level: float,
    grid_size: int,
    step_count: int,
    message_ttl: int,
    misinformation_impact: float,
) -> Dict[str, Any]:
    """Apply all time-based world effects for one step.

    Returns a dict of deltas and updated collections.
    Caller must apply returned values back to env state.
    """
    profile = scenario_profile
    effects: List[str] = []
    panic_delta = 0.0
    trust_delta = 0.0
    deaths = 0

    # -- Message expiry ----------------------------------------------------
    active_messages = [
        m
        for m in messages
        if step_count - int(m.get("timestamp", m.get("time_step", 0))) <= message_ttl
    ]

    # -- Panic propagation from waiting casualties -------------------------
    growth = float(profile.get("panic_growth_rate", 1.5)) * 0.05
    for casualty in casualties:
        if casualty["status"] != "waiting":
            continue
        cx, cy = casualty["pos"]
        local = growth * float(casualty["severity"])
        district_panic[(cx, cy)] = district_panic.get((cx, cy), 0.0) + local
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                district_panic[(nx, ny)] = (
                    district_panic.get((nx, ny), 0.0) + local * 0.35
                )
        panic_delta += local * 0.4
    if panic_delta > 0:
        effects.append("Panic propagated from active casualty zones.")

    # -- Trust dynamics from message quality --------------------------------
    recent = active_messages[-8:]
    low_conf = any(
        float(m.get("confidence", m.get("truth_score", 1.0))) < 0.5 for m in recent
    )
    high_conf = any(
        float(m.get("confidence", m.get("truth_score", 1.0))) >= 0.8 for m in recent
    )
    if low_conf:
        trust_delta -= 0.8 * float(profile.get("trust_decay_rate", 1.0))
        misinformation_impact += 0.2
        effects.append("Trust decayed due to low-confidence communication.")
    elif high_conf:
        trust_delta += 0.4
        effects.append("Trust improved due to verified communication.")

    # -- Fuel & hospital resource degradation ------------------------------
    new_fuel = max(0.0, fuel_level - 0.25)
    if any(c["status"] == "waiting" for c in casualties):
        for hospital in hospitals:
            hospital["occupancy"] = float(hospital.get("occupancy", 0)) + float(
                profile.get("hospital_overload_rate", 1.0)
            ) * 0.02

    if new_fuel < 35:
        system_state["mobility_index"] = max(
            0.25, system_state.get("mobility_index", 1.0) - 0.02
        )
        effects.append("Fuel degradation reduced mobility.")

    # -- Active effect processing ------------------------------------------
    remaining_effects: List[Dict[str, Any]] = []
    for effect in active_effects:
        etype = effect["type"]
        if etype == "aftershock":
            panic_delta += 0.9
            system_state["infrastructure_health"] = (
                system_state.get("infrastructure_health", 100.0) - 0.6
            )
            blocked_roads.add(tuple(effect["location"]))  # type: ignore[arg-type]
        elif etype == "bridge_collapse":
            blocked_roads.add(tuple(effect["location"]))  # type: ignore[arg-type]
            system_state["mobility_index"] = max(
                0.2, system_state.get("mobility_index", 1.0) - 0.03
            )
        elif etype == "misinformation_burst":
            trust_delta -= 1.2
            panic_delta += 0.6
            misinformation_impact += 0.5
        elif etype == "communication_failure":
            system_state["communication_integrity"] = max(
                0.0, system_state.get("communication_integrity", 100.0) - 1.4
            )
            trust_delta -= 0.4
        elif etype == "fuel_shortage_spike":
            new_fuel = max(0.0, new_fuel - 1.2)
            system_state["mobility_index"] = max(
                0.15, system_state.get("mobility_index", 1.0) - 0.04
            )
        effect["remaining"] -= 1
        if effect["remaining"] > 0:
            remaining_effects.append(effect)

    # -- Delayed deaths from hospital overload -----------------------------
    overload = hospital_overload_ratio(hospitals)
    if overload > 0.1:
        for casualty in casualties:
            if (
                casualty["status"] == "waiting"
                and casualty["time_waiting"] >= 4
                and rng.random() < overload * 0.08
            ):
                casualty["status"] = "dead"
                deaths += 1
        if deaths > 0:
            effects.append("Hospital overload increased delayed mortality.")

    # -- System state bounds -----------------------------------------------
    system_state["infrastructure_health"] = max(
        0.0,
        min(
            100.0,
            system_state.get("infrastructure_health", 100.0)
            - (len(blocked_roads) / max(1, grid_size * 2)) * 0.1,
        ),
    )
    system_state["communication_integrity"] = max(
        0.0, min(100.0, system_state.get("communication_integrity", 100.0))
    )
    system_state["mobility_index"] = max(
        0.1,
        min(
            1.0,
            1.0
            - (len(blocked_roads) / float(max(1, grid_size * grid_size)))
            - (0.15 if new_fuel < 30 else 0.0),
        ),
    )

    return {
        "panic_delta": panic_delta,
        "trust_delta": trust_delta,
        "deaths": deaths,
        "effects": effects,
        "new_fuel": new_fuel,
        "active_effects": remaining_effects,
        "messages": active_messages,
        "misinformation_impact": misinformation_impact,
    }


# ---------------------------------------------------------------------------
# Casualty deterioration
# ---------------------------------------------------------------------------


def advance_world_dynamics(
    casualties: List[Dict[str, Any]],
    hospitals: List[Dict[str, Any]],
    system_state: Dict[str, float],
) -> Dict[str, int]:
    """Advance casualty deterioration for one time step.

    Mutates casualty statuses in-place.
    Returns {'deaths': int, 'lives_saved': int}.
    """
    deaths = 0
    overload_ratio = hospital_overload_ratio(hospitals)
    for casualty in casualties:
        if casualty["status"] != "waiting":
            continue
        casualty["time_waiting"] += 1
        death_threshold = 8 - casualty["severity"]
        if overload_ratio > 0.0:
            death_threshold -= int(overload_ratio * 2)
        if casualty["time_waiting"] >= max(2, death_threshold):
            casualty["status"] = "dead"
            deaths += 1
            system_state["infrastructure_health"] = max(
                0.0, system_state.get("infrastructure_health", 100.0) - 0.6
            )
    return {"deaths": deaths, "lives_saved": 0}


# ---------------------------------------------------------------------------
# Causal chain construction
# ---------------------------------------------------------------------------


def build_causal_chain(
    outcome: Dict[str, Any],
    components: Dict[str, float],
    step_count: int = 0,
    panic_level: float = 0.0,
    overload_ratio: float = 0.0,
    waiting_casualties: int = 0,
) -> List[str]:
    """Build a fully-expanded multi-step causal chain from one step's outcome.

    Format:  TRIGGER -> INTERMEDIATE EFFECT -> SYSTEM STATE -> REWARD CONSEQUENCE

    Example:
      "Rescue delayed 4 steps -> Hospital intake pressure rising -> Capacity at 85% ->
       Casualty timed out -> 1 death -> Reward: -100 survival penalty applied"

    These chains are shown in the frontend causal chain panel and can be used as
    interpretability evidence in demos and benchmarks.
    """
    chain: List[str] = []

    # ── Rescue success chain ─────────────────────────────────────────────────
    if outcome.get("lives_saved", 0) > 0:
        saved = outcome["lives_saved"]
        reward_signal = round(50.0 * saved, 1)
        if outcome.get("coordinated_sequence"):
            chain.append(
                f"[COORDINATION SUCCESS] Police secured zone "
                f"-> Medical cleared safe path "
                f"-> Dispatch reached casualty "
                f"-> {saved} casualty/casualties treated "
                f"-> Reward +{reward_signal} (survival) +14 (sequence bonus)"
            )
        else:
            chain.append(
                f"[RESCUE] Dispatch reached casualty location "
                f"-> Treatment administered "
                f"-> {saved} life/lives saved "
                f"-> Reward +{reward_signal} survival bonus"
            )

    # ── Death consequence chain (full multi-step) ───────────────────────────
    if outcome.get("deaths", 0) > 0:
        deaths = outcome["deaths"]
        death_penalty = round(100.0 * deaths, 1)
        if components.get("triage_conflict_penalty", 0.0) < 0:
            chain.append(
                f"[TRIAGE FAILURE] All ambulances deployed "
                f"-> No unit available for new casualty "
                f"-> Triage conflict at hospital "
                f"-> {deaths} casualty untreated "
                f"-> Reward -12 (triage) -100 (death penalty)"
            )
        elif overload_ratio > 0.05:
            chain.append(
                f"[OVERLOAD CASCADE] Rescue delayed {step_count} steps "
                f"-> Casualty waited too long "
                f"-> Hospital overload ratio {overload_ratio:.0%} "
                f"-> Capacity pressure exceeded "
                f"-> {deaths} death(s) occurred "
                f"-> Reward -{death_penalty} survival penalty triggered"
            )
        else:
            chain.append(
                f"[DELAY PENALTY] Rescue not dispatched in time "
                f"-> Casualty severity timer expired "
                f"-> {deaths} death(s) occurred "
                f"-> Panic +{round(panic_level * 0.1, 1)} "
                f"-> Reward -{death_penalty} survival penalty"
            )

    # ── Police-Medical coordination chain ────────────────────────────────────
    if outcome.get("coordinated_sequence") and outcome.get("lives_saved", 0) == 0:
        chain.append(
            "[SEQUENCE REWARD] Police secured zone "
            "-> Zone marked safe in memory "
            "-> Medical dispatch confirmed safe corridor "
            "-> Coordination sequence valid "
            "-> Reward +14 coordination bonus"
        )

    if outcome.get("unsafe_medical_dispatch"):
        chain.append(
            "[UNSAFE DISPATCH] Medical sent to unsecured zone "
            "-> Police not yet on-site "
            "-> No safe corridor established "
            "-> Increased risk to unit and casualty "
            "-> Reward -10 unsafe dispatch penalty"
        )

    # ── Misinformation chain ─────────────────────────────────────────────────
    if outcome.get("misinformation"):
        chain.append(
            "[MISINFORMATION] Low-confidence broadcast transmitted "
            "-> Public receives conflicting information "
            "-> Trust score decreasing "
            f"-> Panic rising (currently {round(panic_level, 1)}) "
            "-> Reward -15 misinformation penalty"
        )

    # ── Coordination message chain ────────────────────────────────────────────
    elif outcome.get("cooperation"):
        conf = round(float(outcome.get("cooperation_confidence", 0.0)) * 10, 1)
        chain.append(
            "[COORDINATION] Agent followed verified coordination message "
            f"-> Message confidence weight applied "
            f"-> Joint action aligned "
            f"-> Trust improving "
            f"-> Reward +{conf} coordination reward"
        )

    # ── Resource chain ────────────────────────────────────────────────────────
    if outcome.get("overloaded"):
        chain.append(
            "[RESOURCE OVERLOAD] Allocation exceeded hospital capacity "
            f"-> Overload ratio: {overload_ratio:.0%} "
            "-> Secondary casualty processing degraded "
            "-> Panic spike triggered "
            "-> Reward -20 overload penalty"
        )
    elif outcome.get("balanced_allocation"):
        chain.append(
            "[RESOURCE BALANCE] Load distributed across facilities "
            "-> 30-80% utilization maintained "
            "-> Response capacity preserved "
            "-> Reward +10 efficiency bonus"
        )

    # ── System effect chain ───────────────────────────────────────────────────
    for eff in outcome.get("system_effects", []):
        entry = f"[SYSTEM] {eff}"
        if entry not in chain:
            chain.append(entry)

    # ── Waiting pressure note ─────────────────────────────────────────────────
    if waiting_casualties > 0 and outcome.get("lives_saved", 0) == 0 and outcome.get("deaths", 0) == 0:
        chain.append(
            f"[PENDING] {waiting_casualties} casualty/casualties still waiting "
            "-> Each step without rescue increases death probability "
            "-> Ignored task penalty accumulating if no progress"
        )

    return chain if chain else ["[STABLE] No adverse causal chain this step."]


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------


def compute_uncertainty(
    blocked_roads: Set[Tuple[int, int]],
    communication_integrity: float,
    misinformation_impact: float,
    panic_level: float,
) -> Dict[str, float]:
    """Return per-source uncertainty estimates for observation augmentation.

    Values are in [0, 1]. Agents can use this to weight their decisions.
    """
    comm_noise = max(0.0, 1.0 - communication_integrity / 100.0)
    road_noise = min(1.0, len(blocked_roads) / 20.0)
    info_noise = min(1.0, misinformation_impact / 10.0)
    panic_noise = min(1.0, panic_level / 100.0)
    overall = (comm_noise + road_noise + info_noise + panic_noise) / 4.0
    return {
        "communication": round(comm_noise, 3),
        "road_network": round(road_noise, 3),
        "information": round(info_noise, 3),
        "panic": round(panic_noise, 3),
        "overall": round(overall, 3),
    }
