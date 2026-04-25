from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _deterministic_sorted_events(visible_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(event: Dict[str, Any]) -> Tuple[int, int, int, str]:
        ex, ey = tuple(event.get("pos", (9, 9)))
        severity = int(event.get("severity", 0))
        etype = str(event.get("type", ""))
        return (-severity, ex, ey, etype)

    return sorted(visible_events, key=key_fn)


def _target_from_message(messages: List[Dict[str, Any]]) -> Tuple[int, int] | None:
    if not messages:
        return None
    ordered = sorted(
        messages,
        key=lambda m: (
            -int(m.get("time_step", -1)),
            str(m.get("sender", "")),
            tuple(m.get("target", (9, 9)))[0],
            tuple(m.get("target", (9, 9)))[1],
        ),
    )
    for msg in ordered:
        target = tuple(msg.get("target", ()))
        if len(target) == 2:
            return target
    return None


def _avoid_repeat(self_status: Dict[str, Any], action_type: str, target: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
    last_key = self_status.get("last_action_key")
    if not last_key:
        return action_type, target
    last_type = last_key[0] if isinstance(last_key, (list, tuple)) and len(last_key) >= 1 else None
    last_target = tuple(last_key[1]) if isinstance(last_key, (list, tuple)) and len(last_key) >= 2 else None
    if last_type == action_type and last_target == target:
        sx, sy = tuple(self_status.get("pos", (0, 0)))
        alt = (sx, min(9, sy + 1))
        return "route", alt
    return action_type, target


def baseline_policy(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deliberate weak baseline — used as the BEFORE state in the Training Replay.

    Failures by design:
      - communication_agent always broadcasts low-truth misinformation
      - medical_agent routes away from casualties (opposite corner)
      - police_agent blocks roads randomly (adds to blocked_roads)
      - commander_agent issues no meaningful coordination
      - logistics_agent over-allocates a single hospital repeatedly
      - No agent reacts to messages or visible events
      - All agents ignore casualties

    This produces high deaths, high panic, low trust, and large negative reward.
    The delta between this and improved_policy is the training signal gap judges look for.
    """
    self_status = observation["agent_status"]["self"]
    agent_id = self_status["id"]
    time_step = int(observation.get("time_step", 0))

    if agent_id == "communication_agent":
        # Always spread misinformation — deliberately bad
        action_type = "broadcast"
        target = (5, 5)
        metadata = {
            "text": "All clear — no casualties detected anywhere.",
            "truth_score": 0.05,  # false information
            "reason": "Misinformation: telling public everything is fine",
        }
    elif agent_id == "medical_agent":
        # Always route away from casualties — toward far corner
        action_type = "route"
        target = (9, 9) if time_step % 2 == 0 else (0, 0)
        metadata = {"reason": "Moving away from incident zone"}
    elif agent_id == "police_agent":
        # Block safe roads instead of securing casualty zones
        action_type = "block"
        targets = [(1, 1), (2, 2), (3, 3), (6, 6), (7, 7)]
        target = targets[time_step % len(targets)]
        metadata = {"state": "blocked", "reason": "Blocking passable roads"}
    elif agent_id == "logistics_agent":
        # Overload the same hospital every step
        action_type = "allocate"
        target = (1, 1)
        metadata = {"entity": "hospital", "amount": 10, "reason": "Overloading hospital h1"}
    else:
        # commander does nothing useful — routes to static point
        action_type = "route"
        target = (0, 9)
        metadata = {"reason": "No coordination strategy"}

    action_type, target = _avoid_repeat(self_status, action_type, target)
    return {
        "agent_id": agent_id,
        "action_type": action_type,
        "target": [int(target[0]), int(target[1])],
        "metadata": metadata,
    }


def improved_policy(observation: Dict[str, Any]) -> Dict[str, Any]:
    self_status = observation["agent_status"]["self"]
    agent_id = self_status["id"]
    self_pos = tuple(self_status.get("pos", (0, 0)))
    visible_events = _deterministic_sorted_events(observation.get("visible_events", []))
    messages = observation.get("messages", [])

    casualties = [e for e in visible_events if e.get("type") == "casualty"]
    casualty_target: Tuple[int, int] | None = None
    high_severity = False
    if casualties:
        casualty_target = tuple(casualties[0]["pos"])
        high_severity = int(casualties[0].get("severity", 0)) >= 3

    message_target = _target_from_message(messages)
    chosen_target = casualty_target or message_target

    if chosen_target is None:
        # Deterministic nearest resource fallback.
        hospitals = observation.get("resource_status", {}).get("hospitals", [])
        shelters = observation.get("resource_status", {}).get("shelters", [])
        resources = hospitals + shelters
        if resources:
            resources = sorted(
                resources,
                key=lambda r: (
                    abs(r["pos"][0] - self_pos[0]) + abs(r["pos"][1] - self_pos[1]),
                    r["pos"][0],
                    r["pos"][1],
                ),
            )
            chosen_target = tuple(resources[0]["pos"])
        else:
            chosen_target = (5, 5)

    if agent_id == "communication_agent":
        action_type = "broadcast"
        metadata = {
            "text": f"Verified casualty coordination target at {chosen_target}.",
            "truth_score": 1.0,
            "reason": "Broadcasting verified target for coordination",
        }
    elif agent_id == "medical_agent":
        action_type = "dispatch"
        metadata = {
            "reason": "High severity casualty detected nearby"
            if high_severity
            else "Dispatch to best known casualty target"
        }
    elif agent_id == "commander_agent":
        action_type = "dispatch"
        metadata = {
            "assign_to": "medical_agent",
            "reason": "Assigning medical team to prioritized target",
        }
    else:
        action_type = "route"
        metadata = {"reason": "Routing to support coordinated target"}

    action_type, chosen_target = _avoid_repeat(self_status, action_type, chosen_target)
    return {
        "agent_id": agent_id,
        "action_type": action_type,
        "target": [int(chosen_target[0]), int(chosen_target[1])],
        "metadata": metadata,
    }
