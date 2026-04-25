"""
comms.py — Selective, targeted, necessity-driven communication engine.

Agents communicate ONLY when:
  1. Information Gap  — confidence < 0.6 OR uncertainty > 0.5
  2. Support Required — action needs another agent to be effective

Messages are targeted (agent → specific agent, never broadcast-to-all).
Anti-spam: max 2 messages per agent per step; penalty applied for excess.

Reward integration:
  +10  message led to successful coordination
  -10  useful message was ignored
  -10  unnecessary message (high confidence + no support needed)
  -20  wrong/misinformation message caused failure
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ─── Role dependency map ───────────────────────────────────────────────────────
# Defines which agents an agent depends on for specific situations.
# key = (agent_id, reason) → target_agent_id
DEPENDENCY_MAP: Dict[Tuple[str, str], str] = {
    ("medical_agent", "route_unknown"):       "police_agent",
    ("medical_agent", "resource_conflict"):   "logistics_agent",
    ("medical_agent", "needs_escort"):        "police_agent",
    ("medical_agent", "priority_unclear"):    "commander_agent",
    ("police_agent",  "public_panic"):        "communication_agent",
    ("police_agent",  "resource_conflict"):   "commander_agent",
    ("police_agent",  "casualty_update"):     "medical_agent",
    ("logistics_agent", "capacity_unknown"):  "medical_agent",
    ("logistics_agent", "priority_unclear"):  "commander_agent",
    ("logistics_agent", "resource_conflict"): "commander_agent",
    ("communication_agent", "route_unknown"): "police_agent",
    ("communication_agent", "severity_unconfirmed"): "commander_agent",
    ("communication_agent", "misinformation"): "commander_agent",
    ("commander_agent", "resource_conflict"): "logistics_agent",
    ("commander_agent", "route_update"):      "police_agent",
    ("commander_agent", "casualty_critical"): "medical_agent",
}

# ─── Message type colors (for frontend) ───────────────────────────────────────
MSG_TYPE_COLOR: Dict[str, str] = {
    "request":      "#4499ff",   # blue
    "advisory":     "#00ff88",   # green
    "alert":        "#ff3355",   # red
    "coordination": "#bb44ff",   # purple
    "override":     "#ff9900",   # orange
}

# Roles that have high situational awareness (reduce info-gap triggers)
HIGH_AWARENESS_ROLES = {"commander_agent"}


# ─── Communication need assessment ────────────────────────────────────────────

def _compute_obs_confidence(observation: Dict[str, Any]) -> float:
    """Derive an overall confidence score from the observation's uncertainty dict."""
    unc = observation.get("uncertainty", {})
    if not unc:
        return 0.85  # assume moderate confidence if no uncertainty data
    values = [float(v) for v in unc.values() if isinstance(v, (int, float))]
    if not values:
        return 0.85
    avg_uncertainty = sum(values) / len(values)
    return max(0.0, min(1.0, 1.0 - avg_uncertainty))


def _requires_other_agent(agent_id: str, action: Dict[str, Any], observation: Dict[str, Any]) -> Optional[str]:
    """
    Determine if the planned action requires another agent's support.
    Returns the reason string if support is needed, None otherwise.
    """
    atype = action.get("action_type", "")
    target = tuple(action.get("target", [5, 5]))
    blocked = {
        tuple(x)
        for x in observation.get("resource_status", {}).get("local_blocked_roads", [])
    }
    secured = {
        tuple(x)
        for x in observation.get("resource_status", {}).get("secured_zones", [])
    }

    if agent_id == "medical_agent":
        # Medical needs escort if target is in a blocked/unsecured zone
        if target in blocked:
            return "route_unknown"
        # Medical needs police escort if zone is not yet secured
        if target not in secured and atype in {"dispatch"}:
            casual = observation.get("visible_events", [])
            high_sev = any(e.get("type") == "casualty" and e.get("severity", 0) >= 3 for e in casual)
            if high_sev:
                return "needs_escort"
        # Hospital resource conflict
        hosp = observation.get("resource_status", {}).get("hospital_load", 0.0)
        if float(hosp) > 0.85 and atype == "allocate":
            return "resource_conflict"

    elif agent_id == "police_agent":
        panic = observation.get("resource_status", {}).get("panic_level", 0.0)
        if float(panic) > 60.0:
            return "public_panic"

    elif agent_id == "logistics_agent":
        hosp = observation.get("resource_status", {}).get("hospital_load", 0.0)
        if float(hosp) > 0.9:
            return "resource_conflict"
        fuel = observation.get("resource_status", {}).get("fuel_level", 100.0)
        if float(fuel) < 25.0:
            return "capacity_unknown"

    elif agent_id == "communication_agent":
        # Comm agent needs confirmation before broadcast if low confidence
        messages = observation.get("messages", [])
        unverified = any(
            float(m.get("confidence", m.get("truth_score", 1.0))) < 0.5
            for m in messages[-3:]
        )
        if unverified:
            return "misinformation"

    elif agent_id == "commander_agent":
        casualties = observation.get("visible_events", [])
        critical = any(e.get("type") == "casualty" and int(e.get("severity", 0)) >= 4 for e in casualties)
        if critical:
            return "casualty_critical"

    return None


def should_communicate(
    agent_id: str,
    observation: Dict[str, Any],
    action: Dict[str, Any],
    messages_sent_this_step: int,
    last_sent_step: int,
    current_step: int,
) -> Tuple[bool, Optional[str]]:
    """
    Returns (should_send, reason_string).

    Conditions:
      1. Info gap: confidence < 0.6 OR any uncertainty > 0.5
      2. Support needed: action depends on another agent
      3. Anti-spam: max 2 messages per step; cooldown 1 step
    """
    # Anti-spam gate
    if messages_sent_this_step >= 2:
        return False, None
    if current_step - last_sent_step < 1:
        return False, None

    confidence = _compute_obs_confidence(observation)

    # Gate 1: Information gap
    unc = observation.get("uncertainty", {})
    any_high_unc = any(float(v) > 0.5 for v in unc.values() if isinstance(v, (int, float)))
    info_gap = confidence < 0.6 or any_high_unc

    # Gate 2: Support dependency
    support_reason = _requires_other_agent(agent_id, action, observation)
    needs_support = support_reason is not None

    if not info_gap and not needs_support:
        return False, None

    reason = support_reason or ("info_gap" if info_gap else None)
    return True, reason


def determine_receiver(agent_id: str, reason: str, observation: Dict[str, Any]) -> Optional[str]:
    """Map (agent, reason) → specific target agent. Returns None if no match."""
    key = (agent_id, reason)
    receiver = DEPENDENCY_MAP.get(key)

    # Fallback heuristics
    if receiver is None:
        if "route" in reason or "blocked" in reason:
            receiver = "police_agent"
        elif "capacity" in reason or "resource" in reason:
            receiver = "logistics_agent"
        elif "conflict" in reason or "priority" in reason:
            receiver = "commander_agent"
        elif "panic" in reason or "misinfo" in reason:
            receiver = "communication_agent"
        elif "casualty" in reason or "rescue" in reason:
            receiver = "medical_agent"

    # Avoid self-messaging
    if receiver == agent_id:
        return None

    return receiver


def _build_message_content(
    agent_id: str,
    receiver: str,
    reason: str,
    action: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the structured content body for the message."""
    atype = action.get("action_type", "idle")
    target = action.get("target", [5, 5])
    events = observation.get("visible_events", [])
    casualties = [e for e in events if e.get("type") == "casualty"]

    if reason == "route_unknown":
        blocked = list(observation.get("resource_status", {}).get("local_blocked_roads", []))
        return {
            "request": "Is route to target safe?",
            "target": target,
            "blocked_roads": blocked[:3],
            "action_planned": atype,
        }
    if reason == "needs_escort":
        sev = max((int(c.get("severity", 0)) for c in casualties), default=0)
        return {
            "request": "Requesting police escort to casualty zone.",
            "target": target,
            "casualty_severity": sev,
            "action_planned": atype,
        }
    if reason == "resource_conflict":
        hosp = observation.get("resource_status", {}).get("hospital_load", 0.0)
        return {
            "alert": "Hospital capacity critical.",
            "hospital_load": round(float(hosp), 2),
            "target": target,
        }
    if reason == "public_panic":
        panic = observation.get("resource_status", {}).get("panic_level", 0.0)
        return {
            "alert": "Public panic exceeds threshold. Broadcast advised.",
            "panic_level": round(float(panic), 1),
        }
    if reason == "capacity_unknown":
        return {
            "request": "What is current hospital/shelter capacity?",
            "target": target,
        }
    if reason == "misinformation":
        return {
            "alert": "Unverified information circulating. Requesting confirmation.",
            "target": target,
        }
    if reason == "casualty_critical":
        sev = max((int(c.get("severity", 0)) for c in casualties), default=0)
        return {
            "alert": "Critical casualty detected. Immediate dispatch required.",
            "target": target,
            "severity": sev,
        }
    if reason == "info_gap":
        return {
            "request": "Requesting situational update for my zone.",
            "target": target,
            "confidence": round(_compute_obs_confidence(observation), 2),
        }
    return {"info": reason, "target": target}


def _determine_message_type(reason: str, agent_id: str) -> str:
    """Map reason → message type string."""
    if "alert" in reason or "critical" in reason or "panic" in reason:
        return "alert"
    if agent_id == "commander_agent":
        return "coordination"
    if "request" in reason or "unknown" in reason or "gap" in reason:
        return "request"
    if "advisory" in reason or "route" in reason or "escort" in reason:
        return "advisory"
    return "request"


def _build_human_text(agent_id: str, receiver: str, reason: str, action: Dict[str, Any]) -> str:
    """Generate a human-readable message text for the communication feed."""
    atype = action.get("action_type", "idle")
    target = action.get("target", [5, 5])
    label_map = {
        "medical_agent": "MEDIC",
        "police_agent": "POLICE",
        "logistics_agent": "LOGISTICS",
        "communication_agent": "COMMS",
        "commander_agent": "CMD",
    }
    fr = label_map.get(agent_id, agent_id[:3].upper())
    to = label_map.get(receiver, receiver[:3].upper())

    if reason == "route_unknown":
        return f"{fr}→{to}: Is route to ({target[0]},{target[1]}) safe? Planning {atype.upper()}."
    if reason == "needs_escort":
        return f"{fr}→{to}: Need escort for high-severity casualty at ({target[0]},{target[1]})."
    if reason == "resource_conflict":
        return f"{fr}→{to}: Hospital capacity critical — need resource reallocation."
    if reason == "public_panic":
        return f"{fr}→{to}: Panic level critical. Requesting public broadcast."
    if reason == "capacity_unknown":
        return f"{fr}→{to}: What is current hospital capacity? Allocating to ({target[0]},{target[1]})."
    if reason == "misinformation":
        return f"{fr}→{to}: Unverified info circulating. Need commander confirmation."
    if reason == "casualty_critical":
        return f"{fr}→{to}: CRITICAL casualty at ({target[0]},{target[1]}). Dispatch immediately."
    if reason == "info_gap":
        return f"{fr}→{to}: Low confidence in zone. Requesting situational update."
    return f"{fr}→{to}: Coordination required — {reason}."


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_step_comms(
    agent_actions: Dict[str, Dict[str, Any]],
    observations: Dict[str, Dict[str, Any]],
    agent_last_sent: Dict[str, int],
    agent_sent_count: Dict[str, int],
    current_step: int,
) -> List[Dict[str, Any]]:
    """
    For each agent, decide whether to send a targeted communication this step.

    Returns a list of structured message dicts (max 2 per agent, selective only).
    Each message has the canonical format:
      {
        "message_id": str,
        "from": agent_id,
        "to": agent_id,
        "type": "request|advisory|alert|coordination",
        "reason": str,
        "content": dict,
        "confidence": float,
        "text": str,          ← human-readable, for feed display
        "time_step": int,
        "color": str,         ← hex color for frontend line
        "used": bool,
      }
    """
    messages: List[Dict[str, Any]] = []

    for agent_id, action in agent_actions.items():
        obs = observations.get(agent_id, {})
        sent_this_step = agent_sent_count.get(agent_id, 0)
        last_sent = agent_last_sent.get(agent_id, -99)

        should_send, reason = should_communicate(
            agent_id, obs, action, sent_this_step, last_sent, current_step
        )

        if not should_send or not reason:
            continue

        receiver = determine_receiver(agent_id, reason, obs)
        if not receiver:
            continue

        confidence = _compute_obs_confidence(obs)
        msg_type = _determine_message_type(reason, agent_id)
        content = _build_message_content(agent_id, receiver, reason, action, obs)
        text = _build_human_text(agent_id, receiver, reason, action)

        msg: Dict[str, Any] = {
            "message_id": f"cm_{uuid.uuid4().hex[:8]}",
            "from": agent_id,
            "to": receiver,
            "type": msg_type,
            "reason": reason,
            "content": content,
            "confidence": round(confidence, 3),
            "text": text,
            "time_step": current_step,
            "color": MSG_TYPE_COLOR.get(msg_type, "#ffffff"),
            "used": False,
            "effective": None,   # filled in after step evaluation
        }
        messages.append(msg)

        # Update tracking
        agent_last_sent[agent_id] = current_step
        agent_sent_count[agent_id] = sent_this_step + 1

    return messages


# ─── Post-step outcome evaluation ─────────────────────────────────────────────

def evaluate_comm_outcomes(
    comm_messages: List[Dict[str, Any]],
    per_agent_outcomes: Dict[str, Dict[str, Any]],
    agent_actions: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    After env.step_multi(), evaluate whether each targeted message was useful.

    Returns reward_deltas: {agent_id: float_delta}

    Rules:
      +10  receiver acted in coordination with sender's request → effective
      -10  receiver's outcome failed AND message was relevant → ignored
      -10  message was sent but confidence was high (unnecessary)
      -20  message content was wrong and caused adverse outcome
    """
    reward_deltas: Dict[str, float] = {}

    for msg in comm_messages:
        sender = msg["from"]
        receiver = msg["to"]
        confidence = float(msg.get("confidence", 0.5))
        reason = msg.get("reason", "")
        msg_type = msg.get("type", "request")

        receiver_outcome = per_agent_outcomes.get(receiver, {})
        sender_outcome = per_agent_outcomes.get(sender, {})

        # --- Unnecessary message (confidence was actually high)
        if confidence >= 0.8 and msg_type == "request":
            reward_deltas[sender] = reward_deltas.get(sender, 0.0) - 10.0
            msg["effective"] = False
            continue

        # --- Coordination success
        if receiver_outcome.get("cooperation") or receiver_outcome.get("progress"):
            receiver_action = agent_actions.get(receiver, {})
            req_tgt = msg.get("content", {}).get("target")
            recv_tgt = receiver_action.get("target")
            if req_tgt and recv_tgt and list(req_tgt) == list(recv_tgt):
                # Receiver moved to the target the sender requested
                reward_deltas[sender] = reward_deltas.get(sender, 0.0) + 10.0
                msg["effective"] = True
                continue

        # --- Wrong message caused failure
        if sender_outcome.get("misinformation") and msg_type in {"advisory", "coordination"}:
            reward_deltas[sender] = reward_deltas.get(sender, 0.0) - 20.0
            msg["effective"] = False
            continue

        # --- Useful message ignored (receiver failed despite guidance)
        if not receiver_outcome.get("progress") and msg_type in {"request", "advisory"}:
            if reason in {"route_unknown", "needs_escort", "casualty_critical", "resource_conflict"}:
                reward_deltas[sender] = reward_deltas.get(sender, 0.0) - 10.0
                msg["effective"] = False
                continue

        msg["effective"] = None   # neutral — no clear outcome yet

    return reward_deltas


# ─── Response generation (advisory back-messages) ────────────────────────────

def generate_advisory_responses(
    incoming: List[Dict[str, Any]],
    agent_actions: Dict[str, Dict[str, Any]],
    observations: Dict[str, Dict[str, Any]],
    current_step: int,
) -> List[Dict[str, Any]]:
    """
    Generate response messages from the receiver back to the sender.

    Only generated if the receiver has useful information to share.
    Max 1 response per incoming request.
    """
    responses: List[Dict[str, Any]] = []
    responded: set = set()

    for msg in incoming:
        if msg.get("type") != "request":
            continue
        receiver = msg["to"]
        sender = msg["from"]
        if receiver in responded:
            continue

        obs = observations.get(receiver, {})
        action = agent_actions.get(receiver, {})
        conf = _compute_obs_confidence(obs)

        # Only respond if receiver has high confidence (useful info to share)
        if conf < 0.55:
            continue

        reason = msg.get("reason", "")
        if reason == "route_unknown":
            blocked = list(obs.get("resource_status", {}).get("local_blocked_roads", []))
            secured = list(obs.get("resource_status", {}).get("secured_zones", []))
            tgt = msg.get("content", {}).get("target", [5, 5])
            safe = tuple(tgt) not in {tuple(b) for b in blocked}
            text = (
                f"POLICE→MEDIC: Route to ({tgt[0]},{tgt[1]}) {'CLEAR' if safe else 'BLOCKED'}. "
                f"Secured zones: {secured[:2]}."
            )
            responses.append({
                "message_id": f"rsp_{uuid.uuid4().hex[:6]}",
                "from": receiver,
                "to": sender,
                "type": "advisory",
                "reason": f"response_to_{reason}",
                "content": {"safe": safe, "blocked": blocked[:3], "secured": secured[:2]},
                "confidence": round(conf, 3),
                "text": text,
                "time_step": current_step,
                "color": MSG_TYPE_COLOR["advisory"],
                "used": False,
                "effective": None,
            })
            responded.add(receiver)

        elif reason == "resource_conflict":
            hosp = obs.get("resource_status", {}).get("hospital_load", 0.0)
            fuel = obs.get("resource_status", {}).get("fuel_level", 100.0)
            text = (
                f"LOGISTICS→MEDIC: Hospital at {round(float(hosp)*100)}%. "
                f"Fuel {round(float(fuel))}%. Rerouting advised."
            )
            responses.append({
                "message_id": f"rsp_{uuid.uuid4().hex[:6]}",
                "from": receiver,
                "to": sender,
                "type": "advisory",
                "reason": "resource_status_update",
                "content": {"hospital_load": hosp, "fuel_level": fuel},
                "confidence": round(conf, 3),
                "text": text,
                "time_step": current_step,
                "color": MSG_TYPE_COLOR["advisory"],
                "used": False,
                "effective": None,
            })
            responded.add(receiver)

    return responses
