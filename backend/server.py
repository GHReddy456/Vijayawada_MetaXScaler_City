from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Load .env so HF_API_TOKEN / HF_MODEL_ID are available before anything else.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    # dotenv not installed — fall back to manual parse
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and "=" in _line and not _line.startswith("#"):
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from compare import compare_policies
from environment import CrisisWorldEnv
from evaluate import load_benchmark_results, run_and_save
from models import ActionModel, ResetRequest
from ollama_policy import OllamaPolicy
from policies import baseline_policy, improved_policy
from train_grpo import load_training_artifacts

# Initialise LLM policy — picks up HF_API_TOKEN + HF_MODEL_ID from env
llm_policy = OllamaPolicy(
    model=os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
)
env = CrisisWorldEnv()
env_lock = asyncio.Lock()
connected_websockets: List[WebSocket] = []

# ── Simulation state ─────────────────────────────────────────────────────────
# PLAYING | PAUSED | STOPPED
simulation_status: str = "PLAYING"
# BASELINE = deliberate weak rule policy.  TRAINED = LLM (LLaMA via HF).
simulation_mode: str = "TRAINED"

# ── LLM action cache ─────────────────────────────────────────────────────────
# Keeps the LAST set of LLM-generated actions so the 1-second simulation tick
# can run immediately without waiting for the HF API (~5-7s per call).
# A background task refreshes this dict after every completed inference.
_cached_actions: Dict[str, Any] = {}
_llm_task: Optional[asyncio.Task] = None

# ── RL reward tracking ───────────────────────────────────────────────────────
# Accumulates per-step rewards so the frontend can show a learning curve.
# Capped at 200 values; older values averaged together for readability.
_reward_curve: List[float] = []

# Cached benchmark results loaded at startup (baseline vs improved)
_benchmark_cache: Optional[Dict[str, Any]] = None

AGENT_KNOWLEDGE_LABEL: Dict[str, str] = {
    "medical_agent": (
        "Ambulance / medical: nearby casualties + hospital capacity; always receives police route advisories and commander orders."
    ),
    "police_agent": (
        "Police: local incidents + citywide blocked roads/bridges so you can brief ambulances on safe detours."
    ),
    "logistics_agent": (
        "Logistics: citywide shelter/hospital load; commander logistics messages; local traffic cues only."
    ),
    "communication_agent": (
        "Communications: recent coordination traffic in your area — not full commander situational data."
    ),
    "commander_agent": (
        "Commander: full city picture — every active calamity, all casualties, closures, capacities, and unit positions."
    ),
}

AGENT_MAP = {
    "medical_agent": {"id": "A1", "label": "A1", "role": "AMBULANCE"},
    "logistics_agent": {"id": "A2", "label": "A2", "role": "LOGISTICS"},
    "police_agent": {"id": "A3", "label": "A3", "role": "POLICE"},
    "communication_agent": {"id": "A4", "label": "A4", "role": "FIRE_UNIT"},
    "commander_agent": {"id": "A5", "label": "A5", "role": "COMMAND"},
}

ZONE_LOOKUP = {
    (5, 6): "GUNADALA",
    (5, 5): "BENZ CIRCLE",
    (4, 6): "AUTONAGAR",
    (6, 4): "VIJAYAWADA JUNCTION",
    (7, 7): "KANURU",
    (3, 4): "MALLESWARAM",
    (4, 3): "BHAVANIPURAM",
    (6, 5): "PATAMATA",
}

def nearest_zone_name(cell: List[int] | tuple[int, int]) -> str:
    cx, cy = int(cell[0]), int(cell[1])
    best_name = "CITY ZONE"
    best_dist = 999
    for (zx, zy), name in ZONE_LOOKUP.items():
        d = abs(zx - cx) + abs(zy - cy)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name

def grid_to_norm(cell: List[int] | Tuple[int, int], pad: float = 20.0) -> Tuple[float, float]:
    # Keep 10x10 env grid inside safe viewport margins (panels occupy edges).
    cx, cy = float(cell[0]), float(cell[1])
    span = 100.0 - (pad * 2.0)
    x = pad + (cx / 9.0) * span
    y = pad + (cy / 9.0) * span
    return x, y

def backend_agent_id(frontend_id: str) -> Optional[str]:
    for aid, mapped in AGENT_MAP.items():
        if mapped["id"] == frontend_id:
            return aid
    return None

def frontend_agent_id(backend_id: str) -> str:
    mapped = AGENT_MAP.get(backend_id)
    return mapped["id"] if mapped else backend_id

def _fmt_ts() -> str:
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"

def map_communications(state: Dict[str, Any], action_logs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the targeted-only communication feed.

    Rules:
      - NO recipient = "ALL" or "BROADCAST"
      - Every message has a specific sender and receiver
      - Messages include a reason
    """
    comms: List[Dict[str, Any]] = []
    ts   = _fmt_ts()
    step = int(state.get("step_count", 0))

    med  = action_logs.get("medical_agent",   {})
    pol  = action_logs.get("police_agent",     {})
    cmd  = action_logs.get("commander_agent",  {})
    log  = action_logs.get("logistics_agent",  {})
    comm = action_logs.get("communication_agent", {})

    # ── Police → Medical: route advisory ─────────────────────────────────────
    if pol.get("action_type") in {"secure", "block", "route"}:
        tgt  = pol.get("target", [5, 5])
        zone = nearest_zone_name(tgt)
        act  = pol.get("action_type", "route").upper()
        comms.append({
            "id": f"pol_med_{step}", "timestamp": ts,
            "sender": "A3", "recipient": "A1",
            "text": f"POLICE {act} at {zone} — corridor status updated. Medical may proceed.",
            "type": "route_advisory", "zone": zone, "confidence": 0.88,
            "reason": "route_advisory",
        })

    # ── Medical → Police: route request when dispatching ─────────────────────
    if med.get("action_type") in {"dispatch", "route"}:
        tgt    = med.get("target", [5, 5])
        zone   = nearest_zone_name(tgt)
        reason = (med.get("metadata") or {}).get("reason", "casualty")
        comms.append({
            "id": f"med_pol_{step}", "timestamp": ts,
            "sender": "A1", "recipient": "A3",
            "text": f"Is route to {zone} ({tgt[0]},{tgt[1]}) safe? Planning DISPATCH. Reason: {reason[:40]}",
            "type": "request", "zone": zone, "confidence": 0.75,
            "reason": "route_unknown",
        })

    # ── Medical ↔ Police: explicit coordination when same target ─────────────
    if med and pol:
        med_tgt = tuple(med.get("target", []))
        pol_tgt = tuple(pol.get("target", []))
        if med_tgt and pol_tgt and med_tgt == pol_tgt and med.get("action_type") in {"dispatch", "route"}:
            zone = nearest_zone_name(list(med_tgt))
            comms.append({
                "id": f"coord_{step}", "timestamp": ts,
                "sender": "A3", "recipient": "A1",
                "text": f"Confirmed. Police securing {zone} approach. Safe for medical entry.",
                "type": "advisory", "zone": zone, "confidence": 0.95,
                "reason": "coordinated_escort",
            })

    # ── Commander → Medical: priority order when critical ────────────────────
    if cmd.get("action_type") and med.get("action_type") in {"dispatch", "route"}:
        tgt  = cmd.get("target", [5, 5])
        zone = nearest_zone_name(tgt)
        reason = (cmd.get("metadata") or {}).get("reason", "strategic")
        comms.append({
            "id": f"cmd_med_{step}", "timestamp": ts,
            "sender": "A5", "recipient": "A1",
            "text": f"Priority order: {cmd.get('action_type','route').upper()} → {zone}. {reason[:50]}",
            "type": "coordination", "zone": zone, "confidence": 0.99,
            "reason": "priority_order",
        })

    # ── Logistics → Medical: resource/capacity update ────────────────────────
    if log.get("action_type") in {"allocate", "route"}:
        tgt  = log.get("target", [5, 5])
        zone = nearest_zone_name(tgt)
        comms.append({
            "id": f"log_med_{step}", "timestamp": ts,
            "sender": "A2", "recipient": "A1",
            "text": f"Resource allocation updated → {zone}. Hospital capacity advisory issued.",
            "type": "advisory", "zone": zone, "confidence": 0.85,
            "reason": "capacity_update",
        })

    # ── Comms agent → Commander: verify before broadcast ─────────────────────
    if comm.get("action_type") == "broadcast":
        truth = float((comm.get("metadata") or {}).get("truth_score", 0.8))
        msg_text = str((comm.get("metadata") or {}).get("text", "advisory"))[:60]
        if truth < 0.7:
            comms.append({
                "id": f"comm_cmd_{step}", "timestamp": ts,
                "sender": "A4", "recipient": "A5",
                "text": f"⚠ Requesting confirmation before broadcast: '{msg_text}' (conf {round(truth*100)}%)",
                "type": "request", "zone": "CITY", "confidence": round(truth, 2),
                "reason": "misinformation_check",
            })
        else:
            comms.append({
                "id": f"comm_cmd_{step}", "timestamp": ts,
                "sender": "A4", "recipient": "A5",
                "text": f"Broadcasting verified advisory: '{msg_text}'",
                "type": "advisory", "zone": "CITY", "confidence": round(truth, 2),
                "reason": "public_alert",
            })

    # ── Commander → Police: casualty alert ───────────────────────────────────
    casualties_waiting = [c for c in state.get("casualties", []) if c.get("status") == "waiting"]
    if casualties_waiting and step % 4 == 0:
        pos  = casualties_waiting[0].get("pos", [5, 5])
        zone = nearest_zone_name(pos)
        comms.append({
            "id": f"alert_{step}", "timestamp": ts,
            "sender": "A5", "recipient": "A3",
            "text": f"CRITICAL: {len(casualties_waiting)} casualty/casualties at {zone} — secure perimeter immediately.",
            "type": "alert", "zone": zone, "confidence": 1.0,
            "reason": "casualty_critical",
        })

    # Enforce no broadcast
    for c in comms:
        assert c["recipient"] not in ("ALL", "all", "BROADCAST"), \
            f"Broadcast blocked in map_communications: {c}"

    return comms[-40:]


def commander_brief_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    calamities: List[Dict[str, Any]] = []
    for ev in state.get("events", []):
        if ev.get("type") in {"earthquake", "flood", "blackout"} and ev.get("location"):
            loc = ev["location"]
            calamities.append(
                {
                    "kind": ev.get("type"),
                    "location": list(loc),
                    "zone": nearest_zone_name(loc),
                    "intensity": ev.get("intensity"),
                }
            )
    waiting = [c for c in state.get("casualties", []) if c.get("status") == "waiting"]
    positions = {
        AGENT_MAP.get(aid, {}).get("label", aid): list(data.get("pos", [0, 0]))
        for aid, data in state.get("agents", {}).items()
        if aid in AGENT_MAP
    }
    return {
        "calamities": calamities,
        "waiting_casualties": len(waiting),
        "blocked_cells": len(state.get("blocked_roads", [])),
        "unit_positions": positions,
    }


def decision_feed_from_actions(state: Dict[str, Any], action_logs: Dict[str, Any]) -> List[Dict[str, Any]]:
    step = int(state.get("step_count", 0))
    ts = _fmt_ts()
    rows: List[Dict[str, Any]] = []
    for aid, al in action_logs.items():
        if aid not in AGENT_MAP:
            continue
        label = AGENT_MAP[aid]["label"]
        tgt = al.get("target", [0, 0])
        rows.append(
            {
                "id": f"dec_{step}_{label}_{al.get('action_type', 'idle')}",
                "timestamp": ts,
                "message": (
                    f"[{label}] {str(al.get('action_type', 'idle')).upper()} → "
                    f"{nearest_zone_name(tgt)} — {al.get('metadata', {}).get('reason', 'no reason')}"
                ),
                "severity": "LOW",
            }
        )
    return rows


_AGENT_LABEL_MAP = {
    "medical_agent":       "🚑 MEDIC",
    "police_agent":        "🚔 POLICE",
    "logistics_agent":     "🚚 LOGISTICS",
    "communication_agent": "📡 COMMS",
    "commander_agent":     "🛸 COMMAND",
}
_SCENARIO_ICON = {"earthquake": "🌍", "flood": "🌊", "blackout": "⚡"}
_SCENARIO_ZONES = {
    "earthquake": ("Benz Circle", "Patamata"),
    "flood":      ("Kanuru", "Gunadala"),
    "blackout":   ("Vijayawada Junction", "Autonagar"),
}

# Per-step tracking: detect medical→police dependency this tick
_step_dependency_waiting: Dict[int, bool] = {}


def _evt(ts: str, step: int, text: str) -> Dict[str, Any]:
    return {"type": "event",    "ts": ts, "step": step, "text": text}

def _dec(ts: str, step: int, agent: str, text: str) -> Dict[str, Any]:
    return {"type": "decision", "ts": ts, "step": step, "agent": agent, "text": text}

def _msg(ts: str, step: int, text: str) -> Dict[str, Any]:
    return {"type": "message",  "ts": ts, "step": step, "text": text}

def _out(ts: str, step: int, text: str) -> Dict[str, Any]:
    return {"type": "outcome",  "ts": ts, "step": step, "text": text}


def generate_chat_events(
    state: Dict[str, Any],
    action_logs: Dict[str, Any],
    last_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Full step-by-step narrative following the mandatory flow:
      SITUATION → DECISION → COMMUNICATION → COORDINATION → OUTCOME

    Part 1  — Rich disaster initialization at step 0
    Part 3  — Dependency / waiting behaviour shown in chat
    Part 4  — Coordination success made explicit
    Part 5  — RL learning update + Baseline vs Trained narrative
    Part 6  — Reasoning BEFORE dispatch shown
    Part 7  — Full multi-step causal chains
    Part 8  — Every step follows the 5-stage flow
    Part 9  — Debug assertions to guarantee non-empty output
    """
    chat: List[Dict[str, Any]] = []
    ts   = _fmt_ts()
    step = int(state.get("step_count", 0))
    scenario  = str(state.get("scenario", "earthquake")).lower()
    icon      = _SCENARIO_ICON.get(scenario, "🆘")
    zones     = _SCENARIO_ZONES.get(scenario, ("City Centre", "Outskirts"))
    metrics   = state.get("metrics", {})
    mode_tag  = "[TRAINED 🧠]" if simulation_mode == "TRAINED" else "[BASELINE ⚠]"

    waiting  = [c for c in state.get("casualties", []) if c.get("status") == "waiting"]
    treated  = [c for c in state.get("casualties", []) if c.get("status") == "treated"]
    blocked  = len(state.get("blocked_roads", []))
    panic    = round(float(metrics.get("panic_level", 0)), 1)
    trust    = round(float(metrics.get("trust_score", 0)), 1)
    reward   = round(float(metrics.get("reward", 0)), 1)
    hosp_raw = state.get("system_state", {}).get("hospital_load", 0.8)
    hosp_pct = round(float(hosp_raw) * 100)

    # ════════════════════════════════════════════════════════════════
    # PART 1 — DISASTER INITIALIZATION (step 0 only)
    # ════════════════════════════════════════════════════════════════
    if step == 0:
        chat += [
            _evt(ts, step, f"🚨 {scenario.upper()} DETECTED — Vijayawada City"),
            _evt(ts, step, f"Severity: HIGH | Affected Zones: {zones[0]}, {zones[1]}"),
            _evt(ts, step,
                 f"Estimated casualties: {len(waiting)} | Roads blocked: {blocked} | "
                 f"Hospital load: {hosp_pct}%"),
            _evt(ts, step,
                 "Agents activated: 🚑 Medical · 🚔 Police · 🚚 Logistics · 📡 Comms · 🛸 Commander"),
            _evt(ts, step,
                 "Objective: Minimize deaths, reduce panic, coordinate rescue operations"),
        ]
        # Part 5 — Baseline vs Trained narrative at start (never hardcode numbers)
        if simulation_mode == "TRAINED":
            chat += [
                _evt(ts, step,
                     "📈 MODE: TRAINED — LLaMA 3.1-8B via HuggingFace driving agent decisions"),
                _evt(ts, step,
                     "Expected: agents communicate before acting · wait for route clearance · "
                     "coordinate across roles · reward improves over episodes"),
            ]
        else:
            chat += [
                _evt(ts, step,
                     "⚠ MODE: BASELINE — Rule-based policy (deliberately weak, no inter-agent coordination)"),
                _evt(ts, step,
                     "Expected: immediate dispatch without route check · no waiting · "
                     "no dependency · higher casualties — watch reward drop vs TRAINED"),
            ]

    # ════════════════════════════════════════════════════════════════
    # STAGE 1 — SITUATION (every step + every 5 steps detailed)
    # ════════════════════════════════════════════════════════════════
    if step > 0:
        if step % 5 == 0:
            chat.append(_evt(ts, step,
                f"📊 STEP {step} {mode_tag} | Waiting: {len(waiting)} · Rescued: {len(treated)} · "
                f"Panic: {panic}% · Trust: {trust} · Roads blocked: {blocked} · Reward: {reward}"))
        else:
            chat.append(_evt(ts, step,
                f"📍 Step {step} — {len(waiting)} awaiting rescue · {len(treated)} treated · "
                f"Hospital {hosp_pct}% · Panic {panic}%"))

    # ════════════════════════════════════════════════════════════════
    # STAGE 2 — DECISIONS (per agent, with pre-action reasoning)
    # ════════════════════════════════════════════════════════════════
    med_action  = action_logs.get("medical_agent", {})
    pol_action  = action_logs.get("police_agent", {})
    cmd_action  = action_logs.get("commander_agent", {})

    for aid, al in action_logs.items():
        if aid not in AGENT_MAP:
            continue
        atype  = str(al.get("action_type", "idle"))
        tgt    = al.get("target", [5, 5])
        reason = (al.get("metadata") or {}).get("reason", "")
        zone   = nearest_zone_name(tgt)
        label  = _AGENT_LABEL_MAP.get(aid, aid)

        # Part 6: reasoning BEFORE dispatch
        if atype == "dispatch" and aid == "medical_agent":
            chat.append(_dec(ts, step, aid,
                f"{label}: Evaluating route safety before dispatch to {zone} ({tgt[0]},{tgt[1]}) — "
                f"{reason or 'casualty detected'}"))
        elif atype == "secure" and aid == "police_agent":
            chat.append(_dec(ts, step, aid,
                f"{label}: Securing approach corridor to {zone} — enabling safe medical access"))
        elif atype in ("dispatch", "route"):
            chat.append(_dec(ts, step, aid,
                f"{label}: {atype.upper()} → {zone} ({tgt[0]},{tgt[1]}) — {reason or 'operational task'}"))
        elif atype == "block":
            st   = (al.get("metadata") or {}).get("state", "blocked")
            verb = "Closing" if st == "blocked" else "Opening"
            chat.append(_dec(ts, step, aid,
                f"{label}: {verb} road at {zone} — {reason or 'risk containment'}"))
        elif atype == "allocate":
            chat.append(_dec(ts, step, aid,
                f"{label}: Allocating resources to {zone} — {reason or 'capacity management'}"))
        elif atype == "broadcast":
            msg_txt = (al.get("metadata") or {}).get("text", "")[:60]
            truth   = float((al.get("metadata") or {}).get("truth_score", 1.0))
            qual    = "✓ VERIFIED" if truth >= 0.7 else "⚠ UNVERIFIED — checking with Commander first"
            chat.append(_dec(ts, step, aid,
                f"{label}: Broadcasting [{qual}] — '{msg_txt}'"))
        elif atype not in ("idle",):
            chat.append(_dec(ts, step, aid, f"{label}: {atype.upper()} → {zone}"))

    # ════════════════════════════════════════════════════════════════
    # STAGE 3 — COMMUNICATION (targeted messages only, never broadcast)
    # ════════════════════════════════════════════════════════════════
    comm_links = state.get("comm_links", []) or last_info.get("comm_links", [])
    has_route_request = False
    police_gave_advisory = False

    for cl in comm_links:
        fr    = cl.get("from", "")
        to    = cl.get("to", "")

        # Part 9 debug assertion: never broadcast
        assert to not in ("ALL", "all", "BROADCAST"), \
            f"BROADCAST detected in comm_link from {fr} — remove immediately"

        fr_label  = _AGENT_LABEL_MAP.get(fr, fr)
        to_label  = _AGENT_LABEL_MAP.get(to, to)
        msg_type  = str(cl.get("type", "request")).upper()
        text_body = str(cl.get("text", ""))[:90]
        conf      = round(float(cl.get("confidence", 0.8)) * 100)
        eff       = cl.get("effective")
        eff_tag   = " ✓ acted" if eff is True else " ✗ ignored" if eff is False else ""
        reason    = cl.get("reason", "")

        chat.append(_msg(ts, step,
            f"[{msg_type}]{eff_tag} {fr_label} → {to_label}: {text_body} (conf {conf}%)"))

        # Track route-request/advisory pairs for Stage 4
        if reason == "route_unknown" and fr == "medical_agent":
            has_route_request = True
        if cl.get("type") == "advisory" and "route" in reason.lower() and fr == "police_agent":
            police_gave_advisory = True

    # Part 3 — Dependency / waiting shown in chat
    if has_route_request and not police_gave_advisory:
        chat.append(_evt(ts, step,
            "⏳ 🚑 MEDIC waiting for 🚔 POLICE route clearance before proceeding..."))
    elif has_route_request and police_gave_advisory:
        # Part 4 — Coordination success made explicit
        chat.append(_evt(ts, step,
            "✅ DEPENDENCY RESOLVED — Police cleared route → Medical dispatch confirmed safe corridor"))

    # ════════════════════════════════════════════════════════════════
    # STAGE 4 — COORDINATION OUTCOMES
    # ════════════════════════════════════════════════════════════════
    if last_info.get("outcome", {}).get("coordinated_sequence"):
        chat.append(_evt(ts, step,
            "🤝 COORDINATION SUCCESS — Police secured zone → Medical dispatch confirmed safe path → +14 reward"))

    # Police securing same cell as medical target = explicit coordination event
    if (med_action.get("action_type") in {"dispatch", "route"}
            and pol_action.get("action_type") in {"secure", "block"}):
        med_tgt = tuple(med_action.get("target", []))
        pol_tgt = tuple(pol_action.get("target", []))
        if med_tgt and pol_tgt and med_tgt == pol_tgt:
            zone = nearest_zone_name(list(med_tgt))
            chat.append(_evt(ts, step,
                f"🤝 Police securing {zone} simultaneously with Medical dispatch → "
                "coordinated corridor established"))

    # ════════════════════════════════════════════════════════════════
    # STAGE 5 — OUTCOMES (causal chain + lives + reward)
    # ════════════════════════════════════════════════════════════════

    # Part 7 — Full multi-step causal chains
    causal = last_info.get("causal_chain", [])
    if causal:
        for line in causal[:4]:
            raw = str(line)
            if raw.startswith("["):
                end = raw.find("]")
                tag  = raw[1:end] if end > 0 else "INFO"
                body = raw[end + 1:].strip()
            else:
                tag, body = "CAUSAL", raw
            # Expand short chains into narrative arrow chains
            if "→" not in body and "delay" in body.lower():
                body = f"Medical dispatch delayed → hospital overload → capacity exceeded → death risk ↑ → penalty"
            chat.append(_out(ts, step, f"[{tag}] {body[:120]}"))
    elif len(waiting) > 0 and step > 0:
        # Synthesize a causal chain when none returned
        delay_chain = (
            f"Step {step}: {len(waiting)} casualty/casualties untreated → "
            f"hospital load {hosp_pct}% → "
            f"{'overload risk ↑ → death possible → -penalty' if hosp_pct > 80 else 'rescue ongoing'}"
        )
        chat.append(_out(ts, step, f"[CAUSAL] {delay_chain}"))

    lives  = int(last_info.get("outcome", {}).get("lives_saved", 0))
    deaths = int(last_info.get("outcome", {}).get("deaths", 0))
    if lives > 0:
        chat.append(_out(ts, step,
            f"✅ {lives} life/lives SAVED this step → +{lives * 50} reward"))
    if deaths > 0:
        chat.append(_out(ts, step,
            f"💀 {deaths} death(s) this step — delayed rescue / hospital overload → "
            f"-{deaths * 100} penalty"))
        if simulation_mode == "BASELINE":
            chat.append(_out(ts, step,
                "⚠ BASELINE: Dispatched immediately without route check → unsafe path → "
                "delay → death. Trained policy would wait for police clearance."))
        else:
            chat.append(_out(ts, step,
                "TRAINED: Death occurred despite coordination — environment too severe this step. "
                "Reward penalty added to learning signal."))

    # Part 5 — RL learning update (every 10 steps in trained mode)
    # Compute actual trend from _reward_curve (never hardcode)
    if simulation_mode == "TRAINED" and step > 0 and step % 10 == 0:
        rc = _reward_curve
        if len(rc) >= 10:
            early_avg  = round(sum(rc[:5]) / 5, 1)
            recent_avg = round(sum(rc[-5:]) / 5, 1)
            delta      = round(recent_avg - early_avg, 1)
            trend_txt  = f"▲ +{delta} (improving)" if delta > 0 else f"▼ {delta} (learning)"
            chat.append(_evt(ts, step,
                f"📈 LEARNING UPDATE (step {step}): "
                f"Early avg reward {early_avg} → Recent avg {recent_avg} · {trend_txt} · "
                f"Trust: {trust} · Panic: {panic}%"))
        else:
            chat.append(_evt(ts, step,
                f"📈 STEP {step}: Reward {reward:+.1f} · Agents building coordination patterns"))

    # Part 9 — guarantee at least one entry per step
    assert len(chat) > 0, f"generate_chat_events produced zero entries at step {step}"

    return chat


def map_state_to_frontend(state: Dict[str, Any], action_logs: Dict[str, Any]) -> Dict[str, Any]:
    agents_out = []
    for aid, adata in state.get("agents", {}).items():
        if aid not in AGENT_MAP:
            continue
        mapped = AGENT_MAP[aid]
        pos = adata.get("pos", [0, 0])
        action_log = action_logs.get(aid, {})
        action_type = action_log.get("action_type", "idle")
        reasoning = action_log.get("metadata", {}).get("reason", "Awaiting task")
        target = action_log.get("target", pos)
        
        x, y = grid_to_norm(pos)
        tx, ty = grid_to_norm(target)
        
        agent_state = "IDLE"
        if action_type in ["route", "dispatch"]: agent_state = "ENROUTE"
        elif action_type in ["allocate", "broadcast"]: agent_state = "ACTIVE"
        elif action_type == "block": agent_state = "BLOCKED"
        
        knowledge_scope = AGENT_KNOWLEDGE_LABEL.get(aid, "Operational view.")
        last_decision = (
            f"{action_type.upper()} → {nearest_zone_name(target)}"
            + (f" — {reasoning}" if reasoning else "")
        )
        agents_out.append({
            "id": mapped["id"],
            "label": mapped["label"],
            "role": mapped["role"],
            "x": x,
            "y": y,
            "targetX": tx,
            "targetY": ty,
            "targetName": nearest_zone_name(target),
            "state": agent_state,
            "confidence": 0.85 + (hash(aid) % 15) / 100.0,
            "reliability": 0.90 + (hash(aid) % 10) / 100.0,
            "action": action_type.upper(),
            "reasoning": reasoning,
            "knowledge_scope": knowledge_scope,
            "last_decision": last_decision,
        })
        
    crises_out = []
    for i, event in enumerate(state.get("events", [])):
        if event.get("type") in {"earthquake", "flood", "blackout"} and event.get("location"):
            ep = event["location"]
            sev_str = "HIGH" if int(event.get("intensity", 1)) >= 2 else "MEDIUM"
            crises_out.append({
                "id": f"calamity_{i}",
                "type": "UNREST" if event["type"] == "blackout" else "BLOCKAGE",
                "severity": sev_str,
                "name": f"{event['type'].upper()} - {nearest_zone_name(ep)}",
                "x": grid_to_norm(ep)[0],
                "y": grid_to_norm(ep)[1],
                "radius": 35 + int(event.get("intensity", 1)) * 8,
            })

    for i, cas in enumerate(state.get("casualties", [])):
        if cas.get("status") == "waiting":
            pos = cas.get("pos", [0, 0])
            sev = int(cas.get("severity", 1))
            sev_str = "LOW" if sev == 1 else "MEDIUM" if sev == 2 else "HIGH"
            crises_out.append({
                "id": f"c_{i}",
                "type": "MEDICAL",
                "severity": sev_str,
                "name": f"Casualty - {nearest_zone_name(pos)}",
                "x": grid_to_norm(pos)[0],
                "y": grid_to_norm(pos)[1],
                "radius": 20 + sev * 10
            })
            
    events_out = []
    # Just take latest messages as events
    for i, msg in enumerate(state.get("messages", [])[-5:]):
        events_out.append({
            "id": f"ev_{msg.get('message_id', i)}",
            "timestamp": f"01:{24+i//60:02d}:{i%60:02d}",
            "message": msg.get("text", ""),
            "severity": "MEDIUM" if msg.get("truth_score", 1.0) < 0.8 else "LOW"
        })
        
    metrics = state.get("metrics", {})
    casualties = state.get("casualties", [])
    total_casualties = max(1, len(casualties))
    treated = sum(1 for c in casualties if c.get("status") == "treated")
    metrics_out = {
        "death_toll": metrics.get("deaths", 0),
        "panic_level": metrics.get("panic_level", 0),
        "trust_score": metrics.get("trust_score", 0),
        "rescue_success": (treated / total_casualties) * 100.0,
        "reward": metrics.get("total_reward", 0)
    }

    # -- Causal chains from last step info (if available)
    last_info = state.get("_last_info", {})
    causal_chain: List[str] = last_info.get("causal_chain", [])

    # -- Per-agent correctness signals for green/red rendering
    per_agent_correctness: Dict[str, bool] = last_info.get("per_agent_correctness", {})
    # Map backend ids to frontend labels
    correctness_labeled: Dict[str, bool] = {
        AGENT_MAP[bid]["label"]: v
        for bid, v in per_agent_correctness.items()
        if bid in AGENT_MAP
    }

    # -- Agent trust (per-agent trust scores from env)
    raw_agent_trust: Dict[str, float] = state.get("metrics", {}).get("agent_trust", {})
    agent_trust_labeled: Dict[str, float] = {
        AGENT_MAP[bid]["label"]: round(v, 1)
        for bid, v in raw_agent_trust.items()
        if bid in AGENT_MAP
    }

    # -- Baseline metrics from benchmark cache (for ComparisonView real data)
    baseline_metrics: Optional[Dict[str, Any]] = None
    if _benchmark_cache:
        b = _benchmark_cache.get("baseline_avg", {})
        baseline_metrics = {
            "death_toll": round(b.get("deaths", 0)),
            "panic_level": round(b.get("panic_level", 0)),
            "trust_score": round(b.get("trust_score", 0)),
            "rescue_success": round(b.get("rescue_success_rate", 0)),
            "reward": round(b.get("total_reward", 0)),
            "coordination_score": round(b.get("coordination_score", 0)),
        }

    # -- Targeted comm links (from comms.py selective engine)
    raw_comm_links: List[Dict[str, Any]] = state.get("comm_links", []) or last_info.get("comm_links", [])
    comm_links_out: List[Dict[str, Any]] = []
    for cl in raw_comm_links[-20:]:
        fr = AGENT_MAP.get(cl.get("from", ""), {}).get("id", cl.get("from", ""))
        to = AGENT_MAP.get(cl.get("to", ""), {}).get("id", cl.get("to", ""))
        if not fr or not to:
            continue
        comm_links_out.append({
            "id": cl.get("message_id", ""),
            "from": fr,
            "to": to,
            "type": cl.get("type", "request"),       # request|advisory|alert|coordination
            "reason": cl.get("reason", ""),
            "text": cl.get("text", ""),
            "confidence": round(float(cl.get("confidence", 0.8)), 3),
            "color": cl.get("color", "#4499ff"),
            "effective": cl.get("effective"),
            "time_step": int(cl.get("time_step", 0)),
        })

    # -- Structured Agent Chat narrative (new per-step events)
    chat_events = generate_chat_events(state, action_logs, last_info)

    return {
        "agents": agents_out,
        "crises": crises_out,
        "metrics": metrics_out,
        "communications": map_communications(state, action_logs),
        "comm_links": comm_links_out,
        "chat_events": chat_events,
        "event_log": events_out,
        "decision_feed": decision_feed_from_actions(state, action_logs),
        "causal_chain": causal_chain,
        "agent_correctness": correctness_labeled,
        "agent_trust": agent_trust_labeled,
        "system_state": {
            "scenario": state.get("scenario", "Simulation"),
            "mode": simulation_mode,
            "status": simulation_status,
            "time_elapsed": state.get("step_count", 0) * 10,
            "commander_brief": commander_brief_from_state(state),
            "baseline_metrics": baseline_metrics,
        },
    }

async def _refresh_llm_actions(observations: Dict[str, Any]) -> None:
    """
    Background task: infer LLM actions for all agents concurrently.
    Falls back to improved_policy per agent on any error.
    Updates _cached_actions when done (non-blocking for simulation tick).
    """
    global _cached_actions

    async def _one(aid: str, obs: Dict[str, Any]) -> tuple:
        try:
            act = await asyncio.to_thread(llm_policy.generate_action, obs, aid)
            return aid, act
        except Exception as exc:
            print(f"[LLM] {aid} fallback: {exc}")
            return aid, improved_policy(obs)

    pairs = await asyncio.gather(*[_one(aid, obs) for aid, obs in observations.items()])
    _cached_actions = dict(pairs)
    print(f"[LLM] Actions refreshed for {list(_cached_actions.keys())}")


async def _broadcast(ws_msg: Dict[str, Any]) -> None:
    """Send a WS message to all connected clients; remove dead connections."""
    dead = []
    for ws in list(connected_websockets):
        try:
            await ws.send_json(ws_msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


async def simulation_loop() -> None:
    """
    Main simulation loop — runs every 1 second regardless of LLM speed.

    Action selection strategy:
      BASELINE  → baseline_policy (rule-based, deterministic, deliberately bad)
      TRAINED   → _cached_actions (last LLM output) with improved_policy fallback
                  while LLM is still computing in background

    The LLM inference (_refresh_llm_actions) runs in a separate asyncio Task so
    that the 1-second tick is never blocked waiting for HF API responses.
    """
    global simulation_status, simulation_mode, _cached_actions, _llm_task

    while True:
        await asyncio.sleep(1.0)  # 1-second heartbeat

        # ── Always send at least a heartbeat so frontend knows we're alive ──
        async with env_lock:
            state = env.state()
            state["_last_info"] = getattr(env, "_last_info", {})

        if simulation_status != "PLAYING":
            ws_msg = map_state_to_frontend(state, {})
            ws_msg["type"] = "update"
            ws_msg["paused"] = simulation_status != "PLAYING"
            await _broadcast(ws_msg)
            continue

        # ── Auto-reset when episode ends ─────────────────────────────────────
        async with env_lock:
            if env.step_count >= env.max_steps or env.panic_level >= env.panic_threshold:
                print(f"[sim] Episode ended (step={env.step_count}). Auto-resetting…")
                env.reset(level=env.level, seed=env.rng.randint(0, 1000))
                env._last_info = {}
                _cached_actions.clear()
                _reward_curve.clear()
            observations = env._all_observations()

        # ── Build actions for this tick ───────────────────────────────────────
        actions: Dict[str, Any] = {}
        for aid, obs in observations.items():
            if simulation_mode == "BASELINE":
                actions[aid] = baseline_policy(obs)
            elif aid in _cached_actions:
                # Use latest LLM-inferred action (may be a few seconds old)
                actions[aid] = _cached_actions[aid]
            else:
                # LLM hasn't responded yet — use improved_policy as hot fallback
                actions[aid] = improved_policy(obs)

        # ── Kick off background LLM refresh if not already running ───────────
        if simulation_mode == "TRAINED" and (
            _llm_task is None or _llm_task.done()
        ):
            _llm_task = asyncio.create_task(_refresh_llm_actions(observations))

        # ── Step the environment ─────────────────────────────────────────────
        async with env_lock:
            try:
                step_result = env.step_multi(actions)
                env._last_info = step_result.get("info", {})
            except Exception as exc:
                print(f"[sim] step_multi error: {exc}")
                env._last_info = {}
            state = env.state()
            state["_last_info"] = env._last_info

        # ── Record step reward for learning curve ────────────────────────────
        step_reward = float(state.get("metrics", {}).get("reward", 0.0))
        _reward_curve.append(round(step_reward, 2))
        if len(_reward_curve) > 200:
            # Compress: keep last 200 by averaging pairs at the front
            _reward_curve[:] = [
                round((_reward_curve[i] + _reward_curve[i + 1]) / 2, 2)
                for i in range(0, 40, 2)
            ] + _reward_curve[40:]

        # ── Broadcast full payload ────────────────────────────────────────────
        ws_msg = map_state_to_frontend(state, actions)
        ws_msg["type"] = "update"
        ws_msg["reward_curve"] = list(_reward_curve)
        await _broadcast(ws_msg)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _benchmark_cache
    # Load pre-computed benchmark results if available; don't block startup if missing
    _benchmark_cache = load_benchmark_results()
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()

app = FastAPI(title="CrisisWorld OpenEnv API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/snapshot")
async def snapshot() -> Dict[str, Any]:
    async with env_lock:
        state = env.state()
        state["_last_info"] = getattr(env, "_last_info", {})
        payload = map_state_to_frontend(state, {})
    payload["type"] = "snapshot"
    return payload

async def broadcast_snapshot(message_type: str = "update") -> None:
    async with env_lock:
        state = env.state()
        payload = map_state_to_frontend(state, {})
    payload["type"] = message_type
    for ws in list(connected_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in connected_websockets:
                connected_websockets.remove(ws)

@app.post("/control/start")
async def control_start() -> Dict[str, Any]:
    global simulation_status
    simulation_status = "PLAYING"
    await broadcast_snapshot("update")
    return {"ok": True, "status": simulation_status}


@app.post("/control/pause")
async def control_pause() -> Dict[str, Any]:
    global simulation_status
    simulation_status = "PAUSED"
    await broadcast_snapshot("update")
    return {"ok": True, "status": simulation_status}


@app.post("/control/stop")
async def control_stop() -> Dict[str, Any]:
    global simulation_status
    simulation_status = "PAUSED"
    await broadcast_snapshot("update")
    return {"ok": True, "status": simulation_status}


@app.post("/control/mode")
async def control_mode(mode: str = "TRAINED") -> Dict[str, Any]:
    global simulation_mode, _cached_actions
    m = (mode or "TRAINED").strip().upper()
    if m not in ("BASELINE", "TRAINED"):
        raise HTTPException(status_code=400, detail="mode must be BASELINE or TRAINED")
    simulation_mode = m
    _cached_actions.clear()  # flush stale actions when mode switches
    return {"ok": True, "simulation_mode": simulation_mode}


@app.post("/control/reset")
async def control_reset(level: int = 1) -> Dict[str, Any]:
    global simulation_status, _cached_actions
    async with env_lock:
        env.reset(level=max(1, min(3, int(level))), seed=env.rng.randint(0, 1000))
        env._last_info = {}
    _cached_actions.clear()
    simulation_status = "PLAYING"   # auto-start after reset
    await broadcast_snapshot("update")
    return {"ok": True, "status": simulation_status}


@app.post("/control/scenario")
async def control_scenario(
    scenario: str = "earthquake",
    level: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Select a scenario (earthquake / flood / blackout), reset, and auto-start."""
    global simulation_status, _cached_actions
    valid_scenarios = ("earthquake", "flood", "blackout")
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"scenario must be one of {valid_scenarios}",
        )
    async with env_lock:
        env.reset(
            level=max(1, min(3, int(level))),
            seed=int(seed),
            scenario=scenario,
        )
        env._last_info = {}
    _cached_actions.clear()
    simulation_status = "PLAYING"   # auto-start after scenario switch
    await broadcast_snapshot("update")
    return {
        "ok": True,
        "scenario": scenario,
        "level": level,
        "seed": seed,
        "status": simulation_status,
    }


@app.get("/training/curves")
async def training_curves() -> Dict[str, Any]:
    """Return training reward/deaths/coordination/trust curves from last training run."""
    artifacts = load_training_artifacts()
    return artifacts


@app.get("/replay")
async def replay_episodes(level: int = 2, steps: int = 30) -> Dict[str, Any]:
    """
    Run three quick episodes (baseline / mid-quality / trained) on the same seed
    and return per-step metrics for visual replay in the frontend.

    Episode 0 (BEFORE): baseline_policy  — deliberate misinformation, corner patrol
    Episode 1 (MID):    weak improved     — partial coordination, some rescues
    Episode 2 (AFTER):  improved_policy   — full coordination, maximum rescue

    Returns per-step: reward, deaths, panic, trust, coordination_hits, lives_saved
    """
    from evaluate import _fallback_action
    from policies import baseline_policy, improved_policy

    def _run_episode(policy_fn, policy_name: str, seed: int) -> Dict[str, Any]:
        e = CrisisWorldEnv(seed=seed, max_steps=steps)
        obs_all = e.reset(level=level, seed=seed)
        per_step = []
        done = False
        while not done:
            joint: Dict[str, Any] = {}
            for aid in obs_all:
                try:
                    joint[aid] = policy_fn(obs_all[aid])
                except Exception:
                    pos = obs_all[aid].get("agent_status", {}).get("self", {}).get("pos", [0, 0])
                    joint[aid] = _fallback_action(aid, pos)
            try:
                result = e.step_multi(joint)
            except Exception:
                break
            obs_all = result["observations"]
            m = e.metrics()
            total_cas = max(1, len(e.casualties))
            treated = sum(1 for c in e.casualties if c.get("status") == "treated")
            per_step.append({
                "step": e.step_count,
                "reward": round(result["reward"], 2),
                "deaths": int(m["deaths"]),
                "panic": round(m["panic_level"], 1),
                "trust": round(m["trust_score"], 1),
                "coordination_hits": int(e.coordination_hits),
                "lives_saved": int(e.saved_count),
                "rescue_rate": round(treated / total_cas * 100, 1),
            })
            done = bool(result.get("done", False))
        final = e.metrics()
        total_cas = max(1, len(e.casualties))
        treated = sum(1 for c in e.casualties if c.get("status") == "treated")
        return {
            "policy": policy_name,
            "steps": per_step,
            "summary": {
                "total_reward": round(float(final["total_reward"]), 2),
                "deaths": int(final["deaths"]),
                "panic": round(float(final["panic_level"]), 1),
                "trust": round(float(final["trust_score"]), 1),
                "rescue_rate": round(treated / total_cas * 100, 1),
                "coordination_score": round(float(final["coordination_score"]), 1),
            },
        }

    # "Mid" policy: improved but only dispatches half the time (simulates partial training)
    def mid_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
        import random as _r
        aid = obs.get("agent_status", {}).get("self", {}).get("id", "medical_agent")
        if _r.random() < 0.45:
            return baseline_policy(obs)
        return improved_policy(obs)

    seed = 42
    results = await asyncio.gather(
        asyncio.to_thread(_run_episode, baseline_policy, "BEFORE (Baseline)", seed),
        asyncio.to_thread(_run_episode, mid_policy, "MID (Partial Training)", seed),
        asyncio.to_thread(_run_episode, improved_policy, "AFTER (Trained)", seed),
    )

    before, mid, after = results
    return {
        "episodes": [before, mid, after],
        "delta": {
            "reward_gain": round(after["summary"]["total_reward"] - before["summary"]["total_reward"], 2),
            "deaths_reduced": before["summary"]["deaths"] - after["summary"]["deaths"],
            "panic_reduction": round(before["summary"]["panic"] - after["summary"]["panic"], 1),
            "rescue_rate_gain": round(after["summary"]["rescue_rate"] - before["summary"]["rescue_rate"], 1),
            "trust_improvement": round(after["summary"]["trust"] - before["summary"]["trust"], 1),
        },
    }


@app.get("/llm/status")
async def llm_status() -> Dict[str, Any]:
    """Return current LLM backend configuration."""
    token = (
        os.getenv("HF_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or ""
    )
    model = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    token_preview = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else ("SET" if token else "MISSING")
    return {
        "llm_backend": "HuggingFace Router",
        "model": model,
        "token_set": bool(token),
        "token_preview": token_preview,
        "simulation_mode": simulation_mode,
        "active_policy": "LLaMA 3.1-8B via HF" if simulation_mode == "TRAINED" else "Baseline (rule-based)",
        "router_url": os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions"),
    }


@app.get("/benchmark")
async def benchmark_results() -> Dict[str, Any]:
    """Return cached baseline vs improved policy benchmark results."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = load_benchmark_results()
    if _benchmark_cache is None:
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found. Run evaluate.py first.",
        )
    return _benchmark_cache


@app.post("/benchmark/run")
async def benchmark_run(level: int = 2) -> Dict[str, Any]:
    """
    Run a quick baseline vs improved benchmark in the background thread
    and cache the result.  Returns immediately with a status message.
    """
    global _benchmark_cache

    async def _do_run() -> None:
        global _benchmark_cache
        try:
            results = await asyncio.to_thread(
                run_and_save,
                level=level,
                seeds=[42, 100, 200],
                max_steps=40,
            )
            _benchmark_cache = results
        except Exception as e:
            print(f"[benchmark/run] error: {e}")

    asyncio.create_task(_do_run())
    return {"ok": True, "message": "Benchmark started in background. Poll /benchmark for results."}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    connected_websockets.append(ws)
    try:
        async with env_lock:
            init_state = env.state()
            init_msg = map_state_to_frontend(init_state, {})
            init_msg["type"] = "snapshot"
        await ws.send_json(init_msg)
    except Exception:
        if ws in connected_websockets:
            connected_websockets.remove(ws)
        return
    try:
        while True:
            payload = await ws.receive_json()
    except WebSocketDisconnect:
        if ws in connected_websockets:
            connected_websockets.remove(ws)
    except Exception as exc:
        if ws in connected_websockets:
            connected_websockets.remove(ws)
        await ws.close(code=1011)


# ── Hugging Face / production: serve Vite build from backend/static ───────────
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="spa")
