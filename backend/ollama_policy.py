from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import RequestException

from models import ActionModel


class OllamaPolicy:
    ROLE_INSTRUCTIONS = {
        "medical_agent": "Focus on saving lives. Prioritize highest severity casualties.",
        "police_agent": "Control roads, block unsafe paths, and maintain order.",
        "logistics_agent": "Keep resources balanced and avoid overload in hospitals/shelters.",
        "communication_agent": "Reduce panic and misinformation with truthful broadcasts.",
        "commander_agent": "Coordinate teams, assign tasks, and reduce duplicate work.",
    }

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        base_url: str = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
        timeout: float = 10.0,
        candidates: int = 3,
    ) -> None:
        self.model = os.getenv("HF_MODEL_ID", model)
        self.base_url = os.getenv(
            "HF_INFERENCE_URL",
            f"https://api-inference.huggingface.co/models/{self.model}",
        )
        self.timeout = timeout
        self.candidates = max(1, int(candidates))
        self.last_action_cache: Dict[str, Dict[str, Any]] = {}
        self.hf_token = (
            os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_API_TOKEN")
        )
        # Routed endpoint supports paid providers/credits with one API format.
        self.router_url = os.getenv(
            "HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions"
        )

    def build_prompt(
        self,
        observation: Dict[str, Any],
        agent_id: str,
        correction_note: str = "",
    ) -> str:
        role = self.ROLE_INSTRUCTIONS.get(agent_id, "Follow mission priorities and coordinate safely.")
        messages = observation.get("messages", [])
        visible_events = observation.get("visible_events", [])
        resource_status = observation.get("resource_status", {})
        agent_status = observation.get("agent_status", {})
        self_info = agent_status.get("self", {})
        knowledge_scope = str(self_info.get("knowledge_scope", "") or "").strip()
        time_step = observation.get("time_step", 0)
        correction = f"\nPrevious output was invalid: {correction_note}\nFix format strictly.\n" if correction_note else ""
        scope_block = (
            f"Your available knowledge (only use facts consistent with this scope):\n{knowledge_scope}\n\n"
            if knowledge_scope
            else ""
        )
        return (
            "You MUST return ONLY valid JSON.\n"
            "Allowed actions: dispatch, route, block, allocate, broadcast.\n"
            "Return EXACT format:\n"
            "{\n"
            '  "agent_id": "...",\n'
            '  "action_type": "...",\n'
            '  "target": [x,y],\n'
            '  "metadata": {"reason": "..."}\n'
            "}\n\n"
            f"Agent role: {agent_id}\n"
            f"Role instructions: {role}\n\n"
            f"{scope_block}"
            "Priority rules:\n"
            "1. Save lives > reduce panic\n"
            "2. High severity events first\n"
            "3. Avoid blocked roads\n"
            "4. Coordinate using messages\n\n"
            f"Recent messages from other agents:\n{json.dumps(messages, ensure_ascii=True)}\n"
            "If another agent is handling a task, avoid duplication.\n\n"
            f"Visible events:\n{json.dumps(visible_events, ensure_ascii=True)}\n"
            f"Resource status:\n{json.dumps(resource_status, ensure_ascii=True)}\n"
            f"Agent status:\n{json.dumps(agent_status, ensure_ascii=True)}\n"
            f"Time step: {time_step}\n"
            f"{correction}"
        )

    def generate_action(self, observation: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        actions: List[Dict[str, Any]] = []
        errors: List[str] = []
        for _ in range(self.candidates):
            action, err = self._generate_single(observation, agent_id)
            if action is not None:
                actions.append(action)
            if err:
                errors.append(err)
        if not actions:
            cached = self.last_action_cache.get(agent_id)
            return cached if cached is not None else self.fallback(observation, agent_id, "fallback")
        best = max(actions, key=lambda action: self._score_action(action, observation, agent_id))
        self.last_action_cache[agent_id] = best
        return best

    def _generate_single(self, observation: Dict[str, Any], agent_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        prompt = self.build_prompt(observation, agent_id)
        action, err = self._query_and_parse(prompt, observation, agent_id)
        if action is not None:
            return action, None

        retry_prompt = self.build_prompt(observation, agent_id, correction_note=err or "invalid JSON/action")
        retry_action, retry_err = self._query_and_parse(retry_prompt, observation, agent_id)
        if retry_action is not None:
            return retry_action, None
        return None, retry_err or err or "generation failed"

    def _query_and_parse(
        self, prompt: str, observation: Dict[str, Any], agent_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.hf_token:
            return None, "Missing Hugging Face token (set HF_API_TOKEN)."

        action, err = self._query_router(prompt, observation, agent_id)
        if action is not None:
            return action, None

        # Fallback to model inference endpoint if router/provider is unavailable.
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "parameters": {
                    "temperature": 0.01,
                    "max_new_tokens": 200,
                    "return_full_text": False
                }
            }
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            res_json = response.json()
            if isinstance(res_json, dict) and res_json.get("error"):
                return None, str(res_json["error"])
            if isinstance(res_json, list) and len(res_json) > 0 and "generated_text" in res_json[0]:
                output = res_json[0]["generated_text"]
            else:
                output = str(res_json)
            parsed = self.safe_parse(output, observation, agent_id)
            return parsed, None
        except RequestException as exc:
            return None, err or str(exc)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return None, err or str(exc)

    def _query_router(
        self, prompt: str, observation: Dict[str, Any], agent_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.01,
                "max_tokens": 220,
                "stream": False,
            }
            response = requests.post(
                self.router_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if not choices:
                return None, f"HF router returned no choices: {data}"
            msg = choices[0].get("message", {})
            output = msg.get("content", "") if isinstance(msg, dict) else ""
            if not output:
                return None, f"HF router returned empty content: {data}"
            parsed = self.safe_parse(output, observation, agent_id)
            return parsed, None
        except (RequestException, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return None, str(exc)

    def safe_parse(self, text: str, observation: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        payload = text.strip()
        if "{" in payload and "}" in payload:
            payload = payload[payload.find("{") : payload.rfind("}") + 1]
        action = json.loads(payload)
        if not isinstance(action, dict):
            raise ValueError("action must be object")
        if "metadata" not in action or not isinstance(action["metadata"], dict):
            action["metadata"] = {"reason": "llm-generated action"}
        if "reason" not in action["metadata"]:
            action["metadata"]["reason"] = "llm-generated action"
        action["agent_id"] = agent_id
        validated = ActionModel.model_validate(action)
        return validated.model_dump()

    def fallback(self, observation: Dict[str, Any], agent_id: str, reason: str) -> Dict[str, Any]:
        self_pos = observation.get("agent_status", {}).get("self", {}).get("pos", [0, 0])
        target = [int(self_pos[0]), int(self_pos[1])] if len(self_pos) == 2 else [0, 0]
        return {
            "agent_id": agent_id,
            "action_type": "route",
            "target": target,
            "metadata": {"reason": reason},
        }

    def _score_action(self, action: Dict[str, Any], observation: Dict[str, Any], agent_id: str) -> float:
        score = 0.0
        target_list = action.get("target", [9, 9])
        target = tuple(target_list) if isinstance(target_list, (list, tuple)) and len(target_list) == 2 else (9, 9)
        self_status = observation.get("agent_status", {}).get("self", {})
        self_pos = tuple(self_status.get("pos", [0, 0]))

        blocked = {
            tuple(x)
            for x in observation.get("resource_status", {}).get("local_blocked_roads", [])
        }
        if target in blocked and action.get("action_type") in {"route", "dispatch"}:
            score -= 100.0

        casualties = [e for e in observation.get("visible_events", []) if e.get("type") == "casualty"]
        if casualties:
            best = sorted(
                casualties,
                key=lambda e: (
                    -int(e.get("severity", 0)),
                    abs(e["pos"][0] - self_pos[0]) + abs(e["pos"][1] - self_pos[1]),
                ),
            )[0]
            best_pos = tuple(best["pos"])
            dist_to_best = abs(target[0] - best_pos[0]) + abs(target[1] - best_pos[1])
            score += 30.0 - float(dist_to_best * 2)
            if int(best.get("severity", 0)) >= 3:
                score += 20.0

        if agent_id == "medical_agent" and action.get("action_type") == "dispatch":
            score += 15.0
        if agent_id == "communication_agent" and action.get("action_type") == "broadcast":
            score += 12.0
        if agent_id == "police_agent" and action.get("action_type") in {"block", "route"}:
            score += 8.0
        if agent_id == "logistics_agent" and action.get("action_type") in {"allocate", "route"}:
            score += 8.0

        for msg in observation.get("messages", []):
            msg_target = tuple(msg.get("target", []))
            if len(msg_target) == 2 and msg_target == target:
                score += 10.0

        last_key = self_status.get("last_action_key")
        if isinstance(last_key, (list, tuple)) and len(last_key) >= 2:
            if last_key[0] == action.get("action_type") and tuple(last_key[1]) == target:
                score -= 12.0
        return score
