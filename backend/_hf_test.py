"""
Test HuggingFace LLaMA agent.
Run: python _hf_test.py
"""
import os, sys, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Load .env
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

token   = os.getenv("HF_API_TOKEN", "")
model   = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
router  = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")

print("=== HuggingFace LLaMA Agent Test ===")
print(f"  Model  : {model}")
print(f"  Router : {router}")
print(f"  Token  : {token[:8]}...{token[-4:] if token else 'MISSING'}")
print()

if not token:
    print("[ERROR] HF_API_TOKEN not set. Check backend/.env")
    sys.exit(1)

import requests

# Build a small prompt
prompt = (
    "You are a medical_agent in a crisis response simulation.\n"
    "Return ONLY valid JSON:\n"
    '{"agent_id":"medical_agent","action_type":"dispatch","target":[3,4],"metadata":{"reason":"..."}}\n\n'
    "Visible events: [{\"type\":\"casualty\",\"pos\":[3,4],\"severity\":3}]\n"
    "Time step: 1\n"
    "Choose the best action."
)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "Return only valid JSON, no extra text."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.01,
    "max_tokens": 150,
    "stream": False,
}

print("[1] Calling HF Router (LLaMA)...")
try:
    r = requests.post(router, headers=headers, json=payload, timeout=30)
    print(f"  HTTP status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  Raw response:\n{content}")
        # Try to parse as JSON
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start >= 0 and end > start:
            action = json.loads(content[start:end])
            print(f"\n[OK] Parsed action: {action}")
            print(f"  action_type : {action.get('action_type')}")
            print(f"  target      : {action.get('target')}")
        else:
            print("[WARN] Response is not valid JSON but request succeeded.")
    else:
        print(f"  Error body: {r.text[:400]}")
except Exception as e:
    print(f"  [FAIL] {e}")

print()
print("[2] Testing OllamaPolicy.generate_action() with full observation...")
try:
    from ollama_policy import OllamaPolicy
    policy = OllamaPolicy(model=model)
    obs = {
        "agent_status": {
            "self": {
                "id": "medical_agent",
                "pos": [3, 3],
                "energy": 80,
                "role": "medical_agent",
                "knowledge_scope": "Local casualties + hospital capacity",
            }
        },
        "visible_events": [
            {"type": "casualty", "pos": [3, 4], "severity": 3, "status": "waiting"}
        ],
        "messages": [],
        "time_step": 2,
        "uncertainty": {"comm_noise": 0.1},
    }
    result = policy.generate_action(obs, "medical_agent")
    print(f"  [OK] Action: {result}")
except Exception as e:
    print(f"  [FAIL] {e}")

print()
print("=== Test complete ===")
