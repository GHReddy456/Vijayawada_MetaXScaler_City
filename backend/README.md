# CrisisWorld (OpenEnv-style RL Environment)

Production-grade, multi-agent disaster simulation with strict action/observation schemas and FastAPI/WebSocket serving.

## Files

- `environment.py`: Core environment (`reset()`, `step(action)`, `state()`)
- `models.py`: Pydantic schemas for strict validation
- `server.py`: FastAPI + WebSocket runtime
- `train_grpo.py`: Minimal TRL `GRPOTrainer` training setup with Unsloth

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Hugging Face Llama setup

Set your token before running the server (required for agent LLM actions):

```bash
export HF_API_TOKEN=hf_xxx
```

Optional overrides:

```bash
export HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
export HF_INFERENCE_URL=https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Example API usage

```bash
curl -s -X POST http://127.0.0.1:8000/reset -H "content-type: application/json" -d '{"level":2,"seed":42}'
curl -s -X POST http://127.0.0.1:8000/step -H "content-type: application/json" -d '{"agent_id":"medical_agent","action_type":"dispatch","target":[5,5],"metadata":{}}'
curl -s http://127.0.0.1:8000/metrics
curl -s "http://127.0.0.1:8000/compare?level=2&seed=42&max_steps=60"
curl -s "http://127.0.0.1:8000/compare?level=2&seed=42&max_steps=60&include_ollama=true&ollama_model=llama3"
```

## Ollama policy inference (optional)

If Ollama is running locally (`http://localhost:11434`), enable LLM policy in compare endpoint:

```bash
curl -s "http://127.0.0.1:8000/compare?include_ollama=true&ollama_model=llama3"
```

This is policy inference (decision engine), not RL training.

## Training setup (GRPO)

Install training dependencies separately:

```bash
pip install -r requirements-train.txt
```

### Train (cross-platform default)

```bash
python train_grpo.py
```

### Train with Unsloth acceleration (Linux/CUDA)

```bash
CRISISWORLD_USE_UNSLOTH=1 python train_grpo.py
```

Outputs:
- checkpoints in `checkpoints/crisisworld-grpo/`
- reward history in `checkpoints/crisisworld-grpo/reward_history.json`

# scaler_X_meta_city
