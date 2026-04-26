# OpenEnv Hackathon — submission checklist (CrisisWorld)

**Root `README.md` note:** If GitHub shows a broken YAML header, re-save `README.md` as **UTF-8** in your editor and paste the table of contents entry plus Section 0 from this file into the main README.

## 0. Links (replace placeholders before submit)

| Deliverable | URL |
|-------------|-----|
| **Hugging Face Space** | `https://huggingface.co/spaces/ghreddy/metaXscalar_crisis_vijayawada_panic_city` |
| **Mini-blog or &lt;2 min video** | Your HF blog or YouTube link |
| **Colab** | [notebooks/CrisisWorld_GRPO_TRL_Colab.ipynb](notebooks/CrisisWorld_GRPO_TRL_Colab.ipynb) |

## 1a. Deploy the Space (Hugging Face CLI only)

Use the modern CLI bundled with `huggingface_hub` (command name **`hf`**, not the old `huggingface-cli` alias on some installs):

```bash
pip install -U huggingface_hub
export HF_TOKEN=hf_...                    # write token
export HF_REPO_ID=ghreddy/metaXscalar_crisis_vijayawada_panic_city
bash deployment/deploy_hf_space.sh
```

**Windows (PowerShell):**

```powershell
pip install -U huggingface_hub
$env:HF_TOKEN = "hf_..."
$env:HF_REPO_ID = "ghreddy/metaXscalar_crisis_vijayawada_panic_city"
.\deployment\deploy_hf_space.ps1
```

This creates a **Docker** Space (if missing), uploads `deployment/huggingface_space/SPACE_README.md` as the Hub **`README.md`** (YAML card + `app_port: 7860`), then uploads the repo root with excludes for `node_modules`, `.git`, etc. Build context stays the **repository root** so the root **`Dockerfile`** and **`backend/`** layout work.

Manual equivalent:

```bash
hf auth login --token "$HF_TOKEN"
hf repo create "$HF_REPO_ID" --repo-type space --space-sdk docker --exist-ok
hf upload "$HF_REPO_ID" deployment/huggingface_space/SPACE_README.md README.md --repo-type space
hf upload "$HF_REPO_ID" . --repo-type space --exclude "**/node_modules/**" --exclude "**/.git/**"
```

## 1. OpenEnv compliance

- Package: **`openenv-core`** in `backend/requirements.txt`.
- Manifest: **[openenv.yaml](openenv.yaml)** (repo root).
- Adapter: **`backend/openenv_crisisworld/`** — `CrisisWorldOpenEnvironment` subclasses `openenv.core.env_server.interfaces.Environment`.
- Docker Space: **[Dockerfile](Dockerfile)** (build context = repository root).

## 2. Training evidence

- Plots: [docs/training_plots/reward_vs_episode.png](docs/training_plots/reward_vs_episode.png), [docs/training_plots/loss_vs_step.png](docs/training_plots/loss_vs_step.png)
- Regenerate from a real checkpoint:

  `cd backend && pip install -r requirements-train.txt && python scripts/plot_training_run.py --checkpoint-dir checkpoints/crisisworld-grpo --out-dir ../docs/training_plots`

- Demo JSON (layout only): [docs/demo_grpo_artifact/](docs/demo_grpo_artifact/)

## 3. Baseline vs improved (rule-based benchmark)

```bash
cd backend && python scripts/compare_baseline_vs_improved.py --seeds 20 --level 2
```

Output: [docs/results/baseline_vs_improved.json](docs/results/baseline_vs_improved.json). Add GRPO-trained numbers when your LoRA is ready.

## 4. GRPO reward story (judges)

Train-time reward in `train_grpo.py` mixes **format / legality**, **sim outcomes** (lives, deaths, panic, trust), **role-appropriate communication**, and a **bounded** mapping of env `comm_reward_bonus` sums (symmetric clamp — avoids a historic `min(cap, raw*k)` bug that sent negative totals to hundreds). Illegal actions are penalized without auto-repair.
