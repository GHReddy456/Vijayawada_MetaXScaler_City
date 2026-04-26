#!/usr/bin/env bash
# Deploy CrisisWorld to Hugging Face Spaces using ONLY the Hugging Face CLI (hf).
# Requires: pip install -U huggingface_hub
#
# Usage:
#   export HF_TOKEN=hf_...
#   export HF_REPO_ID=ghreddy/metaXscalar_crisis_vijayawada_panic_city
#   bash deployment/deploy_hf_space.sh
#
# Optional: SKIP_README=1  SKIP_HUB_CLEANUP=1

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${HF_REPO_ID:-}" ]]; then
  echo "Set HF_REPO_ID (e.g. ghreddy/metaXscalar_crisis_vijayawada_panic_city)" >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN (write token from huggingface.co/settings/tokens)" >&2
  exit 1
fi

hf version
hf auth login --token "$HF_TOKEN"

hf repos create "$HF_REPO_ID" --repo-type space --space-sdk docker --exist-ok

STAGE="$ROOT/.hf_space_staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$ROOT/Dockerfile" "$STAGE/"
[[ -f "$ROOT/openenv.yaml" ]] && cp "$ROOT/openenv.yaml" "$STAGE/" || true
[[ -f "$ROOT/LICENSE" ]] && cp "$ROOT/LICENSE" "$STAGE/" || true

if [[ "${SKIP_README:-0}" != "1" ]] && [[ -f deployment/huggingface_space/SPACE_README.md ]]; then
  cp deployment/huggingface_space/SPACE_README.md "$STAGE/README.md"
fi

mkdir -p "$STAGE/backend"
rsync -a --delete \
  --exclude 'node_modules/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude 'wandb/' \
  --exclude 'runs/' \
  --exclude 'checkpoints/' \
  --exclude 'outputs/' \
  --exclude '.git/' \
  --exclude '.ruff_cache/' \
  --exclude '.eggs/' \
  --exclude 'dist/' \
  --exclude 'build/' \
  --exclude '.tox/' \
  --exclude 'htmlcov/' \
  --exclude '*.pyc' \
  "$ROOT/backend/" "$STAGE/backend/"

CLEANUP_ARGS=()
if [[ "${SKIP_HUB_CLEANUP:-0}" != "1" ]]; then
  CLEANUP_ARGS+=(
    --delete "node_modules/**"
    --delete "**/node_modules/**"
    --delete "package.json"
    --delete "package-lock.json"
    --delete "pnpm-lock.yaml"
    --delete "yarn.lock"
    --delete "index.html"
    --delete "vite.config.ts"
    --delete "vite.config.js"
    --delete "tailwind.config.js"
    --delete "tailwind.config.ts"
    --delete "postcss.config.js"
    --delete "tsconfig.json"
    --delete "tsconfig.app.json"
    --delete "tsconfig.node.json"
    --delete "app.py"
    --delete "app.R"
    --delete "ui.R"
    --delete "server.R"
    --delete "renv.lock"
    --delete "install.R"
    --delete "requirements.txt"
    --delete "runtime.txt"
    --delete "Procfile"
    --delete ".streamlit/**"
  )
fi

hf upload "$HF_REPO_ID" "$STAGE" . --repo-type space \
  --token "$HF_TOKEN" \
  "${CLEANUP_ARGS[@]}" \
  --commit-message "Deploy CrisisWorld OpenEnv Docker Space (minimal + template cleanup)"

echo ""
echo "Space metadata (sdk should be docker):"
hf spaces info "$HF_REPO_ID" --expand sdk,runtime,subdomain --format json --token "$HF_TOKEN"

echo ""
echo "Done. Space: https://huggingface.co/spaces/${HF_REPO_ID}"
echo "After build, API docs: https://huggingface.co/spaces/${HF_REPO_ID}/docs"
echo "Tip: rm -rf .hf_space_staging to reclaim disk space."
