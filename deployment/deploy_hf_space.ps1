# Deploy CrisisWorld to Hugging Face Spaces using ONLY the Hugging Face CLI (hf).
# Requires: pip install -U huggingface_hub
#
# Uploads a minimal tree from .hf_space_staging/ (Dockerfile, README with sdk: docker,
# openenv.yaml, LICENSE, backend/) and deletes stray/template files on the Hub in the
# same commit so the Space rebuilds as Docker + OpenEnv (not Shiny/Gradio templates).
#
# Usage:
#   $env:HF_TOKEN = "hf_..."           # write token; prefer hf auth login
#   $env:HF_REPO_ID = "ghreddy/metaXscalar_crisis_vijayawada_panic_city"
#   .\deployment\deploy_hf_space.ps1
#
# Optional:
#   .\deployment\deploy_hf_space.ps1 -SkipReadme
#   .\deployment\deploy_hf_space.ps1 -SkipHubCleanup

param(
    [string] $RepoId = $env:HF_REPO_ID,
    [switch] $SkipReadme,
    [switch] $SkipHubCleanup
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $Root "Dockerfile"))) {
    Write-Error "Dockerfile not found under $Root - run this script from the CrisisWorld repo."
}

if (-not $RepoId) {
    Write-Error "Set HF_REPO_ID, e.g. ghreddy/metaXscalar_crisis_vijayawada_panic_city"
}

$token = $env:HF_TOKEN
if (-not $token) {
    Write-Error "Set HF_TOKEN to a Hugging Face access token (write permission)."
}

hf version | Out-Host
hf auth login --token $token

hf repos create $RepoId --repo-type space --space-sdk docker --exist-ok

$Stage = Join-Path $Root ".hf_space_staging"
if (Test-Path $Stage) {
    Remove-Item $Stage -Recurse -Force
}
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

Copy-Item (Join-Path $Root "Dockerfile") $Stage -Force
if (Test-Path (Join-Path $Root "openenv.yaml")) {
    Copy-Item (Join-Path $Root "openenv.yaml") $Stage -Force
}
if (Test-Path (Join-Path $Root "LICENSE")) {
    Copy-Item (Join-Path $Root "LICENSE") $Stage -Force
}

if (-not $SkipReadme) {
    $card = Join-Path $PSScriptRoot "huggingface_space\SPACE_README.md"
    if (Test-Path $card) {
        Copy-Item $card (Join-Path $Stage "README.md") -Force
    }
}

$srcBackend = Join-Path $Root "backend"
$dstBackend = Join-Path $Stage "backend"
New-Item -ItemType Directory -Path $dstBackend -Force | Out-Null
robocopy $srcBackend $dstBackend /E `
    /XD node_modules .venv __pycache__ .mypy_cache .pytest_cache wandb runs checkpoints outputs .git .ruff_cache .eggs dist build .tox htmlcov `
    /XF *.pyc `
    /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Host
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

$uploadArgs = @(
    "upload", $RepoId, $Stage, ".",
    "--repo-type", "space",
    "--token", $token,
    "--commit-message", "Deploy CrisisWorld OpenEnv Docker Space (minimal + template cleanup)"
)

if (-not $SkipHubCleanup) {
    $uploadArgs += @(
        "--delete", "node_modules/**",
        "--delete", "**/node_modules/**",
        "--delete", "package.json",
        "--delete", "package-lock.json",
        "--delete", "pnpm-lock.yaml",
        "--delete", "yarn.lock",
        "--delete", "index.html",
        "--delete", "vite.config.ts",
        "--delete", "vite.config.js",
        "--delete", "tailwind.config.js",
        "--delete", "tailwind.config.ts",
        "--delete", "postcss.config.js",
        "--delete", "tsconfig.json",
        "--delete", "tsconfig.app.json",
        "--delete", "tsconfig.node.json",
        "--delete", "app.py",
        "--delete", "app.R",
        "--delete", "ui.R",
        "--delete", "server.R",
        "--delete", "renv.lock",
        "--delete", "install.R",
        "--delete", "requirements.txt",
        "--delete", "runtime.txt",
        "--delete", "Procfile",
        "--delete", ".streamlit/**"
    )
}

& hf @uploadArgs

Write-Host ""
Write-Host "Space metadata (sdk should be docker):"
hf spaces info $RepoId --expand sdk,runtime,subdomain --format json --token $token | Out-Host

$url = 'https://huggingface.co/spaces/' + $RepoId
$docsUrl = $url.TrimEnd('/') + '/docs'
Write-Host ""
Write-Host ('Done. Space: ' + $url)
Write-Host ('After build, API docs: ' + $docsUrl)
Write-Host 'If sdk is not docker, wait for README to refresh or re-run this script.'
Write-Host 'Tip: remove .hf_space_staging locally if you want to reclaim disk space.'
