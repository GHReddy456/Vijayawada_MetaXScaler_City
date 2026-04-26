---
title: metaXscalar — Crisis Vijayawada (Panic City)
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: CrisisWorld OpenEnv Docker API for disaster-response agents.
---

# metaXscalar — Crisis Vijayawada (Panic City)

**OpenEnv** Docker Space: `uvicorn openenv_crisisworld.app:app` on port **7860**.

## What you see in the “App” tab

This Space exposes the **OpenEnv HTTP API** (JSON), not the React map UI from the monorepo.

- **Interactive API:** open **`/docs`** (Swagger UI) on this Space, or the root URL `/` (redirects to `/docs`).
- **Wrong UI?** If you see **“Restaurant tipping”**, Shiny, or random tutorial widgets, the Space is still running an old **template image** or the wrong SDK. In **Settings**, confirm **Docker** is selected; in **Files**, confirm the repo root `Dockerfile` has `CMD ["uvicorn", "openenv_crisisworld.app:app", ...]`; then **Factory reboot** / trigger a fresh build after redeploying.

## Where the CrisisWorld frontend lives

The **Vijayawada map / dashboard** is the **Vite + React** app in the GitHub/clone repo (not uploaded to this Docker Space). Run it locally, e.g. `npm install && npm run dev`, and point it at this Space’s API if your app supports that.

Multi-agent simulation (medical, police, logistics, communication, commander) with partial observability and GRPO-ready hooks.

**Optional storage:** If you attached a Hub bucket at `/data`, use it for checkpoints or logs from Space settings (read/write as configured).

Full source, training (`train_grpo.py`), and evaluation scripts live in the linked Git repository.
