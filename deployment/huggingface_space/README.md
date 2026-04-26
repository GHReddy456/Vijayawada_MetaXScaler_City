# Hugging Face Space (reference)

The **production Space** should use the **repository root** as the Docker build context:

- **Dockerfile**: `Dockerfile` at repo root (builds `openenv_crisisworld` on port **7860**).
- **README** for the Space: copy the YAML card below into the Space’s README (replace `YOUR_ORG`).

```yaml
---
title: CrisisWorld OpenEnv
emoji: 🌐
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: apache-2.0
short_description: OpenEnv multi-agent disaster simulation — reset, step, state API
---
```

After deploy, paste the public Space URL into the root `README.md` → **Hackathon submission links**.
