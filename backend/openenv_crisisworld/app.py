"""
OpenEnv FastAPI application for CrisisWorld.

Run from the `backend` directory:

    uvicorn openenv_crisisworld.app:app --host 0.0.0.0 --port 7860

Hugging Face Spaces: set the container CMD to the same (port 7860).
"""

from __future__ import annotations

import os

from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app

from openenv_crisisworld.environment import CrisisWorldOpenEnvironment
from openenv_crisisworld.models import CrisisWorldAction, CrisisWorldObservation

_max = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    CrisisWorldOpenEnvironment,
    CrisisWorldAction,
    CrisisWorldObservation,
    env_name="crisisworld",
    max_concurrent_envs=_max,
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """HF Spaces open `/` in the App tab; send humans to interactive API docs."""
    return RedirectResponse(url="/docs", status_code=302)


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
