# syntax=docker/dockerfile:1
# Hugging Face Spaces (Docker): builds the React UI, serves API + static app on 7860.
# Set secrets in the Space: HF_TOKEN or HF_API_TOKEN, HF_MODEL_ID (optional).

FROM node:20-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY --from=frontend /build/backend/static /app/backend/static

ENV PORT=7860
EXPOSE 7860
WORKDIR /app/backend
CMD sh -c "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"
