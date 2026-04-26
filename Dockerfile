# Hugging Face Space deployment — serve React frontend + FastAPI backend
FROM node:20-alpine AS frontend-builder
WORKDIR /app

COPY package.json ./
COPY tsconfig.app.json ./
COPY tsconfig.json ./
COPY tsconfig.node.json ./
COPY vite.config.ts ./
COPY index.html ./
COPY src ./src

RUN npm install && npm run build

FROM python:3.11-slim
WORKDIR /app/backend

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY --from=frontend-builder /app/backend/static ./static

ENV PYTHONPATH=/app/backend
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
