# Hugging Face Space / OpenEnv deployment — OpenEnv-compliant CrisisWorld API
FROM python:3.11-slim

WORKDIR /app/backend

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

ENV PYTHONPATH=/app/backend
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "openenv_crisisworld.app:app", "--host", "0.0.0.0", "--port", "7860"]
