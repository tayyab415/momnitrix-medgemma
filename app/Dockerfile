FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-cloudrun.txt /app/requirements-cloudrun.txt
RUN pip install --no-cache-dir -r /app/requirements-cloudrun.txt

COPY momnitrix /app/momnitrix
COPY cloudrun_backend.py /app/cloudrun_backend.py

EXPOSE 8080
CMD ["sh", "-c", "uvicorn cloudrun_backend:app --host 0.0.0.0 --port ${PORT:-8080}"]
