FROM python:3.10

WORKDIR /app/

RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.27.1 langchain==0.3.9 langchain-openai==0.2.11

COPY api.py .

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", \
    "--limit-concurrency", "20", "--workers", "2", "api:app"]