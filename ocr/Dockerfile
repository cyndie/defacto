FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app/
RUN mkdir -p /tmp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY content-mmd-to-html.html .
COPY tikz.html .
COPY got.py .

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8001", \
    "--limit-concurrency", "20", "--workers", "2", "got:app"]