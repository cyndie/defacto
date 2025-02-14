FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app/

# Installation des dépendances système en premier
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle pendant la construction de l'image
RUN python -c "from colpali_engine.models import ColQwen2, ColQwen2Processor; \
    model = ColQwen2.from_pretrained('manu/colqwen2-v1.0-alpha', torch_dtype='auto'); \
    processor = ColQwen2Processor.from_pretrained('manu/colqwen2-v1.0-alpha')"

# Copier le code de l'application
COPY main.py .

# Exposer le port
EXPOSE 8001

# Lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]