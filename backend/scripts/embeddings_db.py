'''This script aims at creating vector stores for text and images. It was not used for the MVP.'''

import numpy as np
import faiss
from backend.utils.api_utils import AlbertAPI

api = AlbertAPI()
# Initialize FAISS with Inner Product (IP) for cosine similarity
dimension = 768  # Adjust based on embedding model
vector_db = faiss.IndexFlatIP(dimension)  # IP (dot product) works for cosine similarity
image_metadata = {}  # Store image paths linked to embeddings

def store_text_embeddings(chunks):
    """
    Converts text chunks into embeddings, normalizes them using FAISS, and stores them.
    """
    embeddings = []
    for chunk in chunks:
        embedding = api.create_embedding(chunk)
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize using FAISS (instead of manually with NumPy)
    faiss.normalize_L2(embeddings)

    # Store normalized embeddings in FAISS
    vector_db.add(embeddings)

def store_image_descriptions(image_descriptions):
    """
    Converts image descriptions into embeddings, normalizes them using FAISS, and stores them.
    """
    global image_metadata
    embeddings = []

    for image_path, description in image_descriptions.items():
        embedding = api.create_embedding(description)
        embeddings.append(embedding)
        image_metadata[len(image_metadata)] = {
            "description": description,
            "image_path": image_path
        }

    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize embeddings before adding to FAISS
    faiss.normalize_L2(embeddings)
    vector_db.add(embeddings)
