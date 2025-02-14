import requests
from requests.adapters import HTTPAdapter
import base64
from PIL import Image
import io
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from requests.adapters import Retry

# Configuration
BASE_URL = "http://localhost:8002"

def test_visual_embed(pdf_paths):
    """
    Test l'endpoint visual_embed avec un ou plusieurs PDFs
    """
    url = f"{BASE_URL}/v0/visual_embed"
    
    # Prépare les fichiers pour l'upload
    files = [
        ('files', (pdf_path.split('/')[-1], open(pdf_path, 'rb'), 'application/pdf'))
        for pdf_path in pdf_paths
    ]
    try:        
        response = requests.post(url, files=files)
        response.raise_for_status()
        print("Visual Embed Response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during visual_embed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Error details: {e.response.text}")
        return None
    finally:
        # Ferme tous les fichiers
        for file_tuple in files:
            file_tuple[1][1].close()  # Accède au fichier dans le tuple imbriqué

def test_search(query, k=5):
    """
    Test l'endpoint search avec une requête
    """
    url = f"{BASE_URL}/v0/search"
    payload = {
        "query": query,
        "k": k
    }
    try:        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        results = response.json()["results"]
        print(f"\nFound {len(results)} results for query: '{query}'")
        
        # Sauvegarde les images résultantes
        for i, result in enumerate(results):
            # Décode l'image base64
            img_data = base64.b64decode(result["image"])
            img = Image.open(io.BytesIO(img_data))
            
            # Sauvegarde l'image
            output_path = f"result_{i}_page_{result['page_number']}.png"
            img.save(output_path)
            print(f"Page {result['page_number']} (Score: {result['score']:.4f}) saved as {output_path}")
        
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error during search: {e}")
        if hasattr(e.response, 'text'):
            print(f"Error details: {e.response.text}")
        return None

if __name__ == "__main__":
    # Test visual_embed
    pdf_paths = [
        "dos.pdf"
    ]
    
    print("Testing visual_embed...")
    embed_result = test_visual_embed(pdf_paths)
    
    if embed_result:
        # Test search
        print("\nTesting search...")
        search_queries = [
            "Qui est la victime?"
        ]
        
        for query in search_queries:
            test_search(query, k=2)