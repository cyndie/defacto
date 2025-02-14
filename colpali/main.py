import regex as re
import string
import os
import io
import gc
import tempfile
import argparse
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path
from PIL import Image

from pydantic import BaseModel

import requests
import base64
from PIL import Image
import io
import os
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor
from typing import Optional, List 

app = FastAPI()


colqwen_model = ColQwen2.from_pretrained(
    "manu/colqwen2-v1.0-alpha",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
).eval()

colqwen_processor = ColQwen2Processor.from_pretrained("manu/colqwen2-v1.0-alpha")

def clean_gpu_memory():
    """Clean GPU memory and cache"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def pil_to_file(image_bytes):
    """Convert PIL image bytes to temporary file"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
        image_pil = Image.open(image_bytes)
        image_pil.save(temp_image, format="PNG")
        return temp_image.name

def pdf_to_images_bytes(pdf_path, dpi=300):
    """Convert PDF to list of image bytes"""
    images = convert_from_path(pdf_path, dpi=dpi)
    images_bytes = []
    for img in images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        images_bytes.append(img_bytes)
    return images_bytes

global_state = {
    "embeddings": [],
    "images": []
}

class SearchRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/v0/visual_embed")
async def visual_embed(
    files: List[UploadFile] = File(...),
):
    """
    Create visual embeddings for uploaded PDF files
    """
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            filepath = f"/tmp/{file.filename}"
            with open(filepath, "wb") as temp_file:
                temp_file.write(await file.read())
            temp_files.append(filepath)

        # Convert files to images
        images = []
        for f in temp_files:
            images.extend(convert_from_path(f, thread_count=4))

        if len(images) >= 150:
            return Response(
                status_code=400,
                content="The number of images in the dataset should be less than 150."
            )

        # Process images with ColQwen2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dataloader = DataLoader(
            images,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: colqwen_processor.process_images(x).to(device),
        )

        embeddings = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
                embeddings_doc = colqwen_model(**batch_doc)
            embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        # Update global state
        global_state["embeddings"] = embeddings
        global_state["images"] = images

        return JSONResponse({
            "status": "success",
            "message": f"Processed {len(images)} pages",
            "num_embeddings": len(embeddings)
        })

    except Exception as e:
        return Response(
            status_code=500,
            content=f"Failed to process files: {str(e)}"
        )


@app.post("/v0/search")
async def search(request: SearchRequest):
    """
    Search through indexed documents using a text query
    """
    try:
        if not global_state["embeddings"]:
            return Response(
                status_code=400,
                content="No documents have been indexed. Please call /v0/visual_embed first."
            )

        k = min(request.k, len(global_state["embeddings"]))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Process query
        with torch.no_grad():
            batch_query = colqwen_processor.process_queries([request.query]).to(device)
            embeddings_query = colqwen_model(**batch_query)
            query_embeddings = list(torch.unbind(embeddings_query.to("cpu")))

        # Calculate scores
        scores = colqwen_processor.score(query_embeddings, global_state["embeddings"], device=device)
        top_k_indices = scores[0].topk(k).indices.tolist()

        # Convert results to base64 images
        results = []
        for idx in top_k_indices:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            global_state["images"][idx].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert to base64
            import base64
            img_base64 = base64.b64encode(img_byte_arr).decode()
            
            results.append({
                "page_number": idx,
                "image": img_base64,
                "score": float(scores[0][idx])
            })

        return JSONResponse({
            "results": results
        })

    except Exception as e:
        return Response(
            status_code=500,
            content=f"Search failed: {str(e)}"
        )
    finally:
        clean_gpu_memory()





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)