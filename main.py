from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import numpy as np
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

class ImageURLs(BaseModel):
    original_image_url: str
    new_image_url: str

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None

@app.post("/verify-driver/")
async def verify_driver(images: ImageURLs):
    # تحميل الصورتين
    original_image = load_image_from_url(images.original_image_url)
    new_image = load_image_from_url(images.new_image_url)

    if original_image is None:
        raise HTTPException(status_code=400, detail="Failed to load the original image.")
    if new_image is None:
        raise HTTPException(status_code=400, detail="Failed to load the new image.")

    try:
        result = DeepFace.verify(
            img1_path=np.array(original_image),
            img2_path=np.array(new_image),
            enforce_detection=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DeepFace error: {str(e)}")

    return {
        "match": result["verified"],
        "distance": result["distance"],
        "threshold": result["threshold"],
        "message": "Faces match" if result["verified"] else "Faces do not match"
    }
