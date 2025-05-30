from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import logging

# تهيئة logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ImageURLs(BaseModel):
    original_image_url: str
    new_image_url: str

def load_image_from_url(url: str, timeout: int = 10):
    """تحميل الصورة من URL مع معالجة الأخطاء المحسنة"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading image from URL {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing image {url}: {str(e)}")
        return None

@app.post("/verify-driver/")
async def verify_driver(images: ImageURLs):
    """مقارنة وجهين من صورتين"""
    # تحميل الصورتين مع timeout
    original_image = load_image_from_url(images.original_image_url)
    new_image = load_image_from_url(images.new_image_url)

    if original_image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to load the original image. Please check the URL and try again."
        )
    if new_image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to load the new image. Please check the URL and try again."
        )

    try:
        # تحويل الصور إلى numpy arrays
        img1_array = np.array(original_image)
        img2_array = np.array(new_image)
        
        # استخدام enforce_detection=False لتجنب الأخطاء عند عدم وجود وجوه
        result = DeepFace.verify(
            img1_path=img1_array,
            img2_path=img2_array,
            detector_backend='opencv',  # تحديد backend للكشف
            enforce_detection=True,
            distance_metric='cosine'  # يمكن تجربة 'euclidean' أو 'cosine'
        )
        
        logger.info(f"Verification result: {result}")

        return {
            "match": result["verified"],
            "distance": float(result["distance"]),  # تحويل إلى float لضمان التوافق مع JSON
            "threshold": float(result["threshold"]),
            "message": "Faces match" if result["verified"] else "Faces do not match"
        }

    except ValueError as e:
        if "Face could not be detected" in str(e):
            raise HTTPException(
                status_code=400,
                detail="No face detected in one or both images. Please provide clear frontal face images."
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"DeepFace verification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during face verification. Please try again later."
        )