from fastapi import APIRouter, UploadFile, File, HTTPException
from shared.schemas import ImageForensicsResponse
import logging
from .detectors.ela import ELADetector
from .detectors.metadata import MetadataDetector
import numpy as np
from PIL import Image
import io

router = APIRouter()
logger = logging.getLogger("truthscan.image.router")
engine = ImageDecisionEngine()
ela_detector = ELADetector()
metadata_detector = MetadataDetector()

@router.post("/analyze", response_model=ImageForensicsResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Real implementation of image forensics.
    """
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        image_array = np.asarray(img)

        # 1. Run Detectors
        ela_res = ela_detector.analyze(image_array)
        meta_res = metadata_detector.analyze(image_array)
        
        # 2. Finalize Verdict
        # We also pass a dummy ai_score of 0 for this module, or integrate with ai_detection
        final = engine.finalize_verdict({"ai_score": 0.0}, ela_res)
        
        return ImageForensicsResponse(
            verdict=final["verdict"],
            confidence_score=final["confidence"],
            detailed_analysis={
                "ela": ela_res,
                "metadata": meta_res,
                "explanation": final["explanation"]
            }
        )
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
