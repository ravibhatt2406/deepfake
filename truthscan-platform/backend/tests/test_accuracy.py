import pytest
import numpy as np
from ai_detection.preprocessor import TextPreprocessor
from image_forensics.detectors.ela import ELADetector

@pytest.mark.asyncio
async def test_linguistic_analysis_ai():
    # Content with high LLM markers
    ai_text = "Moreover, it is important to note that Furthermore, in conclusion, consequently it is a fact."
    img = await TextPreprocessor.text_to_image(ai_text)
    # Channel 0 left side (val_m) should be high
    val_m = img[0, 0, 0]
    assert val_m > 100 # Should trigger marker density

@pytest.mark.asyncio
async def test_linguistic_analysis_human():
    # Natural text
    human_text = "Wait, I just went to the store and it was really busy. No markers here."
    img = await TextPreprocessor.text_to_image(human_text)
    val_m = img[0, 0, 0]
    assert val_m < 50 # Should be low

def test_ela_tamper_detection():
    # Create a dummy image with a "tampered" block (different noise)
    img_data = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # Uniform block (will have different ELA signature)
    img_data[50:100, 50:100, :] = 128
    
    det = ELADetector()
    res = det.analyze(img_data)
    assert res['tamper_score'] > 0
