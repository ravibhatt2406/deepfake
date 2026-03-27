import io
import numpy as np
from PIL import Image, ImageChops

class ELADetector:
    """
    Error Level Analysis (ELA)
    Identifies areas of an image that were resaved at different compression levels.
    """
    def __init__(self, quality: int = 90):
        self.quality = quality

    def analyze(self, image_array: np.ndarray) -> dict:
        """
        Performs ELA and returns a tamper score.
        """
        try:
            original = Image.fromarray(image_array)
            
            # 1. Resave at a specific quality
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            resaved = Image.open(buffer)
            
            # 2. Compute absolute difference
            diff = ImageChops.difference(original, resaved)
            
            # 3. Enhance difference for analysis
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            
            enhanced = ImageChops.multiply(diff, Image.new("RGB", diff.size, (int(scale), int(scale), int(scale))))
            
            # 4. Calculate tamper score based on mean pixel difference in enhanced image
            # High variance in ELA map suggests tampering
            diff_array = np.asarray(diff)
            mean_diff = np.mean(diff_array)
            std_diff = np.std(diff_array)
            
            # Heuristic score: higher mean/std diff in ELA often implies manipulation
            # (especially in regions that should be uniform)
            tamper_score = min(100.0, (mean_diff * 5.0) + (std_diff * 2.0))
            
            return {
                "tamper_score": round(float(tamper_score), 2),
                "max_diff": max_diff,
                "mean_diff": round(float(mean_diff), 4)
            }
        except Exception as e:
            return {"tamper_score": 0.0, "error": str(e)}
