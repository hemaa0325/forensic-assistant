from PIL import Image, ExifTags
import os
import io
from enum import Enum
from typing import Dict, List, Tuple

from . import ai_analysis
from . import image_forensics_tools

# Change 1: Use a formal Enum for anomaly keys to prevent typos
class AnomalyKeys(Enum):
    AI_GENERATED = "ai_generated_image_detected"
    CLONE_DETECTED = "clone_detected"
    AI_MODEL_UNAVAILABLE = "ai_model_unavailable"
    AI_ANALYSIS_FAILED = "ai_analysis_failed"
    ELA_INCONSISTENT = "ela_inconsistent_compression"
    EXIF_DATES = "exif_dates"
    EXIF_SOFTWARE = "exif_software"

# Use the Enum members as keys for safety and clarity
IMAGE_ANOMALY_SCORES = {
    AnomalyKeys.AI_GENERATED: {"penalty": 70, "finding": "high probability of being AI-generated (ViT model)"},
    AnomalyKeys.CLONE_DETECTED: {"penalty": 40, "finding": "contains duplicated regions (geometric check)"},
    AnomalyKeys.AI_MODEL_UNAVAILABLE: {"penalty": 30, "finding": "AI model was unavailable for analysis"},
    AnomalyKeys.AI_ANALYSIS_FAILED: {"penalty": 30, "finding": "AI analysis failed during execution"},
    AnomalyKeys.ELA_INCONSISTENT: {"penalty": 15, "finding": "shows inconsistent compression (ELA)"},
    AnomalyKeys.EXIF_DATES: {"penalty": 15, "finding": "has inconsistent timestamps"},
    AnomalyKeys.EXIF_SOFTWARE: {"penalty": 5, "finding": "image edited with software"}
}

# Change 2: Add Type Hinting for professional-grade code
def analyze_exif(image: Image.Image) -> Tuple[List[AnomalyKeys], Dict[str, str]]:
    anomaly_keys, details = [], {}
    try:
        exif_data_raw = image._getexif() if hasattr(image, '_getexif') else None
        if not exif_data_raw: return anomaly_keys, details
        exif_data = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data_raw.items()}
        
        software = exif_data.get('Software')
        if software:
            anomaly_keys.append(AnomalyKeys.EXIF_SOFTWARE)
            details['Editing Software'] = str(software)
        if exif_data.get('GPSInfo'): 
            details['GPS Info Found'] = 'Yes'
        
        original_time, digitized_time = exif_data.get('DateTimeOriginal'), exif_data.get('DateTimeDigitized')
        if original_time and digitized_time and original_time != digitized_time:
            anomaly_keys.append(AnomalyKeys.EXIF_DATES)
    except Exception: pass
    return anomaly_keys, details

def analyze(filepath: str) -> Dict:
    try:
        with open(filepath, 'rb') as f: image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # --- 1. Run All Forensic Tests ---
        anomaly_keys, details = [], {}
        exif_anomalies, exif_details = analyze_exif(image)
        anomaly_keys.extend(exif_anomalies)
        details.update(exif_details)
        
        if image_forensics_tools.perform_ela(image_bytes): 
            anomaly_keys.append(AnomalyKeys.ELA_INCONSISTENT)
        if image_forensics_tools.detect_cloned_regions(image_bytes): 
            anomaly_keys.append(AnomalyKeys.CLONE_DETECTED)
        
        ai_key_str = ai_analysis.detect_synthesis_artifacts(filepath)
        if ai_key_str: 
            anomaly_keys.append(AnomalyKeys(ai_key_str)) # Convert string to Enum member

        # --- 2. Calculate Dynamic Score ---
        final_confidence, findings_text = 100, []
        for key in set(anomaly_keys):
            anomaly = IMAGE_ANOMALY_SCORES.get(key)
            if anomaly:
                final_confidence -= anomaly["penalty"]
                findings_text.append(anomaly["finding"])
        final_confidence = max(10, final_confidence)

        # --- 3. Formulate Final Verdict ---
        if findings_text:
            explanation = "Potential tampering detected: " + ", ".join(sorted(findings_text)) + "."
            return { "status": "Suspicious", "confidence": str(final_confidence), "explanation": explanation, "data": details }
        else:
            return { "status": "Authentic", "confidence": "95", "explanation": "No significant signs of digital tampering were found.", "data": details }
    except Exception as e:
        return { "status": "Error", "confidence": "0", "explanation": f"Image analysis failed. File may be corrupt or invalid. Error: {e}", "data": {} }