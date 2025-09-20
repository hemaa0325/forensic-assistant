import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from . import image_forensics_tools
import numpy as np

DOC_ANOMALY_SCORES = {
    "ocr_misaligned": {"penalty": 35, "finding": "misaligned text detected (potential copy-paste)"},
    "meta_software": {"penalty": 10, "finding": "metadata indicates creator software"},
    "meta_dates": {"penalty": 20, "finding": "metadata shows different creation/modification dates"},
    "embedded_ela": {"penalty": 25, "finding": "embedded image shows inconsistent compression (ELA)"},
    "embedded_clone": {"penalty": 35, "finding": "embedded image contains cloned regions"}
}

def analyze_layout_with_ocr(filepath):
    try:
        images = convert_from_path(filepath)
        for i, page_image in enumerate(images):
            data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
            lines = {}
            for j, text in enumerate(data['text']):
                if text.strip():
                    line_num = data['line_num'][j]
                    if line_num not in lines: lines[line_num] = []
                    lines[line_num].append(data['top'][j])

            for line_num, tops in lines.items():
                if len(tops) > 1 and np.std(tops) > 2:
                    return ["ocr_misaligned"]
    except Exception:
        return ["ocr_failed"] 
    return []

def analyze(filepath):
    anomaly_keys = []
    details = {}
    
    try:
        anomaly_keys.extend(analyze_layout_with_ocr(filepath))

        doc = fitz.open(filepath)
        metadata = doc.metadata
        if metadata:
            if metadata.get('creator') or metadata.get('producer'):
                anomaly_keys.append("meta_software")
            if metadata.get('creationDate') != metadata.get('modDate'):
                anomaly_keys.append("meta_dates")
        
        details['Page Count'] = doc.page_count
        details['Is Encrypted'] = doc.is_encrypted # <-- THIS LINE WAS MISSING

        for page_num in range(len(doc)):
            for img in doc.get_page_images(page_num):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if image_forensics_tools.perform_ela(image_bytes):
                    anomaly_keys.append("embedded_ela")
                if image_forensics_tools.detect_cloned_regions(image_bytes):
                    anomaly_keys.append("embedded_clone")

        full_text = "".join([page.get_text() for page in doc])
        if re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', full_text): details['PAN Card Found'] = 'Yes'
        if re.search(r'\d{4}\s\d{4}\s\d{4}', full_text): details['Aadhaar Card Found'] = 'Yes'

        final_confidence = 100
        findings = []
        for key in set(anomaly_keys):
            anomaly = DOC_ANOMALY_SCORES.get(key)
            if anomaly:
                final_confidence -= anomaly["penalty"]
                findings.append(anomaly["finding"])

        final_confidence = max(10, final_confidence)

        if findings:
            explanation = "Potential forgery detected: " + ", ".join(sorted(findings)) + "."
            return {"status": "Suspicious", "confidence": str(final_confidence), "explanation": explanation, "data": details}
        else:
            explanation = "No major structural or content anomalies detected."
            return {"status": "Authentic", "confidence": "90", "explanation": explanation, "data": details}

    except Exception as e:
        return {"status": "Error", "confidence": "0", "explanation": f"Document analysis failed. Error: {e}", "data": {}}