import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from . import image_forensics_tools
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from collections import Counter
import cv2
import io

def check_metadata_consistency(doc) -> Tuple[int, str]:
    """
    Criterion 1: Check document metadata for inconsistencies
    Returns: (score, reasoning)
    """
    try:
        metadata = doc.metadata
        if not metadata:
            return 1, "No metadata found - potentially stripped"
        
        creation_date = metadata.get('creationDate', '')
        mod_date = metadata.get('modDate', '')
        
        # Check if creation date is after modification date
        if creation_date and mod_date:
            try:
                if creation_date > mod_date:
                    return 1, "Creation date is after modification date (suspicious)"
            except:
                pass
        
        # Check for missing critical metadata
        if not metadata.get('creator') and not metadata.get('producer'):
            return 1, "Missing creator/producer information"
            
        return 0, "Metadata appears consistent"
    except:
        return 1, "Failed to read metadata"

def check_font_uniformity(doc) -> Tuple[int, str]:
    """
    Criterion 2: Check for font inconsistencies
    Returns: (score, reasoning)
    """
    try:
        fonts_used = set()
        font_sizes = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_name = span.get("font", "")
                            font_size = span.get("size", 0)
                            fonts_used.add(font_name)
                            font_sizes.append(font_size)
        
        # Check for excessive font variety
        if len(fonts_used) > 5:
            return 1, f"Too many different fonts used ({len(fonts_used)} fonts)"
        
        # Check for unusual font size variations
        if font_sizes:
            size_std = np.std(font_sizes)
            if size_std > 3.0:
                return 1, f"High font size variation (std: {size_std:.1f})"
        
        return 0, "Font usage appears consistent"
    except:
        return 0, "Could not analyze fonts"

def check_text_alignment_spacing(doc) -> tuple[int, str]:
    """
    Criterion 3: Check for text alignment and spacing issues
    Returns: (score, reasoning)
    """
    try:
        alignment_issues = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            line_positions = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if line["spans"]:
                            x0 = line["bbox"][0]
                            line_positions.append(x0)
            
            # Check for irregular left margins
            if len(line_positions) > 10:
                position_groups = Counter([round(pos) for pos in line_positions])
                if len(position_groups) > 5:  # Too many different starting positions
                    alignment_issues += 1
        
        if alignment_issues > 1:
            return 1, f"Irregular text alignment detected in {alignment_issues} pages"
        
        return 0, "Text alignment appears normal"
    except:
        return 0, "Could not analyze text alignment"

def check_copy_paste_artifacts(doc) -> tuple[int, str]:
    """
    Criterion 4: Check for copy-paste artifacts
    Returns: (score, reasoning)
    """
    try:
        artifacts_found = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Check for embedded images that might indicate copy-paste
            images = page.get_images()
            if len(images) > 5:  # Excessive embedded images
                artifacts_found += 1
            
            # Check for text blocks with suspicious positioning
            blocks = page.get_text("dict")["blocks"]
            overlapping_blocks = 0
            
            for i, block1 in enumerate(blocks):
                for j, block2 in enumerate(blocks[i+1:], i+1):
                    if "bbox" in block1 and "bbox" in block2:
                        # Check for overlapping bounding boxes
                        bbox1, bbox2 = block1["bbox"], block2["bbox"]
                        if (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and 
                            bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]):
                            overlapping_blocks += 1
            
            if overlapping_blocks > 2:
                artifacts_found += 1
        
        if artifacts_found > 0:
            return 1, f"Copy-paste artifacts detected in {artifacts_found} areas"
        
        return 0, "No copy-paste artifacts detected"
    except:
        return 0, "Could not analyze copy-paste artifacts"

def check_signature_seal_integrity(doc) -> tuple[int, str]:
    """
    Criterion 5: Check signature/seal integrity
    Returns: (score, reasoning)
    """
    try:
        suspicious_images = 0
        
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Check image dimensions - signatures/seals are often small
                    width, height = pil_image.size
                    if width < 200 and height < 200:
                        # Convert to numpy array for analysis
                        img_array = np.array(pil_image)
                        
                        # Check for pixelation (low quality signatures)
                        if len(img_array.shape) == 3:
                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = img_array
                        
                        # Check edge sharpness
                        edges = cv2.Canny(gray, 50, 150)
                        edge_ratio = np.sum(edges > 0) / edges.size
                        
                        if edge_ratio < 0.05:  # Too few edges (pixelated)
                            suspicious_images += 1
                            
                except:
                    continue
        
        if suspicious_images > 0:
            return 1, f"Found {suspicious_images} suspicious signatures/seals (pixelated or low quality)"
        
        return 0, "Signatures/seals appear genuine"
    except:
        return 0, "Could not analyze signatures/seals"

def check_color_tone_consistency(doc) -> tuple[int, str]:
    """
    Criterion 6: Check color/tone consistency
    Returns: (score, reasoning)
    """
    try:
        page_colors = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract page as image
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Analyze dominant colors
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                mean_color = np.mean(img_array, axis=(0, 1))
                page_colors.append(mean_color)
        
        if len(page_colors) > 1:
            # Check color consistency across pages
            color_std = np.std(page_colors, axis=0)
            if np.mean(color_std) > 15:  # High color variation
                return 1, f"Inconsistent color/tone across pages (variation: {np.mean(color_std):.1f})"
        
        return 0, "Color/tone appears consistent"
    except:
        return 0, "Could not analyze color consistency"

def check_compression_layer_artifacts(doc) -> tuple[int, str]:
    """
    Criterion 7: Check for compression/layer artifacts
    Returns: (score, reasoning)
    """
    try:
        compression_issues = 0
        
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Check compression type and quality
                    if base_image.get("ext") == "jpg":
                        # JPEG compression analysis
                        image_bytes = base_image["image"]
                        if len(image_bytes) < 5000:  # Very small file size indicates high compression
                            compression_issues += 1
                            
                except:
                    continue
        
        # Check for signs of multiple saves/compressions
        if compression_issues > 2:
            return 1, f"Multiple compression artifacts detected ({compression_issues} images)"
        
        return 0, "No significant compression artifacts"
    except:
        return 0, "Could not analyze compression"

def check_ocr_consistency(filepath) -> tuple[int, str]:
    """
    Criterion 8: Check OCR consistency
    Returns: (score, reasoning)
    """
    try:
        # Convert PDF to images for OCR
        images = convert_from_path(filepath, dpi=150)
        ocr_issues = 0
        
        for i, page_image in enumerate(images[:3]):  # Check first 3 pages
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(page_image)
            
            # Check for OCR confidence
            ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            
            if confidences:
                avg_confidence = np.mean(confidences)
                if avg_confidence < 60:  # Low OCR confidence suggests poor quality or tampering
                    ocr_issues += 1
        
        if ocr_issues > 1:
            return 1, f"OCR inconsistencies detected in {ocr_issues} pages (low confidence)"
        
        return 0, "OCR text appears consistent"
    except:
        return 0, "Could not perform OCR analysis"

def check_background_texture_uniformity(doc) -> tuple[int, str]:
    """
    Criterion 9: Check background/texture uniformity
    Returns: (score, reasoning)
    """
    try:
        texture_variations = []
        
        for page_num in range(min(3, len(doc))):  # Check first 3 pages
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Convert to grayscale and analyze texture
            img_array = np.array(pil_image.convert('L'))
            
            # Calculate local standard deviation (texture measure)
            kernel_size = 20
            h, w = img_array.shape
            
            texture_values = []
            for i in range(0, h-kernel_size, kernel_size):
                for j in range(0, w-kernel_size, kernel_size):
                    patch = img_array[i:i+kernel_size, j:j+kernel_size]
                    texture_values.append(np.std(patch))
            
            if texture_values:
                texture_variations.append(np.std(texture_values))
        
        if texture_variations:
            avg_variation = np.mean(texture_variations)
            if avg_variation > 25:  # High texture variation suggests tampering
                return 1, f"Inconsistent background texture (variation: {avg_variation:.1f})"
        
        return 0, "Background texture appears uniform"
    except:
        return 0, "Could not analyze background texture"

def check_version_history(doc) -> tuple[int, str]:
    """
    Criterion 10: Check version history (if available)
    Returns: (score, reasoning)
    """
    try:
        metadata = doc.metadata
        if not metadata:
            return 0, "No version history available"
        
        # Check for suspicious modification patterns
        creator = metadata.get('creator', '')
        producer = metadata.get('producer', '')
        
        # Check for multiple software signatures
        suspicious_software = ['adobe', 'foxit', 'pdfcreator', 'libreoffice']
        software_count = sum(1 for software in suspicious_software 
                           if software.lower() in (creator + producer).lower())
        
        if software_count > 1:
            return 1, f"Multiple editing software detected: {creator}, {producer}"
        
        # Check creation vs modification time gap
        creation_date = metadata.get('creationDate', '')
        mod_date = metadata.get('modDate', '')
        
        if creation_date and mod_date and creation_date != mod_date:
            return 1, "Document shows modification after creation"
        
        return 0, "Version history appears normal"
    except:
        return 0, "Could not analyze version history"

def analyze(filepath):
    """
    Comprehensive 10-criteria document forensic analysis
    """
    try:
        doc = fitz.open(filepath)
        
        # Initialize criteria results
        criteria_results = {}
        total_score = 0
        
        # Run all 10 criteria checks
        criteria_checks = [
            ("Metadata Consistency", check_metadata_consistency(doc)),
            ("Font Uniformity", check_font_uniformity(doc)),
            ("Text Alignment & Spacing", check_text_alignment_spacing(doc)),
            ("Copy-Paste Artifacts", check_copy_paste_artifacts(doc)),
            ("Signature/Seal Integrity", check_signature_seal_integrity(doc)),
            ("Color/Tone Consistency", check_color_tone_consistency(doc)),
            ("Compression/Layer Artifacts", check_compression_layer_artifacts(doc)),
            ("OCR Consistency", check_ocr_consistency(filepath)),
            ("Background/Texture Uniformity", check_background_texture_uniformity(doc)),
            ("Version History", check_version_history(doc))
        ]
        
        # Process results
        for criterion_name, (score, reasoning) in criteria_checks:
            criteria_results[criterion_name] = {
                'score': score,
                'reasoning': reasoning
            }
            total_score += score
        
        # Generate detailed explanation with ultra-prominent suspicion score
        confidence_value = max(10, 100 - (total_score * 10))
        explanation = f"""COMPREHENSIVE DOCUMENT FORENSIC ANALYSIS

ULTRA-PROMINENT SUSPICION SCORE
[TOTAL SUSPICION SCORE: {total_score}/10 POINTS]
[CONFIDENCE: {confidence_value}%] (Every point = -10% confidence)

=== DETAILED SUSPICION BREAKDOWN ===
[TOTAL SUSPICION SCORE: {total_score}/10 POINTS]
[FINAL CONFIDENCE: {confidence_value}%]

Detailed Criteria Breakdown:
"""
        
        for criterion, result in criteria_results.items():
            score_icon = "[+1 POINT]" if result['score'] == 1 else "[0 POINTS]"
            explanation += f"{score_icon} {criterion}: {result['score']} point - {result['reasoning']}\n"
        
        explanation += f"""
=== CONFIDENCE CALCULATION ===
• Base Confidence: 100%
• Suspicion Points: -{total_score} points
• Each Point: -10% confidence
• [FINAL CONFIDENCE: {confidence_value}%]

=== INTERPRETATION ===
• 0-3 points: Low suspicion (70-100% confidence)
• 4-6 points: Medium suspicion (40-60% confidence)  
• 7-10 points: High likelihood of forgery/tampering (10-30% confidence)

[FINAL ASSESSMENT]:"""
        
        # === DETERMINE FINAL STATUS WITH 80% SUSPICIOUS THRESHOLD ===
        # Every suspicion point reduces confidence by 10%
        confidence_value = max(10, 100 - (total_score * 10))
        confidence = str(confidence_value)
        
        # ENHANCED STATUS DETERMINATION - ANYTHING BELOW 80% IS SUSPICIOUS
        if confidence_value < 80:  # NEW SUSPICIOUS THRESHOLD
            if total_score >= 7:
                status = "HIGHLY SUSPICIOUS - Likely Tampered"
                explanation += f" Document shows CRITICAL tampering indicators ({total_score}/10 points). HIGH RISK of forgery."
            elif total_score >= 4:
                status = "SUSPICIOUS - Evidence of Manipulation"
                explanation += f" Document shows SIGNIFICANT manipulation indicators ({total_score}/10 points). MEDIUM RISK - Manual review required."
            else:
                status = "SUSPICIOUS - Minor Indicators"
                explanation += f" Document shows MINOR suspicious indicators ({total_score}/10 points). LOW-MEDIUM RISK - Review recommended."
        else:
            # Only mark as authentic if confidence >= 80%
            status = "Authentic"
            explanation += f" Document appears authentic with minimal suspicious indicators ({total_score}/10 points). LOW RISK."
        
        explanation += f"""

[FINAL DOCUMENT ASSESSMENT: {status}]
[SUSPICION SCORE: {total_score}/10 POINTS]
[CONFIDENCE: {confidence_value}%]
This document shows a detailed forensic suspicion score of {total_score}/10 points."""
        
        return {
            "status": status,
            "confidence": confidence,
            "explanation": explanation,
            "data": {
                "suspicion_score": total_score,
                "suspicion_points": f"{total_score}/10",
                "confidence_percentage": confidence_value,
                "total_suspiciousness_score": f"{total_score}/10",
                "criteria_breakdown": criteria_results,
                "page_count": len(doc),
                "is_encrypted": doc.is_encrypted,
                "analysis_type": "Comprehensive 10-Criteria Forensic Analysis"
            }
        }
        
    except Exception as e:
        return {
            "status": "Error",
            "confidence": "0",
            "explanation": f"Document analysis failed. File may be corrupt or invalid. Error: {e}",
            "data": {}
        }