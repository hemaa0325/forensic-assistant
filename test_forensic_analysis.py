#!/usr/bin/env python3
"""
Test script to demonstrate the 10-criteria forensic image analysis system.
This shows the exact output format you specified.
"""

import os
import sys
from utils.ai_analysis import analyze_image_forensics, get_criteria_descriptions

def test_forensic_analysis():
    """
    Test the forensic analysis system with a sample image.
    """
    print("=== Forensic Image Analysis Test ===\n")
    
    # Display the 10 criteria descriptions
    print("📋 Analysis Criteria:")
    descriptions = get_criteria_descriptions()
    for i, (criterion, description) in enumerate(descriptions.items(), 1):
        print(f"{i:2d}. {criterion}: {description}")
    
    print("\n" + "="*60)
    print("🔍 SAMPLE ANALYSIS OUTPUT FORMAT")
    print("="*60)
    
    # Show the expected output format
    sample_output = """Metadata Consistency: 1 (Timestamp missing)
Software Signature: 0 (No editing software detected)
Noise Pattern: 1 (Different noise levels in background vs object)
Lighting & Shadows: 0 (Consistent lighting)
Edge Artifacts: 1 (Visible halo around subject)
Resolution Uniformity: 0 (No mismatch)
Compression Artifacts: 1 (Double JPEG detected)
Histogram Anomalies: 0 (Normal histogram)
ELA: 1 (Different error levels in tampered region)
Camera Fingerprint: 0 (Consistent sensor pattern)

✅ Total Suspiciousness Score: 5/10

👉 Interpretation:
0–3 points: Low chance of tampering
4–6 points: Medium suspicion
7–10 points: High likelihood of tampering"""
    
    print(sample_output)
    print("\n" + "="*60)
    
    # Check if there are any sample images to test with
    upload_folder = "uploads"
    sample_images = []
    
    if os.path.exists(upload_folder):
        for file in os.listdir(upload_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(upload_folder, file))
    
    if sample_images:
        print(f"\n🖼️ Testing with sample image: {sample_images[0]}")
        try:
            result = analyze_image_forensics(sample_images[0])
            print("\\nActual Analysis Result:")
            print("-" * 40)
            print(result.get("formatted_output", "No formatted output available"))
            
        except Exception as e:
            print(f"Error during analysis: {e}")
    else:
        print("\\n📝 No sample images found in uploads folder.")
        print("To test with a real image:")
        print("1. Place an image file in the uploads/ folder")
        print("2. Run this script again")
        print("3. Or use the web interface to upload an image")

if __name__ == "__main__":
    test_forensic_analysis()