#!/usr/bin/env python3
"""
Test script to verify PDF generation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_generator import generate_pdf
from datetime import datetime

class MockApp:
    def __init__(self):
        self.config = {
            'REPORT_FOLDER': 'reports'
        }

def test_pdf_generation():
    """Test PDF generation with sample forensic analysis results."""
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Mock app object
    app = MockApp()
    
    # Sample analysis result with Unicode characters (that should be cleaned)
    sample_result = {
        "status": "Suspicious", 
        "confidence": "75",
        "explanation": """Forensic Analysis Results:

✅ Total Suspiciousness Score: 5/10
📊 Suspicion Level: Medium

Detailed Criteria Analysis:
Metadata Consistency: 1 (Timestamp missing)
Software Signature: 0 (No editing software detected)
Noise Pattern: 1 (Different noise levels in background vs object)
Lighting & Shadows: 0 (Consistent lighting)
Edge Artifacts: 1 (Visible halo around subject)
Resolution Uniformity: 0 (No mismatch)
Compression Artifacts: 1 (Double JPEG detected)
Histogram Anomalies: 0 (Normal histogram)
ELA: 1 (Different error levels in tampered region)
Camera Fingerprint: 0 (Consistent sensor pattern)

👉 Conclusion:
- 0–3 points: Low chance of tampering
- 4–6 points: Medium suspicion  
- 7–10 points: High likelihood of tampering

This image scored 5/10, indicating medium suspicion of digital manipulation.""",
        "data": {
            "suspiciousness_score": "5/10",
            "suspicion_level": "Medium"
        }
    }
    
    # Test PDF generation
    print("🔧 Testing PDF generation...")
    try:
        pdf_path = generate_pdf(app, "test_image.jpg", sample_result, "test123")
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"✅ SUCCESS: PDF generated successfully at: {pdf_path}")
            print(f"📄 File size: {os.path.getsize(pdf_path)} bytes")
            return True
        else:
            print("❌ FAILED: PDF generation returned None or file doesn't exist")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: PDF generation failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("=== PDF Generation Test ===")
    success = test_pdf_generation()
    if success:
        print("\n🎉 PDF generation is working correctly!")
        print("🔗 Your forensic app should now be able to generate encrypted reports.")
    else:
        print("\n💥 PDF generation still has issues that need to be resolved.")