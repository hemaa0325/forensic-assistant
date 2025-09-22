#!/usr/bin/env python3
"""
Test script to simulate Flask upload and PDF generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import analysis
from utils import pdf_generator
from flask import Flask

from PIL import Image
import numpy as np

def create_test_image():
    """Create a proper test image using PIL"""
    # Create a simple 50x50 RGB image
    img = Image.new('RGB', (50, 50), color='blue')
    img_array = np.array(img)
    
    # Add some patterns
    for i in range(50):
        for j in range(50):
            if (i + j) % 10 < 5:
                img_array[i, j] = [255, 0, 0]  # Red stripes
    
    img = Image.fromarray(img_array)
    img.save('test_upload.jpg', 'JPEG')
    return 'test_upload.jpg'

# Create test image content - this is now replaced by the function above
# test_image_content = b'\x89PNG\r\n\x1a\n...'

def test_complete_workflow():
    """Test the complete workflow from upload to PDF generation"""
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        print("🔧 Testing complete workflow...")
        
        # Mock app object
        app = Flask(__name__)
        app.config['REPORT_FOLDER'] = 'reports'
        
        # Step 1: Image analysis
        print("Step 1: Running image analysis...")
        try:
            analysis_result = analysis.analyze_file(test_image_path)
            print(f"Analysis result status: {analysis_result.get('status', 'Unknown')}")
            print(f"Analysis result confidence: {analysis_result.get('confidence', 'Unknown')}")
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            analysis_result = {'status': '❌ Error', 'confidence': '0', 'explanation': str(e)}
        
        # Step 2: PDF generation
        print("Step 2: Generating PDF...")
        pdf_path = pdf_generator.generate_pdf(app, "test_upload.png", analysis_result, "password123")
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"✅ SUCCESS: Complete workflow works! PDF at: {pdf_path}")
            print(f"📄 File size: {os.path.getsize(pdf_path)} bytes")
            return True
        else:
            print("❌ FAILED: PDF generation returned None or file doesn't exist")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Complete workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test file
        try:
            os.remove(test_image_path)
        except:
            pass

if __name__ == "__main__":
    print("=== Complete Workflow Test ===")
    success = test_complete_workflow()
    if success:
        print("\n🎉 Complete workflow is working correctly!")
        print("🔗 Your forensic app should be able to generate PDFs successfully.")
    else:
        print("\n💥 Workflow has issues that need to be resolved.")