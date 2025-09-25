import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fpdf import FPDF
from datetime import datetime
import os
import io
import re

# Add a helper function for clean text
def clean_text_thoroughly(text):
    """Clean text for PDF while preserving important content"""
    if not text:
        return "N/A"
    text = str(text)
    
    # Remove emojis and special characters
    clean_chars = []
    allowed_chars = set(' .,!?:;-()[]{}"\'\\/\n\r\t%+=_')
    for char in text:
        if char.isalnum() or char in allowed_chars:
            clean_chars.append(char)
        elif ord(char) > 127:  # Replace Unicode with space
            clean_chars.append(' ')
        else:
            clean_chars.append(char)
    
    text = ''.join(clean_chars)
    text = ' '.join(text.split())
    
    return text.strip() if text.strip() else "N/A"

def add_section_header(pdf, title):
    """Add a formatted section header"""
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)  # Blue color
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_draw_color(0, 102, 204)
    y = pdf.get_y()
    pdf.line(10, y, 200, y)  # Adjust line width to fit page
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.set_font("helvetica", size=10)

def add_bullet_point(pdf, text, indent=0):
    """Add a bullet point with proper formatting"""
    if not text or not text.strip():
        return
    
    # Clean the text
    text = clean_text_thoroughly(text)
    
    # Add bullet point and text with proper wrapping
    pdf.set_x(10 + indent)
    pdf.cell(5, 6, "- ", ln=False)
    # Use multi_cell with proper width to avoid overflow
    pdf.multi_cell(180 - indent, 6, text)  # 180 is page width minus margins

def debug_result_structure(result):
    """Debug function to print the structure of the result"""
    print("[DEBUG] Result structure:")
    print(f"  Type: {type(result)}")
    print(f"  Keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {type(value)} = {str(value)[:100]}...")
            if isinstance(value, dict):
                print(f"    Sub-keys: {value.keys()}")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {type(sub_value)} = {str(sub_value)[:50]}...")

def generate_pdf(app, original_filename, result, password):
    reports_dir = app.config['REPORT_FOLDER']
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    locked_path = os.path.join(reports_dir, f"{base_filename}.pdf")

    try:
        # Debug the result structure
        debug_result_structure(result)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)
        pdf.set_margins(10, 10, 10)  # Set margins: left, top, right
        pdf.set_auto_page_break(auto=True, margin=10)  # Enable auto page breaks

        # Title
        pdf.set_font("helvetica", "B", 18)
        pdf.set_text_color(0, 0, 139)  # Dark blue
        pdf.cell(0, 15, "FORENSIC ANALYSIS REPORT", ln=True, align="C")
        pdf.ln(5)
        
        # Report metadata
        pdf.set_font("helvetica", size=9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f"File: {original_filename}", ln=True)
        pdf.cell(0, 6, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)

        # Get data from result - handle different structures
        data = result.get('data', {})
        print(f"[DEBUG] Data from result: {data}")
        
        # Handle different data structures for various file types
        if isinstance(data, dict) and 'basic_analysis' in data:
            # Image analysis structure
            basic_analysis = data.get('basic_analysis', {})
            detailed_analysis = data.get('detailed_analysis', {})
            criteria_breakdown = detailed_analysis.get('criteria_breakdown', {}) if isinstance(detailed_analysis, dict) else {}
        elif isinstance(data, dict) and 'criteria_breakdown' in data:
            # Document analysis structure
            basic_analysis = {
                'classification': result.get('status', 'Unknown'),
                'confidence': result.get('confidence', 'N/A'),
                'details': {
                    'primary_indicators': [result.get('explanation', 'No explanation available')]
                }
            }
            detailed_analysis = data
            criteria_breakdown = data.get('criteria_breakdown', {}) if isinstance(data, dict) else {}
        else:
            # Video or other analysis structures
            basic_analysis = {
                'classification': result.get('status', 'Unknown'),
                'confidence': result.get('confidence', 'N/A'),
                'details': {
                    'primary_indicators': [result.get('explanation', 'No explanation available')]
                }
            }
            detailed_analysis = data if isinstance(data, dict) else {}
            criteria_breakdown = {}
        
        # Get key values with fallbacks
        suspicion_score = 0
        if isinstance(data, dict):
            suspicion_score = data.get('suspicion_score', 0)
        elif isinstance(result, dict):
            suspicion_score = result.get('suspicion_score', 0)
        
        confidence_percentage = result.get('confidence', 'N/A')
        status = result.get('status', 'Unknown')
        
        # SECTION 1: STATUS CLASSIFICATION
        add_section_header(pdf, "1. STATUS CLASSIFICATION")
        
        # Determine status category
        try:
            conf_val = int(str(confidence_percentage).replace('%', ''))
        except:
            conf_val = 100
            
        # Status classification with 80% threshold
        if conf_val < 80:
            status_category = "SUSPICIOUS"
            status_color = (255, 0, 0)  # Red
        else:
            status_category = "AUTHENTIC"
            status_color = (0, 150, 0)  # Green
            
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(status_color[0], status_color[1], status_color[2])
        pdf.cell(0, 8, f"STATUS: {status_category}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        
        pdf.set_font("helvetica", size=10)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        # Clean the text before adding to PDF
        cleaned_confidence = clean_text_thoroughly(str(confidence_percentage))
        pdf.cell(0, 6, f"Confidence Level: {cleaned_confidence}%", ln=True)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        # Clean the text before adding to PDF
        cleaned_suspicion = clean_text_thoroughly(str(suspicion_score))
        pdf.cell(0, 6, f"Suspicion Score: {cleaned_suspicion}/10 points", ln=True)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        # Clean the text before adding to PDF
        cleaned_status = clean_text_thoroughly(str(status))
        pdf.cell(0, 6, f"Classification: {cleaned_status}", ln=True)
        
        # Confidence interpretation
        pdf.ln(3)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "Confidence Interpretation:", ln=True)
        pdf.set_font("helvetica", size=10)
        
        if conf_val >= 90:
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "High confidence (90-100%): Strongly authentic", ln=True)
        elif conf_val >= 80:
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "Medium confidence (80-89%): Likely authentic", ln=True)
        elif conf_val >= 60:
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "Low confidence (60-79%): Suspicious - requires review", ln=True)
        elif conf_val >= 40:
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "High suspicion (40-59%): Likely manipulated", ln=True)
        else:
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "Very high suspicion (0-39%): Highly likely to be fake", ln=True)
        
        # Critical threshold warning
        pdf.ln(2)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(255, 0, 0)  # Red
        pdf.cell(0, 6, "CRITICAL THRESHOLD:", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        pdf.cell(0, 6, "Confidence below 80% = SUSPICIOUS", ln=True)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        pdf.cell(0, 6, "AI detection is the primary criteria", ln=True)
        pdf.ln(5)

        # SECTION 2: ANALYSIS DETAILS
        add_section_header(pdf, "2. ANALYSIS DETAILS")
        
        # Primary classification
        primary_classification = basic_analysis.get('classification', 'N/A')
        basic_confidence = basic_analysis.get('confidence', 'N/A')
        
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 7, "Primary Detection Results:", ln=True)
        pdf.set_font("helvetica", size=10)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        # Clean the text before adding to PDF
        cleaned_classification = clean_text_thoroughly(str(primary_classification))
        pdf.cell(0, 6, f"Classification: {cleaned_classification}", ln=True)
        pdf.set_x(10)
        pdf.cell(5, 6, "- ", ln=False)
        # Clean the text before adding to PDF
        cleaned_basic_confidence = clean_text_thoroughly(str(basic_confidence))
        pdf.cell(0, 6, f"Detection Confidence: {cleaned_basic_confidence}%", ln=True)
        
        # Key indicators
        details = basic_analysis.get('details', {})
        indicators = details.get('primary_indicators', [])
        
        if indicators and isinstance(indicators, list):
            pdf.ln(3)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 6, "Key Detection Indicators:", ln=True)
            pdf.set_font("helvetica", size=10)
            
            # Limit to first 10 indicators to prevent overflow
            for indicator in indicators[:10]:
                pdf.set_x(10)
                pdf.cell(5, 6, "- ", ln=False)
                # Clean the text before adding to PDF
                cleaned_indicator = clean_text_thoroughly(str(indicator))
                pdf.multi_cell(180, 6, cleaned_indicator)  # 180 is page width minus margins
        elif isinstance(indicators, str):
            pdf.ln(3)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 6, "Key Detection Indicators:", ln=True)
            pdf.set_font("helvetica", size=10)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            # Clean the text before adding to PDF
            cleaned_indicators = clean_text_thoroughly(indicators)
            pdf.multi_cell(180, 6, cleaned_indicators)
        
        # Detailed Criteria Breakdown
        pdf.ln(5)
        add_section_header(pdf, "3. DETAILED CRITERIA BREAKDOWN")
        
        if criteria_breakdown and isinstance(criteria_breakdown, dict) and len(criteria_breakdown) > 0:
            for criterion, result_data in list(criteria_breakdown.items())[:20]:  # Limit to 20 criteria
                if isinstance(result_data, dict):
                    score = result_data.get('score', 0)
                    reasoning = result_data.get('reason', result_data.get('reasoning', 'No details'))
                else:
                    score = 0
                    reasoning = str(result_data)
                
                if score == 1:
                    status_text = "SUSPICIOUS"
                    color = (255, 0, 0)  # Red
                else:
                    status_text = "CLEAN"
                    color = (0, 150, 0)  # Green
                
                pdf.set_font("helvetica", "B", 10)
                # Clean the criterion name
                cleaned_criterion = clean_text_thoroughly(str(criterion))
                pdf.cell(0, 6, f"{cleaned_criterion}:", ln=True)
                pdf.set_font("helvetica", size=10)
                pdf.set_text_color(color[0], color[1], color[2])
                pdf.set_x(10)
                pdf.cell(5, 6, "- ", ln=False)
                pdf.cell(0, 6, f"Status: {status_text} (Score: {score})", ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.set_x(10)
                pdf.cell(5, 6, "- ", ln=False)
                # Clean the reasoning text
                cleaned_reasoning = clean_text_thoroughly(str(reasoning))
                pdf.multi_cell(180, 6, f"Reasoning: {cleaned_reasoning}")
                pdf.ln(2)
        else:
            # For video and other analysis types
            if isinstance(detailed_analysis, dict) and detailed_analysis and len(detailed_analysis) > 0:
                displayed_items = 0
                for key, value in detailed_analysis.items():
                    # Skip already shown items
                    if key in ['suspicion_score', 'confidence_percentage', 'criteria_breakdown']:
                        continue
                    if displayed_items >= 20:  # Limit to 20 items
                        break
                        
                    pdf.set_font("helvetica", "B", 10)
                    pdf.cell(0, 6, f"{key}:", ln=True)
                    pdf.set_font("helvetica", size=10)
                    pdf.set_x(10)
                    pdf.cell(5, 6, "- ", ln=False)
                    pdf.multi_cell(180, 6, str(value))
                    pdf.ln(1)
                    displayed_items += 1
                
                if displayed_items == 0:
                    pdf.set_x(10)
                    pdf.cell(5, 6, "- ", ln=False)
                    pdf.multi_cell(180, 6, "No detailed criteria analysis available")
            else:
                pdf.set_x(10)
                pdf.cell(5, 6, "- ", ln=False)
                pdf.multi_cell(180, 6, "No detailed criteria analysis available")
        
        # Additional data section
        if isinstance(data, dict) and len(data) > 0:
            has_shown_data = False
            for key, value in data.items():
                # Skip keys we've already shown
                if key in ['basic_analysis', 'detailed_analysis', 'criteria_breakdown', 'suspicion_score', 'confidence_percentage']:
                    continue
                    
                if not has_shown_data:
                    pdf.ln(5)
                    add_section_header(pdf, "4. ADDITIONAL DATA")
                    has_shown_data = True
                    
                pdf.set_font("helvetica", "B", 10)
                # Clean the key name
                cleaned_key = clean_text_thoroughly(str(key))
                pdf.cell(0, 6, f"{cleaned_key}:", ln=True)
                pdf.set_font("helvetica", size=10)
                
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        pdf.set_x(10)
                        pdf.cell(5, 6, "- ", ln=False)
                        # Clean the sub-key and sub-value
                        cleaned_sub_key = clean_text_thoroughly(str(sub_key))
                        cleaned_sub_value = clean_text_thoroughly(str(sub_value))
                        pdf.multi_cell(180, 6, f"{cleaned_sub_key}: {cleaned_sub_value}")
                else:
                    pdf.set_x(10)
                    pdf.cell(5, 6, "- ", ln=False)
                    # Clean the value
                    cleaned_value = clean_text_thoroughly(str(value))
                    pdf.multi_cell(180, 6, cleaned_value)
                pdf.ln(1)
        
        # Blockchain Verification
        pdf.ln(5)
        add_section_header(pdf, "5. BLOCKCHAIN VERIFICATION")
        
        blockchain_record = result.get('blockchain_record')
        blockchain_verified = result.get('blockchain_verified', False)
        
        if blockchain_verified and blockchain_record:
            pdf.set_text_color(0, 150, 0)  # Green
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 7, "VERIFIED ON BLOCKCHAIN", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", size=10)
            
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, f"Record ID: {blockchain_record.get('record_id', 'N/A')}", ln=True)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, f"Transaction Hash: {blockchain_record.get('transaction_hash', 'N/A')}", ln=True)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, f"Block Number: {blockchain_record.get('block_number', 'N/A')}", ln=True)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, f"File Hash: {blockchain_record.get('file_hash', 'N/A')}", ln=True)
            
            pdf.ln(2)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 6, "Verification URL:", ln=True)
            pdf.set_font("helvetica", size=9)
            verify_url = f"https://mumbai.polygonscan.com/tx/{blockchain_record.get('transaction_hash', '')}"
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.multi_cell(180, 6, verify_url)
            
            pdf.ln(2)
            pdf.set_font("helvetica", size=9)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.multi_cell(180, 6, "This analysis is permanently recorded on Polygon blockchain and cannot be tampered with or altered.")
        else:
            pdf.set_text_color(200, 100, 0)  # Orange
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 7, "LOCAL STORAGE ONLY", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", size=10)
            
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "Analysis not recorded on blockchain", ln=True)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.cell(0, 6, "Results stored locally and may be modified", ln=True)
            
            pdf.ln(2)
            pdf.set_font("helvetica", size=9)
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.multi_cell(180, 6, "Note: Analysis results remain fully valid and accurate")
            pdf.set_x(10)
            pdf.cell(5, 6, "- ", ln=False)
            pdf.multi_cell(180, 6, "Blockchain provides additional tamper-proof verification")

        # Generate PDF with encryption using fpdf2
        pdf.set_encryption(owner_password=password, user_password=password)
        
        # Save PDF directly
        pdf.output(locked_path)
        
        print(f"[SUCCESS] Enhanced PDF with full report generated: {locked_path}")
        return locked_path

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None