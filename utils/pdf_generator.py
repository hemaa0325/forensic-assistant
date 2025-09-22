from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import os
import io
import re

def clean_text_thoroughly(text):
    """Clean text for PDF while preserving important content"""
    if not text:
        return "N/A"
    text = str(text)
    
    # Step 1: Replace emojis with text equivalents
    emoji_replacements = {
        '🤖': 'AI Generated',
        '✂️': 'Manually Edited', 
        '✅': 'Authentic',
        '⚠️': 'Suspicious',
        '🚨': 'Tampered',
        '🎯': 'Score',
        '📊': 'Analysis',
        '🔍': 'Detection',
        '📋': 'Document',
        '📈': 'Chart',
        '📄': 'Report',
        '🔧': 'Tool',
        '💡': 'Info',
        '⚡': 'Fast',
        '🌊': 'Wave',
        '🎨': 'Art',
        '🌈': 'Color'
    }
    
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)
    
    # Step 2: Remove other Unicode symbols and emojis but keep letters, numbers, spaces, punctuation
    # Keep: letters, numbers, spaces, basic punctuation
    clean_chars = []
    allowed_chars = set(' .,!?:;-()[]{}"\'\\/\n\r\t%+=')
    for char in text:
        if char.isalnum() or char in allowed_chars:
            clean_chars.append(char)
        elif ord(char) > 127:  # Replace Unicode with space
            clean_chars.append(' ')
        else:
            clean_chars.append(char)
    
    text = ''.join(clean_chars)
    
    # Step 3: Clean up excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip() if text.strip() else "N/A"

def generate_pdf(app, original_filename, result, password):
    reports_dir = app.config['REPORT_FOLDER']
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    locked_path = os.path.join(reports_dir, f"{base_filename}.pdf")

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)

        # Title
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "FORENSIC ANALYSIS REPORT", ln=True, align="C")
        pdf.ln(10)

        # === ULTRA-PROMINENT SUSPICION SCORE SECTION ===
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(200, 0, 0)  # Red color
        pdf.cell(0, 10, "=== SUSPICION SCORE ===", ln=True, align="C")
        pdf.set_text_color(0, 0, 0)  # Back to black
        
        # Get suspicion score from data
        suspicion_score = "N/A"
        confidence_percentage = result.get('confidence', 'N/A')
        data = result.get('data', {})
        
        if 'suspicion_score' in data:
            suspicion_score = f"{data['suspicion_score']}/10"
        elif 'suspicion_points' in data:
            suspicion_score = data['suspicion_points']
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, f"TOTAL SUSPICION SCORE: {suspicion_score} POINTS", ln=True, align="C")
        pdf.cell(0, 8, f"CONFIDENCE: {confidence_percentage}% (Every point = -10% confidence)", ln=True, align="C")
        pdf.ln(5)

        # Basic Analysis Results
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "ANALYSIS RESULTS", ln=True)
        pdf.ln(3)

        # Status
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(30, 6, "Status: ")
        pdf.set_font("helvetica", size=10)
        status = clean_text_thoroughly(result.get('status', 'Unknown'))
        pdf.cell(0, 6, status, ln=True)

        # Confidence
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(30, 6, "Confidence: ")
        pdf.set_font("helvetica", size=10)
        pdf.cell(0, 6, f"{confidence_percentage}%", ln=True)

        # File info
        pdf.ln(3)
        pdf.set_font("helvetica", size=9)
        pdf.cell(0, 5, f"Original Filename: {original_filename}", ln=True)
        pdf.cell(0, 5, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)

        # === DETAILED EXPLANATION ===
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "DETAILED FORENSIC ANALYSIS", ln=True)
        pdf.ln(3)
        
        # Get the main explanation text
        explanation = result.get('explanation', 'No explanation available')
        
        # Clean and prepare explanation
        explanation_clean = clean_text_thoroughly(explanation)
        
        # Remove section headers that might cause formatting issues
        lines_to_remove = [
            'ULTRA-PROMINENT SUSPICION SCORE',
            'DETAILED SUSPICION BREAKDOWN',
            'CONFIDENCE CALCULATION',
            'INTERPRETATION',
            'BASIC LEVEL CLASSIFICATION',
            'FINAL ASSESSMENT'
        ]
        
        for line in lines_to_remove:
            explanation_clean = explanation_clean.replace(line, '')
        
        # Split explanation into manageable chunks
        explanation_lines = explanation_clean.split('\n')
        
        pdf.set_font("helvetica", size=9)
        for line in explanation_lines:
            if line.strip():
                # Remove multiple spaces and clean the line
                line = ' '.join(line.split())
                if len(line) > 100:
                    # Split long lines
                    pdf.multi_cell(0, 4, line)
                else:
                    pdf.cell(0, 4, line, ln=True)
        
        pdf.ln(3)
        
        # === CRITERIA BREAKDOWN ===
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 7, "SUSPICION CRITERIA BREAKDOWN", ln=True)
        pdf.ln(2)
        
        # Extract criteria breakdown from data
        criteria_breakdown = data.get('criteria_breakdown', {})
        if not criteria_breakdown:
            detailed_analysis = data.get('detailed_analysis', {})
            criteria_breakdown = detailed_analysis.get('criteria_breakdown', {})
        
        if criteria_breakdown:
            pdf.set_font("helvetica", size=9)
            for criterion, result_data in criteria_breakdown.items():
                score = result_data.get('score', 0)
                reasoning = result_data.get('reason', result_data.get('reasoning', 'No details'))
                
                # Simple, clean formatting
                if score == 1:
                    status = "SUSPICIOUS"
                else:
                    status = "CLEAN"
                
                criterion_clean = clean_text_thoroughly(criterion)
                reasoning_clean = clean_text_thoroughly(reasoning)
                
                # Keep it simple
                line = f"{criterion_clean}: {status} - {reasoning_clean}"
                
                if len(line) > 85:
                    pdf.multi_cell(0, 4, line)
                    pdf.ln(1)
                else:
                    pdf.cell(0, 4, line, ln=True)
        
        pdf.ln(3)
        
        # === KEY FINDINGS ===
        basic_analysis = data.get('basic_analysis', {})
        if basic_analysis:
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 7, "KEY DETECTION FINDINGS", ln=True)
            pdf.set_font("helvetica", size=9)
            
            classification = clean_text_thoroughly(basic_analysis.get('classification', 'N/A'))
            basic_confidence = basic_analysis.get('confidence', 'N/A')
            
            pdf.cell(0, 5, f"Primary Detection: {classification}", ln=True)
            pdf.cell(0, 5, f"Detection Confidence: {basic_confidence}%", ln=True)
            
            # Key indicators - simplified
            details = basic_analysis.get('details', {})
            indicators = details.get('primary_indicators', [])
            if indicators:
                pdf.ln(2)
                pdf.set_font("helvetica", "B", 9)
                pdf.cell(0, 5, "Main Indicators:", ln=True)
                pdf.set_font("helvetica", size=8)
                for i, indicator in enumerate(indicators[:3]):  # Show only top 3
                    clean_indicator = clean_text_thoroughly(str(indicator))
                    # Keep indicators short and clean
                    if len(clean_indicator) > 60:
                        clean_indicator = clean_indicator[:60] + "..."
                    pdf.cell(0, 4, f"- {clean_indicator}", ln=True)
        
        pdf.ln(5)
        
        # === ANALYSIS SUMMARY ===
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 7, "ANALYSIS SUMMARY", ln=True)
        pdf.set_font("helvetica", size=9)
        
        # Create a clean summary
        summary_lines = [
            f"File analyzed: {original_filename}",
            f"Final status: {status}",
            f"Confidence level: {confidence_percentage}%",
            f"Suspicion score: {suspicion_score} points",
            "",
            "Score interpretation:",
            "- 0-3 points: Low suspicion",
            "- 4-6 points: Medium suspicion", 
            "- 7-10 points: High suspicion",
            "",
            "Confidence calculation:",
            f"- Base: 100% - ({suspicion_score} x 10%) = {confidence_percentage}%"
        ]
        
        for line in summary_lines:
            if line:
                pdf.cell(0, 4, line, ln=True)
            else:
                pdf.ln(2)

        # Generate PDF bytes
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin1')

        # Encrypt PDF
        pdf_buffer = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_buffer)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        writer.encrypt(user_password=password)

        with open(locked_path, "wb") as f:
            writer.write(f)

        print(f"[SUCCESS] Enhanced PDF with full report generated: {locked_path}")
        return locked_path

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None