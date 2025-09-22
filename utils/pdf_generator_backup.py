from PyPDF2 import PdfReader, PdfWriter
from fpdf import FPDF
from datetime import datetime
import os
import io

def clean_unicode_text(text):
    """Clean ALL Unicode characters for PDF compatibility"""
    if not text:
        return "N/A"
    
    text = str(text)
    
    # First pass: Replace known emojis and symbols with ASCII
    unicode_map = {
        '✅': '[OK]', '🤖': 'AI', '✂️': 'MANUAL', '⚡': '', '🎯': 'TARGET',
        '🚨': 'ALERT', '⚠️': 'WARNING', '❌': '[ERROR]', '🔍': 'SEARCH',
        '💻': 'COMPUTER', '🛠️': 'TOOL', '🧠': 'BRAIN', '🌈': 'RAINBOW',
        '🌊': 'WAVE', '📷': 'CAMERA', '🎨': 'ART', '📈': 'CHART',
        '🧮': 'MATH', '🏆': 'TROPHY', '🖥️': 'COMPUTER', '💼': 'BRIEFCASE',
        '🔮': 'CRYSTAL', '💭': 'THOUGHT', '✨': 'SPARKLE', '🚫': 'PROHIBITED',
        '🔴': 'RED', '🔵': 'BLUE', '🟢': 'GREEN', '🟡': 'YELLOW',
        '📊': '[CHART]', '👉': '=>', '–': '-', '—': '-', '✂': 'MANUAL',
        '🎵': 'MUSIC', '📝': 'NOTE', '🔧': 'TOOL', '⭐': 'STAR',
        '🎓': 'EDUCATION', '🔬': 'SCIENCE', '📺': 'TV', '📱': 'PHONE',
        '💡': 'IDEA', '🚀': 'ROCKET', '🏠': 'HOME', '🚗': 'CAR'
    }
    
    # Replace known Unicode characters
    for unicode_char, replacement in unicode_map.items():
        text = text.replace(unicode_char, replacement)
    
    # Second pass: Remove ALL remaining Unicode characters by encoding to ASCII
    # This will remove any emoji or special character we missed
    try:
        # Convert to bytes, ignore Unicode chars, then back to string
        text = text.encode('ascii', 'ignore').decode('ascii')
    except:
        # If that fails, manually filter printable ASCII
        text = ''.join(char for char in text if ord(char) < 128 and char.isprintable() or char.isspace())
    
    return text if text.strip() else "N/A"

def generate_pdf(app, original_filename, result, password):
    reports_dir = app.config['REPORT_FOLDER']
    
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    locked_path = os.path.join(reports_dir, f"{base_filename}.pdf")

    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Use built-in fonts instead of Arial to avoid deprecation warnings
        pdf.set_font("helvetica", size=12)

        pdf.set_font("helvetica", "B")
        pdf.cell(0, 10, "FORENSIC ANALYSIS REPORT", ln=True, align="C")
        pdf.ln(5)
        
        # Extract and clean ALL data recursively
        status = clean_unicode_text(result.get('status', 'Unknown'))
        confidence = clean_unicode_text(result.get('confidence', 'N/A'))
        
        # Clean nested data structures
        data_section = result.get('data', {})
        if isinstance(data_section, dict):
            # Clean prominent scores from data section
            for key in ['prominent_score', 'ultra_prominent_display']:
                if key in data_section:
                    data_section[key] = clean_unicode_text(data_section[key])
        
        # Add ultra-prominent suspicion score section
        pdf.set_font("helvetica", "B", 18)
        pdf.set_text_color(220, 20, 20)  # Red color for prominence
        pdf.cell(0, 15, "=== SUSPICION SCORE ===", ln=True, align="C")
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(0, 0, 0)  # Back to black
            
        # Display the prominent score
        score_text = f"CONFIDENCE: {confidence}% | STATUS: {status}"
        pdf.cell(0, 12, score_text, ln=True, align="C")
        pdf.ln(5)
        pdf.ln(5)

        pdf.set_font("helvetica")
        pdf.cell(0, 8, f"Original Filename: {original_filename}", ln=True)
        pdf.cell(0, 8, f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}", ln=True)
        pdf.ln(5)

        pdf.set_font("helvetica", "B")
        pdf.cell(40, 8, "Status:")
        pdf.set_font("helvetica")
        pdf.cell(0, 8, status, ln=True)

        pdf.set_font("helvetica", "B")
        pdf.cell(40, 8, "Confidence:")
        pdf.set_font("helvetica")
        pdf.cell(0, 8, f"{confidence}%", ln=True)

        pdf.set_font("helvetica", "B")
        pdf.cell(40, 8, "Explanation:")
        pdf.set_font("helvetica")
        
        # Clean explanation text
        explanation = clean_unicode_text(result.get('explanation', 'N/A'))
        pdf.multi_cell(0, 8, explanation)
        
        # Generate PDF as bytes for encryption
        try:
            # For fpdf2, output() returns bytes by default
            pdf_output = pdf.output()
            if isinstance(pdf_output, bytes):
                pdf_bytes = pdf_output
            elif isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin1')
            else:
                # Fallback
                pdf_bytes = bytes(pdf_output)
        except Exception as e:
            print(f"[ERROR] PDF output error: {e}")
            return None
        
        pdf_buffer = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_buffer)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        writer.encrypt(user_password=password)

        with open(locked_path, "wb") as f:
            writer.write(f)
        
        print(f"[SUCCESS] Encrypted PDF successfully saved to: {locked_path}")
        return locked_path

    except Exception as e:
        print(f"[ERROR] An error occurred during PDF generation/encryption: {e}")
        return None