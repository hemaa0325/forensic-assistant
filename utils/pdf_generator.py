from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import os
import io

def generate_pdf(app, original_filename, result, password):
    reports_dir = app.config['REPORT_FOLDER']
    
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    locked_path = os.path.join(reports_dir, f"{base_filename}.pdf")

    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Helvetica", size=12)

        pdf.set_font(style="B")
        pdf.cell(0, 10, txt="Forensic Analysis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font(style="")
        pdf.cell(0, 8, txt=f"Original Filename: {original_filename}", ln=True)
        pdf.cell(0, 8, txt=f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}", ln=True)
        pdf.ln(5)

        pdf.set_font(style="B")
        pdf.cell(40, 8, txt="Status:")
        pdf.set_font(style="")
        pdf.cell(0, 8, txt=str(result.get('status', 'N/A')), ln=True)

        pdf.set_font(style="B")
        pdf.cell(40, 8, txt="Confidence:")
        pdf.set_font(style="")
        pdf.cell(0, 8, txt=f"{result.get('confidence', 'N/A')}%", ln=True)

        pdf.set_font(style="B")
        pdf.cell(40, 8, txt="Explanation:")
        pdf.set_font(style="")
        pdf.multi_cell(0, 8, txt=str(result.get('explanation', 'N/A')))
        
        # --- MODIFIED SECTION ---
        # Instead of converting to a string and encoding, we'll write bytes directly to a buffer.
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0) # Rewind the buffer to the beginning for reading
        # ------------------------

        reader = PdfReader(pdf_buffer)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        writer.encrypt(user_password=password)

        with open(locked_path, "wb") as f:
            writer.write(f)
        
        print(f"✅ Encrypted PDF successfully saved to: {locked_path}")
        return locked_path

    except Exception as e:
        print(f"❌ An error occurred during PDF generation/encryption: {e}")
        return None