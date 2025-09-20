import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
from flask import (Flask, render_template, request, redirect, url_for, flash, 
                   send_from_directory)
from werkzeug.utils import secure_filename
from utils import analysis
from utils import pdf_generator

# --- 1. APP SETUP & CONFIGURATION ---
app = Flask(__name__)

# Change 1: Stronger Security for SECRET_KEY
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("No SECRET_KEY set for Flask application. Please set it in your environment variables.")
app.secret_key = SECRET_KEY

# Change 2: Cleaner Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'mp4', 'mov', 'avi'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 2. THE MAIN ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'evidence' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        
        file = request.files['evidence']
        password = request.form.get('pdf_password')

        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        if not password:
            flash('PDF report password is required.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            extension = original_filename.rsplit('.', 1)[1].lower()
            secure_name = f"{uuid.uuid4()}.{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
            file.save(filepath)

            analysis_result = analysis.analyze_file(filepath)
            
            pdf_path = pdf_generator.generate_pdf(app, original_filename, analysis_result, password)
            
            os.remove(filepath)

            if pdf_path:
                # Change 3: Better User Experience
                return render_template('reports.html', 
                                       result=analysis_result, 
                                       report_filename=os.path.basename(pdf_path),
                                       original_filename=original_filename)
            else:
                flash('An error occurred while generating the PDF report.', 'error')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)