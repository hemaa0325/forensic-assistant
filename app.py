import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for, flash, 
                   send_from_directory, jsonify)
from werkzeug.utils import secure_filename
from utils import analysis
from utils import pdf_generator
from utils.blockchain_handler import blockchain_handler

# --- 1. APP SETUP & CONFIGURATION ---
app = Flask(__name__)

# Change 1: Stronger Security for SECRET_KEY
SECRET_KEY = os.environ.get('SECRET_KEY', 'forensic-app-secret-key-2024')
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

# Comprehensive file type support for forensic analysis
ALLOWED_EXTENSIONS = {
    # Image formats
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'ico',
    # Video formats  
    'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v', '3gp',
    # Document formats
    'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt',
    # Archive formats (for forensic analysis)
    'zip', 'rar', '7z', 'tar', 'gz',
    # Audio formats (for potential analysis)
    'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'
}
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
        
        if file and file.filename and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            extension = original_filename.rsplit('.', 1)[1].lower()
            secure_name = f"{uuid.uuid4()}.{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
            file.save(filepath)

            # Read file content for blockchain storage
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            analysis_result = analysis.analyze_file(filepath)
            
            print(f"[DEBUG] Analysis completed for {original_filename}")
            print(f"[DEBUG] Analysis status: {analysis_result.get('status', 'Unknown')}")
            
            # Store analysis on blockchain (if available)
            blockchain_record = None
            if blockchain_handler.is_blockchain_available():
                print("🔗 Storing analysis on blockchain...")
                blockchain_result = blockchain_handler.store_analysis_on_blockchain(file_content, analysis_result)
                
                # Check if we got a proper blockchain record or a status message
                if blockchain_result and isinstance(blockchain_result, dict):
                    if blockchain_result.get('status') == 'confirmed':
                        # Successful storage
                        print(f"✅ Blockchain storage successful - Record ID: {blockchain_result.get('record_id')}")
                        analysis_result['blockchain_record'] = blockchain_result
                        analysis_result['blockchain_verified'] = True
                    elif blockchain_result.get('status') == 'skipped':
                        # Storage skipped (e.g., insufficient funds)
                        print(f"⚠️ {blockchain_result.get('message', 'Blockchain storage skipped')}")
                        analysis_result['blockchain_verified'] = False
                        # Add blockchain status info for the template
                        analysis_result['blockchain_status'] = {
                            'available': True,
                            'configured': True,
                            'storage_possible': False,
                            'reason': blockchain_result.get('reason', 'Unknown')
                        }
                    else:
                        # Other error
                        print("❌ Blockchain storage failed")
                        analysis_result['blockchain_verified'] = False
                        analysis_result['blockchain_status'] = {
                            'available': True,
                            'configured': True,
                            'storage_possible': False,
                            'reason': 'Storage operation failed'
                        }
                else:
                    print("❌ Blockchain storage failed")
                    analysis_result['blockchain_verified'] = False
                    analysis_result['blockchain_status'] = {
                        'available': True,
                        'configured': True,
                        'storage_possible': False,
                        'reason': 'Storage operation failed'
                    }
            else:
                print("⚠️ Blockchain not available - analysis stored locally only")
                analysis_result['blockchain_verified'] = False
                # Add blockchain status info for the template
                analysis_result['blockchain_status'] = {
                    'available': False,
                    'configured': False,
                    'storage_possible': False,
                    'reason': 'Blockchain not properly configured'
                }
            
            pdf_path = pdf_generator.generate_pdf(app, original_filename, analysis_result, password)
            
            print(f"[DEBUG] PDF generation result: {pdf_path}")
            
            os.remove(filepath)

            if pdf_path and os.path.exists(pdf_path):
                print(f"[DEBUG] PDF successfully created at: {pdf_path}")
                # Change 3: Better User Experience
                return render_template('reports.html', 
                                       result=analysis_result, 
                                       report_filename=os.path.basename(pdf_path),
                                       original_filename=original_filename)
            else:
                print(f"[ERROR] PDF generation failed. Path: {pdf_path}")
                flash('An error occurred while generating the PDF report.', 'error')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/reports')
def reports_list():
    """Page to list all available reports"""
    try:
        reports_dir = app.config['REPORT_FOLDER']
        if not os.path.exists(reports_dir):
            reports = []
        else:
            # Get all PDF files in reports directory
            report_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
            
            # Get file details
            reports = []
            for filename in report_files:
                filepath = os.path.join(reports_dir, filename)
                if os.path.exists(filepath):
                    stat = os.stat(filepath)
                    reports.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Sort by creation time (newest first)
            reports.sort(key=lambda x: x['created'], reverse=True)
        
        return render_template('reports_list.html', reports=reports)
    except Exception as e:
        flash(f'Error loading reports: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/verify')
def verify_page():
    """Page for blockchain verification"""
    return render_template('verify.html')

@app.route('/api/verify/<record_id>')
def verify_blockchain_record(record_id):
    """API endpoint to verify blockchain record"""
    try:
        record_id_int = int(record_id)
        verification_data = blockchain_handler.verify_analysis_on_blockchain(record_id_int)
        
        if verification_data:
            return jsonify({
                'success': True,
                'verified': True,
                'data': verification_data
            })
        else:
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'Record not found on blockchain'
            })
    except ValueError:
        return jsonify({
            'success': False,
            'verified': False,
            'error': 'Invalid record ID format'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'verified': False,
            'error': str(e)
        })

@app.route('/api/verify/hash/<file_hash>')
def verify_file_hash(file_hash):
    """API endpoint to verify file hash on blockchain"""
    try:
        verification_data = blockchain_handler.verify_file_hash_on_blockchain(file_hash)
        
        if verification_data:
            return jsonify({
                'success': True,
                'verified': True,
                'data': verification_data
            })
        else:
            return jsonify({
                'success': False,
                'verified': False,
                'error': 'File hash not found on blockchain'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'verified': False,
            'error': str(e)
        })

@app.route('/api/blockchain/status')
def blockchain_status():
    """API endpoint to get blockchain status"""
    try:
        status = blockchain_handler.get_blockchain_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=False)