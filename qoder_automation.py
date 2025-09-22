#!/usr/bin/env python3
"""
🚀 QODER AUTOMATION SCRIPT
Automated Forensic Analysis for Images and Documents

This script automates the complete forensic analysis workflow:
1. Scans directories for files
2. Analyzes each file (images/documents)
3. Generates reports with suspicion scores
4. Saves results to CSV/Excel
5. Can send email alerts for high-risk files
"""

import os
import sys
import argparse
# import pandas as pd  # Optional - only needed if pandas is installed
from datetime import datetime
import glob
import csv
# import smtplib  # Optional for email alerts
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import analysis
from utils import pdf_generator
from flask import Flask

# Configuration
SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
SUPPORTED_DOCUMENT_TYPES = ['.pdf', '.docx', '.doc']
HIGH_RISK_THRESHOLD = 7  # Suspicion score threshold for alerts
MEDIUM_RISK_THRESHOLD = 4

def analyze_file_automated(file_path: str) -> dict:
    """
    Automated analysis of a single file
    Returns standardized result dictionary
    """
    try:
        result = analysis.analyze_file(file_path)
        
        # Extract suspicion score
        suspicion_score = 0
        if 'data' in result:
            if 'total_suspiciousness_score' in result['data']:
                # Document analysis
                score_str = result['data']['total_suspiciousness_score']
                suspicion_score = int(score_str.split('/')[0])
            elif 'detailed_analysis' in result['data']:
                # Image analysis  
                score_str = result['data']['detailed_analysis']['suspiciousness_score']
                suspicion_score = int(score_str.split('/')[0])
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': os.path.splitext(file_path)[1].lower(),
            'status': result['status'],
            'confidence': result['confidence'],
            'suspicion_score': suspicion_score,
            'risk_level': get_risk_level(suspicion_score),
            'explanation': result['explanation'][:200] + '...' if len(result['explanation']) > 200 else result['explanation'],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': os.path.splitext(file_path)[1].lower(),
            'status': 'Error',
            'confidence': '0',
            'suspicion_score': 0,
            'risk_level': 'Unknown',
            'explanation': f'Analysis failed: {str(e)}',
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }

def get_risk_level(suspicion_score: int) -> str:
    """
    Determine risk level based on suspicion score
    """
    if suspicion_score >= HIGH_RISK_THRESHOLD:
        return 'HIGH'
    elif suspicion_score >= MEDIUM_RISK_THRESHOLD:
        return 'MEDIUM'
    else:
        return 'LOW'

def scan_directory(directory: str, recursive: bool = True) -> list:
    """
    Scan directory for supported files
    """
    supported_extensions = SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES
    files = []
    
    if recursive:
        for ext in supported_extensions:
            pattern = os.path.join(directory, '**', f'*{ext}')
            files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in supported_extensions:
            pattern = os.path.join(directory, f'*{ext}')
            files.extend(glob.glob(pattern))
    
    return files

def generate_reports(results: list, output_dir: str = 'automation_reports'):
    """
    Generate CSV and summary reports
    """
    if not results:
        print("No results to report")
        return {}, "", ""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed CSV using built-in csv module
    csv_path = os.path.join(output_dir, f'forensic_analysis_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"📊 Detailed report saved: {csv_path}")
    
    # Generate summary statistics
    total_files = len(results)
    high_risk_count = len([r for r in results if r['risk_level'] == 'HIGH'])
    medium_risk_count = len([r for r in results if r['risk_level'] == 'MEDIUM'])
    low_risk_count = len([r for r in results if r['risk_level'] == 'LOW'])
    error_count = len([r for r in results if r['status'] == 'Error'])
    avg_score = sum(r['suspicion_score'] for r in results) / len(results) if results else 0
    
    summary = {
        'Total Files Analyzed': total_files,
        'High Risk Files': high_risk_count,
        'Medium Risk Files': medium_risk_count,
        'Low Risk Files': low_risk_count,
        'Error Files': error_count,
        'Average Suspicion Score': round(avg_score, 2),
        'Analysis Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("🔍 AUTOMATED FORENSIC ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        # High risk files details
        high_risk_files = [r for r in results if r['risk_level'] == 'HIGH']
        if high_risk_files:
            f.write("\n🚨 HIGH RISK FILES:\n")
            f.write("-" * 30 + "\n")
            for file_info in high_risk_files:
                f.write(f"• {file_info['file_name']} (Score: {file_info['suspicion_score']}/10)\n")
    
    print(f"📋 Summary report saved: {summary_path}")
    return summary, csv_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='🚀 QODER - Automated Forensic Analysis')
    parser.add_argument('input_path', help='Input file or directory path')
    parser.add_argument('--output', '-o', default='automation_reports', help='Output directory for reports')
    parser.add_argument('--recursive', '-r', action='store_true', help='Scan directories recursively')
    parser.add_argument('--generate-pdf', action='store_true', help='Generate PDF reports for high-risk files')
    parser.add_argument('--threshold', type=int, default=7, help='High-risk threshold (default: 7)')
    
    args = parser.parse_args()
    
    print("🚀 QODER AUTOMATION STARTED")
    print("=" * 50)
    print(f"📁 Input: {args.input_path}")
    print(f"📊 Output: {args.output}")
    print(f"🎯 High-risk threshold: {args.threshold}")
    print()
    
    global HIGH_RISK_THRESHOLD
    HIGH_RISK_THRESHOLD = args.threshold
    
    # Get list of files to analyze
    if os.path.isfile(args.input_path):
        files_to_analyze = [args.input_path]
    elif os.path.isdir(args.input_path):
        files_to_analyze = scan_directory(args.input_path, args.recursive)
        print(f"📋 Found {len(files_to_analyze)} files to analyze")
    else:
        print(f"❌ Invalid input path: {args.input_path}")
        return
    
    if not files_to_analyze:
        print("❌ No supported files found")
        return
    
    # Analyze files
    results = []
    high_risk_files = []
    
    print("\\n🔍 Starting analysis...")
    for i, file_path in enumerate(files_to_analyze, 1):
        print(f"[{i}/{len(files_to_analyze)}] Analyzing: {os.path.basename(file_path)}")
        
        result = analyze_file_automated(file_path)
        results.append(result)
        
        if result['risk_level'] == 'HIGH':
            high_risk_files.append(result)
            print(f"  🚨 HIGH RISK detected! (Score: {result['suspicion_score']}/10)")
        elif result['risk_level'] == 'MEDIUM':
            print(f"  ⚠️  Medium risk (Score: {result['suspicion_score']}/10)")
        else:
            print(f"  ✅ Low risk (Score: {result['suspicion_score']}/10)")
    
    print("\\n📊 Generating reports...")
    summary, csv_path, summary_path = generate_reports(results, args.output)
    
    # Generate PDFs for high-risk files
    if args.generate_pdf and high_risk_files:
        print(f"\\n📄 Generating PDF reports for {len(high_risk_files)} high-risk files...")
        app = Flask(__name__)
        app.config['REPORT_FOLDER'] = args.output
        
        for file_info in high_risk_files:
            try:
                pdf_path = pdf_generator.generate_pdf(
                    app, 
                    file_info['file_name'], 
                    {
                        'status': file_info['status'],
                        'confidence': file_info['confidence'],
                        'explanation': file_info['explanation']
                    }, 
                    'forensic123'
                )
                if pdf_path:
                    print(f"  📄 PDF saved: {pdf_path}")
            except Exception as e:
                print(f"  ❌ PDF generation failed for {file_info['file_name']}: {e}")
    
    # Final summary
    print("\\n" + "=" * 50)
    print("🎉 AUTOMATION COMPLETED")
    print(f"📊 Total files analyzed: {summary['Total Files Analyzed']}")
    print(f"🚨 High risk files: {summary['High Risk Files']}")
    print(f"⚠️  Medium risk files: {summary['Medium Risk Files']}")
    print(f"✅ Low risk files: {summary['Low Risk Files']}")
    print(f"📋 Reports saved to: {args.output}")
    
    if high_risk_files:
        print(f"\\n🚨 IMMEDIATE ATTENTION REQUIRED:")
        for file_info in high_risk_files:
            print(f"  • {file_info['file_name']} (Score: {file_info['suspicion_score']}/10)")

if __name__ == "__main__":
    main()