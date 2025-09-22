# AI Analysis module - Forensic Image Analysis with 10-Criteria Scoring System
# This module provides a comprehensive forensic analysis following standardized criteria

from . import image_analysis
from typing import Dict, Any

def format_forensic_analysis_output(analysis_result: Dict[str, Any]) -> str:
    """
    Format the tiered analysis result into the specified output format.
    
    Args:
        analysis_result: Result from image_analysis.analyze()
        
    Returns:
        Formatted string with basic classification and detailed criteria breakdown
    """
    if analysis_result.get("status") == "Error":
        return analysis_result.get("explanation", "Analysis failed")
    
    # Get basic analysis data
    basic_data = analysis_result.get("data", {}).get("basic_analysis", {})
    classification = basic_data.get("classification", "Unknown")
    basic_confidence = basic_data.get("confidence", 0)
    
    # Get detailed analysis data
    detailed_data = analysis_result.get("data", {}).get("detailed_analysis", {})
    criteria_breakdown = detailed_data.get("criteria_breakdown", {})
    total_score = detailed_data.get("suspiciousness_score", "0/10")
    
    # Format basic classification
    formatted_output = []
    formatted_output.append("=== BASIC LEVEL CLASSIFICATION ===")
    formatted_output.append(f"Primary Detection: {classification}")
    formatted_output.append(f"Confidence: {basic_confidence}%")
    formatted_output.append("")
    
    # Format detailed criteria
    formatted_output.append("=== MEDIUM LEVEL DETAILED ANALYSIS ===")
    for criterion, result in criteria_breakdown.items():
        score = result.get('score', 0)
        reason = result.get('reason', 'No details available')
        formatted_output.append(f"{criterion}: {score} ({reason})")
    
    # Add total score and interpretation
    formatted_output.append("")
    formatted_output.append(f"[CHECK] Total Suspiciousness Score: {total_score}")
    formatted_output.append("")
    formatted_output.append("=> Interpretation:")
    formatted_output.append("0-3 points: Low chance of tampering")
    formatted_output.append("4-6 points: Medium suspicion")
    formatted_output.append("7-10 points: High likelihood of tampering")
    
    return "\n".join(formatted_output)

def analyze_image_forensics(filepath: str) -> Dict[str, Any]:
    """
    Perform TIERED forensic image analysis:
    
    BASIC LEVEL: AI Generation vs Manual Editing detection
    - Primary classification with confidence score
    - Fast, high-level determination of image origin
    
    MEDIUM LEVEL: Comprehensive 10-criteria analysis  
    - Detailed forensic examination across multiple dimensions
    - Secondary verification and detailed scoring
    
    Each detailed criterion awards 0 or 1 point based on tampering indicators.
    
    Args:
        filepath: Path to the image file to analyze
        
    Returns:
        Dictionary containing tiered analysis results with formatted output
    """
    try:
        # Perform the comprehensive analysis
        analysis_result = image_analysis.analyze(filepath)
        
        # Add formatted output for easy display
        formatted_output = format_forensic_analysis_output(analysis_result)
        analysis_result["formatted_output"] = formatted_output
        
        return analysis_result
        
    except Exception as e:
        return {
            "status": "Error",
            "confidence": "0",
            "explanation": f"Forensic analysis failed: {str(e)}",
            "formatted_output": f"Error: Could not analyze image - {str(e)}",
            "data": {}
        }

def detect_synthesis_artifacts(filepath: str) -> Dict[str, Any]:
    """
    Legacy function maintained for compatibility.
    Now redirects to comprehensive forensic analysis.
    """
    return analyze_image_forensics(filepath)

def get_criteria_descriptions() -> Dict[str, Any]:
    """
    Get detailed descriptions of the tiered forensic analysis system.
    
    Returns:
        Dictionary mapping analysis levels and criteria to their descriptions
    """
    return {
        "BASIC_LEVEL": "AI Generation vs Manual Editing Detection - Primary classification using advanced algorithms to distinguish between artificially generated images and manually edited photographs",
        
        "MEDIUM_LEVEL_CRITERIA": {
            "Metadata Consistency": "Checks EXIF data for missing, stripped, or inconsistent information (camera make/model, timestamp, GPS)",
            "Software Signature": "Detects if metadata indicates editing software usage (Photoshop, GIMP, etc.) instead of camera",
            "Noise Pattern": "Analyzes noise/grain level consistency across different regions of the image",
            "Lighting & Shadows": "Examines lighting direction and shadow consistency across objects in the image",
            "Edge Artifacts": "Detects unnatural edges, halos, or blurring around objects that suggest manipulation",
            "Resolution Uniformity": "Checks for regions with different resolution or quality levels within the same image",
            "Compression Artifacts": "Identifies signs of double JPEG compression or JPEG ghost artifacts",
            "Histogram Anomalies": "Analyzes color histograms for strange gaps or peaks suggesting manipulation",
            "ELA": "Error Level Analysis - detects different recompression error levels across image regions",
            "Camera Fingerprint": "Simplified PRNU analysis - checks for consistent sensor noise patterns"
        }
    }