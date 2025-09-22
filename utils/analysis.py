import os
from . import image_analysis, video_analysis, document_analysis

# Comprehensive file type support for forensic analysis
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'}
ARCHIVE_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg'}

def analyze_file(filepath):
    """
    Determines the file type and delegates to the appropriate analysis module.
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    # Analyze different file types
    if file_extension in IMAGE_EXTENSIONS:
        return image_analysis.analyze(filepath)
    
    elif file_extension in VIDEO_EXTENSIONS:
        return video_analysis.analyze(filepath)
        
    elif file_extension in DOCUMENT_EXTENSIONS:
        return document_analysis.analyze(filepath)
    
    elif file_extension in ARCHIVE_EXTENSIONS:
        return {
            "status": "Archive Detected", 
            "confidence": "95",
            "explanation": f"Archive file ({file_extension}) detected. Archive analysis requires specialized tools for content extraction and metadata examination.",
            "data": {
                "file_type": "Archive",
                "extension": file_extension,
                "analysis_note": "Manual extraction may be required for forensic analysis"
            }
        }
    
    elif file_extension in AUDIO_EXTENSIONS:
        return {
            "status": "Audio File Detected",
            "confidence": "90", 
            "explanation": f"Audio file ({file_extension}) detected. Audio forensic analysis includes metadata examination, spectral analysis, and authenticity verification.",
            "data": {
                "file_type": "Audio",
                "extension": file_extension,
                "analysis_note": "Audio forensic analysis module not yet implemented"
            }
        }
        
    else:
        return {
            "status": "Not Supported",
            "confidence": "100",
            "explanation": f"The file type '{file_extension}' is not currently supported for analysis. Supported formats: Images ({', '.join(IMAGE_EXTENSIONS)}), Videos ({', '.join(VIDEO_EXTENSIONS)}), Documents ({', '.join(DOCUMENT_EXTENSIONS)}), Archives ({', '.join(ARCHIVE_EXTENSIONS)}), Audio ({', '.join(AUDIO_EXTENSIONS)}).",
            "data": {
                "unsupported_extension": file_extension,
                "supported_categories": {
                    "images": list(IMAGE_EXTENSIONS),
                    "videos": list(VIDEO_EXTENSIONS), 
                    "documents": list(DOCUMENT_EXTENSIONS),
                    "archives": list(ARCHIVE_EXTENSIONS),
                    "audio": list(AUDIO_EXTENSIONS)
                }
            }
        }