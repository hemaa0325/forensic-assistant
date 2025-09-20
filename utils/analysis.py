import os
from . import image_analysis, video_analysis, document_analysis

# Change 1: Define supported extensions in one place for easy updates
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.txt'}

def analyze_file(filepath):
    """
    Determines the file type and delegates to the appropriate analysis module.
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    # Change 2: The logic is now cleaner and easier to read
    if file_extension in IMAGE_EXTENSIONS:
        return image_analysis.analyze(filepath)
    
    elif file_extension in VIDEO_EXTENSIONS:
        return video_analysis.analyze(filepath)
        
    elif file_extension in DOCUMENT_EXTENSIONS:
        return document_analysis.analyze(filepath)
        
    else:
        return {
            "status": "Not Supported",
            "confidence": "100",
            "explanation": f"The file type '{file_extension}' is not supported for analysis.",
            "data": {}
        }