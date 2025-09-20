from PIL import Image, ImageChops
import numpy as np
import cv2
import io

# Change 1: Added Type Hinting for clarity (image_bytes: bytes) -> bool
def perform_ela(image_bytes: bytes, quality: int = 90, threshold: int = 15) -> bool:
    """Performs a robust Error Level Analysis."""
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        resaved_buffer = io.BytesIO()
        original_image.save(resaved_buffer, 'JPEG', quality=quality)
        resaved_buffer.seek(0)
        resaved_image = Image.open(resaved_buffer)
        diff = ImageChops.difference(original_image, resaved_image)
        diff_array = np.array(diff)
        mean_diff = np.mean(diff_array)
        std_diff = np.std(diff_array)
        if std_diff > threshold and mean_diff > (threshold / 3):
            return True
    except Exception: pass
    return False

# Change 1: Added Type Hinting for clarity
def detect_cloned_regions(image_bytes: bytes) -> bool:
    """Detects copy-move forgery using ORB and RANSAC geometric consistency check."""
    
    # Change 2: Defined parameters as constants for easy tuning and readability
    N_FEATURES = 2000
    RATIO_THRESHOLD = 0.75
    MIN_MATCHES_INITIAL = 20
    RANSAC_THRESHOLD = 5.0
    MIN_INLIERS = 15
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        
        # 1. Detect keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=N_FEATURES)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None or len(keypoints) < MIN_MATCHES_INITIAL: return False

        # 2. Match features and apply Lowe's Ratio Test
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors, descriptors, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < RATIO_THRESHOLD * n.distance and m.queryIdx != m.trainIdx:
                    good_matches.append(m)

        # 3. Perform Geometric Consistency Check (RANSAC)
        if len(good_matches) > MIN_MATCHES_INITIAL:
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
            
            if M is not None and np.sum(mask) > MIN_INLIERS:
                return True
    except Exception: pass
    return False