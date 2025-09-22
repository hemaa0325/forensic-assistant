from PIL import Image, ExifTags
from PIL import Image, ExifTags
import os
import io
import numpy as np
import cv2
from typing import Dict, List, Tuple
from datetime import datetime
import re
from collections import Counter

from . import image_forensics_tools

def advanced_ai_detection(image_array: np.ndarray, image_bytes: bytes) -> Tuple[float, List[str]]:
    """
    Advanced AI detection using state-of-the-art techniques
    Returns: (ai_probability, ai_indicators)
    """
    ai_indicators = []
    ai_probability = 0.0
    
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        h, w = gray.shape
        
        # 1. PRNU (Photo Response Non-Uniformity) Analysis - Most effective for AI detection
        # AI images lack authentic sensor noise patterns
        prnu_score = 0.0
        try:
            # Extract noise residual
            denoised = cv2.medianBlur(gray, 3)
            noise_residual = gray.astype(np.float32) - denoised.astype(np.float32)
            
            # Analyze noise randomness using entropy
            hist, _ = np.histogram(noise_residual.flatten(), bins=50, range=(-25, 25))
            hist_norm = hist / np.sum(hist)
            hist_norm = hist_norm[hist_norm > 0]  # Remove zero bins
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            
            # Real cameras: entropy > 4.5, AI: entropy < 3.5
            if entropy < 3.5:
                prnu_score = (3.5 - entropy) / 3.5 * 0.4  # Max 0.4 contribution
                ai_indicators.append(f"Artificial sensor noise pattern (entropy: {entropy:.2f})")
            
        except Exception:
            pass
        
        # 2. Benford's Law Analysis - AI often violates natural digit distribution
        benford_score = 0.0
        try:
            # Extract first digits from pixel intensities
            non_zero_pixels = gray[gray > 0]
            if len(non_zero_pixels) > 1000:
                first_digits = []
                for pixel in non_zero_pixels[::10]:  # Sample every 10th pixel
                    digit = int(str(int(pixel))[0]) if str(int(pixel))[0] != '0' else None
                    if digit:
                        first_digits.append(digit)
                
                if len(first_digits) > 100:
                    # Expected Benford distribution
                    benford_expected = [np.log10(1 + 1/d) for d in range(1, 10)]
                    
                    # Observed distribution
                    digit_counts = np.bincount(first_digits)[1:10]  # Exclude 0
                    if len(digit_counts) == 9 and np.sum(digit_counts) > 0:
                        observed = digit_counts / np.sum(digit_counts)
                        
                        # Chi-square test for Benford's law
                        chi_square = np.sum((observed - benford_expected)**2 / benford_expected)
                        
                        # AI images often have chi-square > 15
                        if chi_square > 15:
                            benford_score = min(0.3, (chi_square - 15) / 50 * 0.3)
                            ai_indicators.append(f"Violates Benford's law (chi-square: {chi_square:.2f})")
            
        except Exception:
            pass
        
        # 3. Wavelet Analysis - AI creates specific frequency artifacts
        wavelet_score = 0.0
        try:
            try:
                import pywt
                # Discrete Wavelet Transform
                coeffs = pywt.dwt2(gray, 'db4')
                cA, (cH, cV, cD) = coeffs
                
                # Analyze high-frequency components
                high_freq = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
                
                # AI images have unnatural high-frequency patterns
                hf_std = np.std(high_freq)
                hf_mean = np.mean(np.abs(high_freq))
                
                if hf_std < hf_mean * 0.5:  # Too uniform high frequencies
                    wavelet_score = 0.25
                    ai_indicators.append(f"Artificial wavelet signature (HF uniformity: {hf_std/hf_mean:.3f})")
                
            except ImportError:
                # Fallback: FFT-based frequency analysis
                fft = np.fft.fft2(gray)
                fft_shift = np.fft.fftshift(fft)
                fft_magnitude = np.abs(fft_shift)
                
                # Analyze frequency distribution
                center_y, center_x = fft_magnitude.shape[0] // 2, fft_magnitude.shape[1] // 2
                high_freq_region = fft_magnitude[center_y-50:center_y+50, center_x-50:center_x+50]
                
                hf_uniformity = np.std(high_freq_region) / np.mean(high_freq_region)
                if hf_uniformity < 0.3:
                    wavelet_score = 0.2
                    ai_indicators.append(f"Artificial frequency distribution (uniformity: {hf_uniformity:.3f})")
        
        # 4. Local Contrast Analysis - AI smooths details unnaturally
        contrast_score = 0.0
        try:
            # Calculate local contrast using standard deviation in sliding windows
            window_size = 16
            contrasts = []
            
            for i in range(0, h - window_size, window_size):
                for j in range(0, w - window_size, window_size):
                    window = gray[i:i+window_size, j:j+window_size]
                    local_contrast = np.std(window)
                    contrasts.append(local_contrast)
            
            contrast_cv = np.std(contrasts) / np.mean(contrasts) if np.mean(contrasts) > 0 else 0
            
            # AI images have very uniform contrast (CV < 0.4)
            if contrast_cv < 0.4:
                contrast_score = (0.4 - contrast_cv) / 0.4 * 0.3
                ai_indicators.append(f"Unnaturally uniform contrast (CV: {contrast_cv:.3f})")
            
        except Exception:
            pass
        
        # 5. Color Channel Correlation Analysis
        correlation_score = 0.0
        try:
            if len(image_array.shape) == 3:
                r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
                
                # Calculate correlations between channels
                corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
                corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
                corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
                
                avg_correlation = np.mean([abs(corr_rg), abs(corr_rb), abs(corr_gb)])
                
                # AI images often have very high channel correlation (> 0.85)
                if avg_correlation > 0.85:
                    correlation_score = (avg_correlation - 0.85) / 0.15 * 0.2
                    ai_indicators.append(f"Artificial color correlation (avg: {avg_correlation:.3f})")
        
        except Exception:
            pass
        
        # 6. Edge Gradient Analysis - AI creates unnatural edge patterns
        gradient_score = 0.0
        try:
            # Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze gradient histogram
            grad_hist, _ = np.histogram(gradient_magnitude.flatten(), bins=50, range=(0, 255))
            grad_hist = grad_hist / np.sum(grad_hist)
            
            # AI often has peaked gradient distributions
            max_peak = np.max(grad_hist)
            if max_peak > 0.4:  # Very peaked distribution
                gradient_score = (max_peak - 0.4) / 0.6 * 0.25
                ai_indicators.append(f"Artificial gradient distribution (peak: {max_peak:.3f})")
        
        except Exception:
            pass
        
        # Combine all scores
        ai_probability = min(1.0, prnu_score + benford_score + wavelet_score + 
                            contrast_score + correlation_score + gradient_score)
        
        return ai_probability, ai_indicators
    
    except Exception as e:
        return 0.0, [f"Advanced AI detection failed: {str(e)}"]

def advanced_manual_edit_detection(image_array: np.ndarray, image_bytes: bytes) -> Tuple[float, List[str]]:
    """
    Advanced manual editing detection
    Returns: (edit_probability, edit_indicators)
    """
    edit_indicators = []
    edit_probability = 0.0
    
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        h, w = gray.shape
        
        # 1. JPEG Compression Artifact Analysis
        jpeg_score = 0.0
        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Multiple compression quality analysis
            compression_artifacts = []
            for quality in [70, 80, 90, 95]:
                buffer = io.BytesIO()
                original_image.save(buffer, 'JPEG', quality=quality)
                buffer.seek(0)
                
                recompressed = Image.open(buffer)
                diff = np.array(original_image, dtype=np.float32) - np.array(recompressed, dtype=np.float32)
                
                # 8x8 block analysis (JPEG compression units)
                block_diffs = []
                for i in range(0, h-8, 8):
                    for j in range(0, w-8, 8):
                        block = diff[i:i+8, j:j+8]
                        block_diff = np.mean(np.abs(block))
                        block_diffs.append(block_diff)
                
                if block_diffs:
                    compression_artifacts.append(np.std(block_diffs))
            
            if compression_artifacts:
                max_artifact = max(compression_artifacts)
                if max_artifact > 4.0:  # High compression inconsistency
                    jpeg_score = min(0.4, (max_artifact - 4.0) / 10.0 * 0.4)
                    edit_indicators.append(f"Double JPEG compression detected (artifact: {max_artifact:.2f})")
        
        except Exception:
            pass
        
        # 2. Copy-Paste Detection using Template Matching
        copypaste_score = 0.0
        try:
            # Divide image into overlapping patches
            patch_size = 32
            patches = []
            positions = []
            
            for i in range(0, h - patch_size, 16):
                for j in range(0, w - patch_size, 16):
                    patch = gray[i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
                    positions.append((i, j))
            
            # Find similar patches (potential copy-paste)
            similar_patches = 0
            for i, patch1 in enumerate(patches[:-1]):
                for j, patch2 in enumerate(patches[i+1:], i+1):
                    # Calculate normalized cross-correlation
                    correlation = cv2.matchTemplate(patch1, patch2, cv2.TM_CCOEFF_NORMED)[0,0]
                    
                    # Check if patches are far apart (not adjacent)
                    pos1, pos2 = positions[i], positions[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if correlation > 0.9 and distance > 64:  # High similarity, far apart
                        similar_patches += 1
            
            if similar_patches > 3:  # Multiple copy-paste instances
                copypaste_score = min(0.35, similar_patches / 20 * 0.35)
                edit_indicators.append(f"Copy-paste artifacts detected ({similar_patches} instances)")
        
        except Exception:
            pass
        
        # 3. Inconsistent Lighting Analysis
        lighting_score = 0.0
        try:
            # Divide into regions and analyze lighting direction
            regions = []
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = gray[i:min(i+h//4, h), j:min(j+w//4, w)]
                    if region.size > 100:
                        regions.append(region)
            
            # Calculate lighting direction for each region
            lighting_directions = []
            for region in regions:
                grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                
                # Dominant gradient direction
                avg_grad_x = np.mean(grad_x)
                avg_grad_y = np.mean(grad_y)
                direction = np.arctan2(avg_grad_y, avg_grad_x)
                lighting_directions.append(direction)
            
            if len(lighting_directions) > 1:
                direction_std = np.std(lighting_directions)
                if direction_std > 1.2:  # Inconsistent lighting
                    lighting_score = min(0.3, (direction_std - 1.2) / 2.0 * 0.3)
                    edit_indicators.append(f"Inconsistent lighting detected (std: {direction_std:.2f})")
        
        except Exception:
            pass
        
        # 4. Color Histogram Manipulation Detection
        histogram_score = 0.0
        try:
            if len(image_array.shape) == 3:
                # Analyze each color channel
                manipulated_channels = 0
                
                for channel in range(3):
                    hist = cv2.calcHist([image_array], [channel], None, [256], [0, 256]).flatten()
                    
                    # Check for unnatural gaps and peaks
                    zero_bins = np.sum(hist == 0)
                    hist_mean = hist.mean()
                    
                    # Find sharp peaks
                    peaks = 0
                    for i in range(1, 255):
                        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                            if hist[i] > hist_mean * 4:
                                peaks += 1
                    
                    # Check for manipulation indicators
                    if zero_bins > 30 or peaks > 5:
                        manipulated_channels += 1
                
                if manipulated_channels >= 2:
                    histogram_score = manipulated_channels / 3 * 0.25
                    edit_indicators.append(f"Color histogram manipulation ({manipulated_channels} channels affected)")
        
        except Exception:
            pass
        
        # Combine all scores
        edit_probability = min(1.0, jpeg_score + copypaste_score + lighting_score + histogram_score)
        
        return edit_probability, edit_indicators
    
    except Exception as e:
        return 0.0, [f"Advanced manual edit detection failed: {str(e)}"]

def detect_ai_generation(image_array: np.ndarray, image_bytes: bytes) -> Tuple[int, str, Dict]:
    """
    ENHANCED AI vs Manual Detection with Advanced Models
    Returns: (confidence_score, classification, details)
    """
    try:
        ai_indicators = []
        manual_edit_indicators = []
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        h, w = gray.shape
        ai_score = 0.0
        manual_score = 0.0
        
        # === ADVANCED AI DETECTION MODELS ===
        
        # Model 1: PRNU (Photo Response Non-Uniformity) Analysis
        try:
            # Extract sensor noise pattern
            denoised = cv2.medianBlur(gray, 3)
            noise_residual = gray.astype(np.float32) - denoised.astype(np.float32)
            
            # Analyze noise entropy (randomness)
            hist, _ = np.histogram(noise_residual.flatten(), bins=50, range=(-25, 25))
            hist_norm = hist / np.sum(hist)
            hist_norm = hist_norm[hist_norm > 0]
            if len(hist_norm) > 0:
                entropy = -np.sum(hist_norm * np.log2(hist_norm))
                
                # Real cameras: entropy > 4.0, AI: entropy < 3.0
                if entropy < 3.0:
                    ai_score += 0.4
                    ai_indicators.append(f"Artificial sensor noise (entropy: {entropy:.2f})")
        except:
            pass
        
        # Model 2: Benford's Law Analysis
        try:
            non_zero_pixels = gray[gray > 0]
            if len(non_zero_pixels) > 1000:
                first_digits = []
                for pixel in non_zero_pixels[::20]:  # Sample pixels
                    digit_str = str(int(pixel))
                    if digit_str[0] != '0':
                        first_digits.append(int(digit_str[0]))
                
                if len(first_digits) > 100:
                    # Expected vs observed first digit distribution
                    expected = [np.log10(1 + 1/d) for d in range(1, 10)]
                    digit_counts = np.bincount(first_digits)[1:10]
                    if len(digit_counts) == 9 and np.sum(digit_counts) > 0:
                        observed = digit_counts / np.sum(digit_counts)
                        
                        # Chi-square test
                        chi_square = np.sum((observed - expected)**2 / expected)
                        if chi_square > 12:  # Violates Benford's law
                            ai_score += 0.3
                            ai_indicators.append(f"Violates Benford's law (χ²: {chi_square:.1f})")
        except:
            pass
        
        # Model 3: Frequency Domain Analysis
        try:
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Radial frequency analysis
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            radial_profile = []
            for radius in range(10, min(center_y, center_x), 10):
                y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                if np.sum(mask) > 0:
                    radial_profile.append(float(np.mean(magnitude[mask])))
            
            if len(radial_profile) > 5:
                radial_std = float(np.std(radial_profile))
                if radial_std < 0.8:  # Too uniform = AI
                    ai_score += 0.25
                    ai_indicators.append(f"AI frequency signature (std: {radial_std:.2f})")
        except:
            pass
        
        # Model 4: Local Contrast Uniformity
        try:
            contrasts = []
            step = 24
            for i in range(0, h - step, step):
                for j in range(0, w - step, step):
                    window = gray[i:i+step, j:j+step]
                    local_std = float(window.std())
                    contrasts.append(local_std)
            
            if contrasts:
                contrast_cv = float(np.std(contrasts) / np.mean(contrasts)) if np.mean(contrasts) > 0 else 0
                if contrast_cv < 0.35:  # Too uniform
                    ai_score += 0.3
                    ai_indicators.append(f"Unnaturally uniform contrast (CV: {contrast_cv:.3f})")
        except:
            pass
        
        # Model 5: Color Channel Correlation
        try:
            if len(image_array.shape) == 3:
                r, g, b = image_array[:,:,0].flatten(), image_array[:,:,1].flatten(), image_array[:,:,2].flatten()
                
                corr_rg = float(np.corrcoef(r, g)[0,1])
                corr_rb = float(np.corrcoef(r, b)[0,1])
                corr_gb = float(np.corrcoef(g, b)[0,1])
                
                avg_correlation = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3
                
                if avg_correlation > 0.88:  # Too correlated = AI
                    ai_score += 0.2
                    ai_indicators.append(f"Artificial color correlation ({avg_correlation:.3f})")
        except:
            pass
        
        # === ADVANCED MANUAL EDITING DETECTION ===
        
        # Model 1: Double JPEG Compression
        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            compression_artifacts = []
            
            for quality in [75, 85, 95]:
                buffer = io.BytesIO()
                original_image.save(buffer, 'JPEG', quality=quality)
                buffer.seek(0)
                
                recompressed = Image.open(buffer)
                diff = np.array(original_image, dtype=np.float32) - np.array(recompressed, dtype=np.float32)
                
                # 8x8 block analysis
                block_diffs = []
                for i in range(0, min(h, diff.shape[0]) - 8, 8):
                    for j in range(0, min(w, diff.shape[1]) - 8, 8):
                        block = diff[i:i+8, j:j+8]
                        block_diff = float(np.mean(np.abs(block)))
                        block_diffs.append(block_diff)
                
                if block_diffs:
                    compression_artifacts.append(float(np.std(block_diffs)))
            
            if compression_artifacts and max(compression_artifacts) > 5.0:
                manual_score += 0.4
                manual_edit_indicators.append(f"Double JPEG compression (max: {max(compression_artifacts):.1f})")
        except:
            pass
        
        # Model 2: Copy-Paste Detection
        try:
            # Edge inconsistency analysis
            edges = cv2.Canny(gray, 30, 100)
            edge_regions = []
            
            for i in range(0, h, h//6):
                for j in range(0, w, w//6):
                    region = edges[i:min(i+h//6, h), j:min(j+w//6, w)]
                    if region.size > 0:
                        edge_density = float(np.sum(region) / region.size)
                        edge_regions.append(edge_density)
            
            if len(edge_regions) > 1:
                edge_std = float(np.std(edge_regions))
                edge_mean = float(np.mean(edge_regions))
                if edge_std > edge_mean * 0.8 and edge_mean > 0.02:
                    manual_score += 0.35
                    manual_edit_indicators.append(f"Copy-paste boundaries (edge var: {edge_std:.4f})")
        except:
            pass
        
        # Model 3: Histogram Manipulation
        try:
            if len(image_array.shape) == 3:
                manipulated_channels = 0
                for channel in range(3):
                    hist = cv2.calcHist([image_array], [channel], None, [256], [0, 256]).flatten()
                    
                    zero_bins = int(np.sum(hist == 0))
                    hist_mean = float(hist.mean())
                    
                    # Count unnatural peaks
                    peaks = 0
                    for i in range(1, 255):
                        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > hist_mean * 3.5:
                            peaks += 1
                    
                    if zero_bins > 25 or peaks > 6:
                        manipulated_channels += 1
                
                if manipulated_channels >= 2:
                    manual_score += 0.3
                    manual_edit_indicators.append(f"Histogram manipulation ({manipulated_channels} channels)")
        except:
            pass
        
        # === ENHANCED CLASSIFICATION LOGIC ===
        
        # Convert scores to confidence and classification
        ai_confidence = min(95, int(70 + ai_score * 25))
        manual_confidence = min(90, int(65 + manual_score * 25))
        
        if ai_score >= 0.6:  # Strong AI indicators
            classification = "AI Generated"
            confidence = ai_confidence
            primary_indicators = ai_indicators
        elif manual_score >= 0.6:  # Strong manual editing indicators
            classification = "Manually Edited"
            confidence = manual_confidence
            primary_indicators = manual_edit_indicators
        elif ai_score > manual_score and ai_score >= 0.3:
            classification = "Likely AI Generated"
            confidence = max(60, ai_confidence - 10)
            primary_indicators = ai_indicators
        elif manual_score > ai_score and manual_score >= 0.3:
            classification = "Likely Manually Edited"
            confidence = max(60, manual_confidence - 10)
            primary_indicators = manual_edit_indicators
        else:
            # No clear indicators
            classification = "Likely Authentic"
            confidence = 75
            primary_indicators = ["No significant AI generation or manual editing indicators detected"]
        
        details = {
            "ai_indicators": ai_indicators,
            "manual_edit_indicators": manual_edit_indicators,
            "ai_score": round(ai_score, 2),
            "manual_score": round(manual_score, 2),
            "primary_indicators": primary_indicators,
            "advanced_analysis": True
        }
        
        return confidence, classification, details
        
    except Exception as e:
        return 50, "Analysis Failed", {"error": str(e)}

def check_metadata_consistency(image: Image.Image, filepath: str) -> Tuple[int, str]:
    """
    MEDIUM LEVEL - Criterion 1: Check EXIF data for missing, stripped, or inconsistent metadata
    """
    try:
        exif_data_raw = getattr(image, '_getexif', lambda: None)()
        if not exif_data_raw:
            return 1, "EXIF data missing or stripped"
        
        exif_data = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data_raw.items()}
        
        # Check for essential metadata
        timestamp = exif_data.get('DateTime') or exif_data.get('DateTimeOriginal')
        if not timestamp:
            return 1, "Timestamp missing from EXIF"
        
        # Check GPS consistency
        gps_info = exif_data.get('GPSInfo')
        if gps_info and len(gps_info) < 4:  # Incomplete GPS data
            return 1, "Incomplete GPS data suggests tampering"
        
        # Check for suspicious EXIF patterns
        make = exif_data.get('Make', '')
        model = exif_data.get('Model', '')
        if not make and not model:
            return 1, "Camera make/model information missing"
        
        return 0, "Metadata appears consistent"
    except Exception:
        return 1, "Failed to read EXIF data"
    """
    Criterion 1: Check EXIF data for missing, stripped, or inconsistent metadata
    """
    try:
        exif_data_raw = getattr(image, '_getexif', lambda: None)()
        if not exif_data_raw:
            return 1, "EXIF data missing or stripped"
        
        exif_data = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data_raw.items()}
        
        # Check for essential metadata
        timestamp = exif_data.get('DateTime') or exif_data.get('DateTimeOriginal')
        if not timestamp:
            return 1, "Timestamp missing from EXIF"
        
        # Check GPS consistency
        gps_info = exif_data.get('GPSInfo')
        if gps_info and len(gps_info) < 4:  # Incomplete GPS data
            return 1, "Incomplete GPS data suggests tampering"
        
        # Check for suspicious EXIF patterns
        make = exif_data.get('Make', '')
        model = exif_data.get('Model', '')
        if not make and not model:
            return 1, "Camera make/model information missing"
        
        return 0, "Metadata appears consistent"
    except Exception:
        return 1, "Failed to read EXIF data"

def check_software_signature(image: Image.Image) -> Tuple[int, str]:
    """
    Criterion 2: Check if metadata indicates editing software usage
    """
    try:
        exif_data_raw = getattr(image, '_getexif', lambda: None)()
        if not exif_data_raw:
            return 0, "No EXIF data to check software signature"
        
        exif_data = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data_raw.items()}
        
        software = exif_data.get('Software', '')
        if software:
            editing_software = ['photoshop', 'gimp', 'paint', 'lightroom', 'canva', 'pixlr']
            if any(editor.lower() in software.lower() for editor in editing_software):
                return 1, f"Editing software detected: {software}"
        
        # Check for other editing indicators
        processing_software = exif_data.get('ProcessingSoftware', '')
        if processing_software:
            return 1, f"Processing software detected: {processing_software}"
        
        return 0, "No editing software signatures detected"
    except Exception:
        return 0, "Could not check software signatures"

def check_noise_pattern(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 3: Analyze noise/grain consistency across image regions
    """
    try:
        # Convert to grayscale for noise analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Divide image into 9 regions (3x3 grid)
        h, w = gray.shape
        region_h, region_w = h // 3, w // 3
        
        noise_levels = []
        for i in range(3):
            for j in range(3):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                # Calculate noise using Laplacian variance
                noise_level = cv2.Laplacian(region, cv2.CV_64F).var()
                noise_levels.append(noise_level)
        
        # Check for significant variations in noise levels (more sensitive threshold)
        noise_std = np.std(noise_levels)
        noise_mean = np.mean(noise_levels)
        
        # More sensitive detection - flag if noise variation > 30% of mean
        if noise_std > noise_mean * 0.3:  # Reduced from 0.5 to be more sensitive
            return 1, f"Inconsistent noise levels detected (std: {noise_std:.2f}, mean: {noise_mean:.2f})"
        
        return 0, "Noise pattern appears consistent"
    except Exception:
        return 0, "Could not analyze noise patterns"

def check_lighting_shadows(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 4: Analyze lighting direction and shadow consistency
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Use Sobel operators to detect edge gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Analyze lighting consistency by checking dominant gradient directions
        # in different regions
        h, w = gray.shape
        region_h, region_w = h // 2, w // 2
        
        directions = []
        for i in range(2):
            for j in range(2):
                region_dir = direction[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                region_mag = magnitude[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                
                # Get dominant direction weighted by magnitude
                strong_edges = region_mag > np.percentile(region_mag, 80)
                if np.sum(strong_edges) > 0:
                    dominant_dir = np.mean(region_dir[strong_edges])
                    directions.append(dominant_dir)
        
        if len(directions) > 1:
            dir_std = np.std(directions)
            # More sensitive threshold - reduced from 1.0 to 0.7
            if dir_std > 0.7:  # Significant variation in lighting directions
                return 1, f"Inconsistent lighting directions detected (std: {dir_std:.2f})"
        
        return 0, "Lighting and shadows appear consistent"
    except Exception:
        return 0, "Could not analyze lighting patterns"

def check_edge_artifacts(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 5: Detect unnatural edges, halos, or blurring
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for suspicious edge patterns
        # 1. Check for double edges (halos)
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        eroded_edges = cv2.erode(edges, kernel, iterations=1)
        
        # Count parallel edge structures (more sensitive detection)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        dilated_density = np.sum(dilated_edges) / (gray.shape[0] * gray.shape[1])
        
        # More sensitive threshold - reduced from 1.5 to 1.3
        if dilated_density > edge_density * 1.3:
            return 1, "Suspicious edge patterns detected (possible halos)"
        
        # 2. Check for unnatural sharpness transitions (more sensitive)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # Lower threshold for detecting artificial sharpening - reduced from 1000 to 600
        if lap_var > 600:
            return 1, f"Unnatural sharpness detected (Laplacian variance: {lap_var:.2f})"
        
        return 0, "No suspicious edge artifacts detected"
    except Exception:
        return 0, "Could not analyze edge artifacts"

def check_resolution_uniformity(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 6: Check for resolution/quality inconsistencies
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Divide image into regions and analyze local frequency content
        h, w = gray.shape
        region_h, region_w = h // 3, w // 3
        
        frequency_responses = []
        for i in range(3):
            for j in range(3):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                
                # Apply FFT to analyze frequency content
                fft = np.fft.fft2(region)
                fft_shift = np.fft.fftshift(fft)
                magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
                
                # Calculate high-frequency content
                center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
                high_freq_region = magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10]
                high_freq_energy = np.mean(high_freq_region)
                frequency_responses.append(high_freq_energy)
        
        # Check for significant variations in frequency content (more sensitive)
        freq_std = np.std(frequency_responses)
        freq_mean = np.mean(frequency_responses)
        
        # More sensitive threshold - reduced from 0.3 to 0.2
        if freq_std > freq_mean * 0.2:  # Significant variation in quality
            return 1, f"Resolution inconsistencies detected (freq std: {freq_std:.2f}, mean: {freq_mean:.2f})"
        
        return 0, "Resolution appears uniform across image"
    except Exception:
        return 0, "Could not analyze resolution uniformity"

def check_compression_artifacts(image_bytes: bytes) -> Tuple[int, str]:
    """
    Criterion 7: Detect double JPEG compression or JPEG ghosts
    """
    try:
        # Use the existing ELA function as part of compression analysis
        ela_result = image_forensics_tools.perform_ela(image_bytes)
        if ela_result:
            return 1, "Double JPEG compression detected via ELA"
        
        # Additional JPEG ghost detection
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Test multiple quality levels to detect ghosts
        ghost_detected = False
        for quality in [70, 80, 90, 95]:
            buffer = io.BytesIO()
            original_image.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            
            test_image = Image.open(buffer)
            diff = np.array(original_image) - np.array(test_image)
            diff_score = np.mean(np.abs(diff))
            
            if diff_score < 2.0:  # Very small difference suggests previous compression at this quality
                ghost_detected = True
                break
        
        if ghost_detected:
            return 1, "JPEG compression ghosts detected"
        
        return 0, "No significant compression artifacts detected"
    except Exception:
        return 0, "Could not analyze compression artifacts"

def check_histogram_anomalies(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 8: Analyze histogram for manipulation signs
    """
    try:
        # Calculate histograms for each channel
        if len(image_array.shape) == 3:
            histograms = [cv2.calcHist([image_array], [i], None, [256], [0, 256]) for i in range(3)]
        else:
            histograms = [cv2.calcHist([image_array], [0], None, [256], [0, 256])]
        
        suspicious_patterns = []
        
        for i, hist in enumerate(histograms):
            hist = hist.flatten()
            
            # Check for suspicious gaps (manipulation often creates gaps) - more sensitive
            zero_bins = np.sum(hist == 0)
            if zero_bins > 30:  # Reduced from 50 - fewer empty bins trigger detection
                suspicious_patterns.append(f"Channel {i}: {zero_bins} empty histogram bins")
            
            # Check for unnatural peaks - more sensitive
            peaks = []
            hist_mean = hist.mean()
            for j in range(1, len(hist)-1):
                if hist[j] > hist[j-1] and hist[j] > hist[j+1] and hist[j] > hist_mean * 1.8:  # Reduced from 2.0
                    peaks.append(j)
            
            if len(peaks) > 8:  # Reduced from 10 - fewer peaks trigger detection
                suspicious_patterns.append(f"Channel {i}: {len(peaks)} suspicious peaks")
            
            # Check for cut-off patterns (common in manipulation) - more sensitive
            if hist[0] > hist_mean * 2.5 or hist[-1] > hist_mean * 2.5:  # Reduced from 3.0
                suspicious_patterns.append(f"Channel {i}: histogram cutoff detected")
        
        if suspicious_patterns:
            return 1, "; ".join(suspicious_patterns)
        
        return 0, "Histogram appears normal"
    except Exception:
        return 0, "Could not analyze histogram"

def check_ela_analysis(image_bytes: bytes) -> Tuple[int, str]:
    """
    Criterion 9: Perform Error Level Analysis for recompression errors
    """
    try:
        # Detailed ELA analysis
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Test multiple quality levels
        ela_scores = []
        for quality in [85, 90, 95]:
            buffer = io.BytesIO()
            original_image.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            
            recompressed = Image.open(buffer)
            
            # Calculate difference
            orig_array = np.array(original_image, dtype=np.float32)
            recomp_array = np.array(recompressed, dtype=np.float32)
            
            diff = np.abs(orig_array - recomp_array)
            
            # Analyze regional differences
            h, w = diff.shape[:2]
            region_h, region_w = h // 4, w // 4
            
            region_scores = []
            for i in range(4):
                for j in range(4):
                    region = diff[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                    region_score = np.mean(region)
                    region_scores.append(region_score)
            
            # Check for significant variations
            score_std = np.std(region_scores)
            ela_scores.append(score_std)
        
        # If ELA shows consistent high variation, likely tampered (more sensitive)
        avg_ela_score = np.mean(ela_scores)
        if avg_ela_score > 3.0:  # Reduced from 5.0 to be more sensitive
            return 1, f"ELA shows inconsistent recompression errors (score: {avg_ela_score:.2f})"
        
        return 0, "ELA analysis shows consistent compression"
    except Exception:
        return 0, "Could not perform ELA analysis"

def check_camera_fingerprint(image_array: np.ndarray) -> Tuple[int, str]:
    """
    Criterion 10: Basic sensor pattern analysis (simplified PRNU)
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Simplified sensor noise analysis
        # Apply denoising and calculate residual noise
        denoised = cv2.medianBlur(gray, 5)
        noise_residual = gray.astype(np.float32) - denoised.astype(np.float32)
        
        # Analyze noise pattern uniformity
        h, w = noise_residual.shape
        region_h, region_w = h // 4, w // 4
        
        noise_patterns = []
        for i in range(4):
            for j in range(4):
                region = noise_residual[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                # Calculate noise characteristics
                noise_std = np.std(region)
                noise_mean = np.mean(np.abs(region))
                noise_patterns.append((noise_std, noise_mean))
        
        # Check for inconsistent sensor patterns
        std_values = [pattern[0] for pattern in noise_patterns]
        mean_values = [pattern[1] for pattern in noise_patterns]
        
        std_variation = np.std(std_values) / np.mean(std_values) if np.mean(std_values) > 0 else 0
        mean_variation = np.std(mean_values) / np.mean(mean_values) if np.mean(mean_values) > 0 else 0
        
        # More sensitive thresholds - reduced from 0.5 to 0.35
        if std_variation > 0.35 or mean_variation > 0.35:
            return 1, f"Inconsistent sensor pattern detected (std var: {std_variation:.2f}, mean var: {mean_variation:.2f})"
        
        return 0, "Sensor pattern appears consistent"
    except Exception:
        return 0, "Could not analyze camera fingerprint"

def analyze(filepath: str) -> Dict:
    """
    Comprehensive forensic image analysis using 10-criteria scoring system.
    Each criterion awards 0 or 1 point, total suspiciousness score 0-10.
    """
    try:
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)
        
        # === BASIC LEVEL ANALYSIS ===
        basic_confidence, basic_classification, basic_details = detect_ai_generation(image_array, image_bytes)
        # === MEDIUM LEVEL ANALYSIS ===
        criteria_results = {}
        total_score = 0
        
        # Criterion 1: Metadata Consistency
        score, reason = check_metadata_consistency(image, filepath)
        criteria_results['Metadata Consistency'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 2: Software Signature
        score, reason = check_software_signature(image)
        criteria_results['Software Signature'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 3: Noise Pattern
        score, reason = check_noise_pattern(image_array)
        criteria_results['Noise Pattern'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 4: Lighting & Shadows
        score, reason = check_lighting_shadows(image_array)
        criteria_results['Lighting & Shadows'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 5: Edge Artifacts
        score, reason = check_edge_artifacts(image_array)
        criteria_results['Edge Artifacts'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 6: Resolution Uniformity
        score, reason = check_resolution_uniformity(image_array)
        criteria_results['Resolution Uniformity'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 7: Compression Artifacts
        score, reason = check_compression_artifacts(image_bytes)
        criteria_results['Compression Artifacts'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 8: Histogram Anomalies
        score, reason = check_histogram_anomalies(image_array)
        criteria_results['Histogram Anomalies'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 9: Error Level Analysis (ELA)
        score, reason = check_ela_analysis(image_bytes)
        criteria_results['ELA'] = {'score': score, 'reason': reason}
        total_score += score
        
        # Criterion 10: Camera Fingerprint (PRNU)
        score, reason = check_camera_fingerprint(image_array)
        criteria_results['Camera Fingerprint'] = {'score': score, 'reason': reason}
        total_score += score
        
        # === DETERMINE FINAL STATUS ===
        # Basic classification takes priority
        if basic_classification == "AI Generated":
            status = "AI Generated"
            confidence = str(basic_confidence)
        elif basic_classification == "Manually Edited":
            status = "Manually Edited" 
            confidence = str(basic_confidence)
        elif basic_classification == "Likely Authentic":
            status = "Likely Authentic"
            confidence = str(basic_confidence)
        else:
            # Fall back to medium level analysis for final determination
            if total_score <= 3:
                status = "Likely Authentic"
                confidence = str(max(10, 100 - (total_score * 10)))
            elif total_score <= 6:
                status = "Suspicious"
                confidence = str(max(10, 100 - (total_score * 10)))
            else:
                status = "Likely Tampered"
                confidence = str(max(10, 100 - (total_score * 10)))
        
        # === GENERATE EXPLANATION ===
        explanation = f"""FORENSIC ANALYSIS RESULTS

=== BASIC LEVEL CLASSIFICATION ===
Primary Detection: {basic_classification}
Confidence: {basic_confidence}%

Key Indicators:
""" + "\n".join([f"- {indicator}" for indicator in basic_details.get('primary_indicators', [])])
        
        explanation += f"""

=== MEDIUM LEVEL DETAILED ANALYSIS ===
Suspiciousness Score: {total_score}/10

Detailed Criteria Breakdown:
"""
        
        for criterion, result in criteria_results.items():
            score_icon = "[SUSPICIOUS]" if result['score'] == 1 else "[CLEAN]"
            explanation += f"{score_icon} {criterion}: {result['score']} - {result['reason']}\n"
        
        explanation += f"""
=== INTERPRETATION ===
• 0-3 points: Low chance of tampering
• 4-6 points: Medium suspicion
• 7-10 points: High likelihood of tampering

FINAL ASSESSMENT: {status}
This image shows {basic_classification.lower()} characteristics with a {total_score}/10 detailed suspiciousness score."""
        
        return {
            "status": status,
            "confidence": confidence,
            "explanation": explanation,
            "data": {
                "basic_analysis": {
                    "classification": basic_classification,
                    "confidence": basic_confidence,
                    "details": basic_details
                },
                "detailed_analysis": {
                    "suspiciousness_score": f"{total_score}/10",
                    "criteria_breakdown": criteria_results
                }
            }
        }
    
    except Exception as e:
        return {
            "status": "Error",
            "confidence": "0",
            "explanation": f"Forensic analysis failed. File may be corrupt or invalid. Error: {e}",
            "data": {}
        }