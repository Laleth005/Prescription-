"""
Image forensics utilities for detecting fake prescriptions.
"""
import cv2
import numpy as np
import os
from PIL import Image
import io
from typing import Tuple, Dict, List, Any

def check_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Analyze image metadata for signs of tampering or editing.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with metadata analysis results
    """
    result = {
        'metadata_present': False,
        'edited': False,
        'editing_software': None,
        'creation_date': None,
        'modification_date': None,
        'suspicious_flags': []
    }
    
    try:
        # Use PIL to extract metadata
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data:
            result['metadata_present'] = True
            
            # Check for editing software
            if 305 in exif_data:  # Software tag
                result['editing_software'] = exif_data[305]
                if any(editor in exif_data[305] for editor in ['photoshop', 'gimp', 'paint', 'editor']):
                    result['edited'] = True
                    result['suspicious_flags'].append(f"Edited with {exif_data[305]}")
            
            # Check creation date
            if 36867 in exif_data:  # DateTimeOriginal tag
                result['creation_date'] = exif_data[36867]
                
            # Check modification date
            if 306 in exif_data:  # DateTime tag
                result['modification_date'] = exif_data[306]
                
                # If modification date is significantly different from creation date
                if result['creation_date'] and result['modification_date'] != result['creation_date']:
                    result['suspicious_flags'].append("Creation and modification dates differ")
        else:
            result['suspicious_flags'].append("No metadata found - possible sign of metadata removal")
            
    except Exception as e:
        result['suspicious_flags'].append(f"Error analyzing metadata: {str(e)}")
    
    return result

def detect_copy_paste_artifacts(image_path: str) -> Dict[str, Any]:
    """
    Detect potential copy-paste artifacts that might indicate tampering.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with copy-paste detection results
    """
    result = {
        'artifacts_detected': False,
        'confidence': 0,
        'regions': [],
        'suspicious_flags': []
    }
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            result['suspicious_flags'].append("Unable to read image")
            return result
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply ELA (Error Level Analysis)
        temp_filename = "temp_ela.jpg"
        cv2.imwrite(temp_filename, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Read the compressed image
        compressed = cv2.imread(temp_filename)
        os.remove(temp_filename)
        
        if compressed is None:
            result['suspicious_flags'].append("Error in ELA analysis")
            return result
            
        # Calculate the difference
        diff = cv2.absdiff(img, compressed)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold the mask
        _, thresh = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if len(significant_contours) > 0:
            result['artifacts_detected'] = True
            result['confidence'] = min(100, len(significant_contours) * 10)
            
            for i, contour in enumerate(significant_contours[:5]):  # Limit to top 5 regions
                x, y, w, h = cv2.boundingRect(contour)
                result['regions'].append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                })
                
            if len(significant_contours) > 3:
                result['suspicious_flags'].append(f"Multiple editing artifacts detected ({len(significant_contours)})")
        
    except Exception as e:
        result['suspicious_flags'].append(f"Error in artifact detection: {str(e)}")
    
    return result

def check_font_consistency(image_path: str) -> Dict[str, Any]:
    """
    Check for font consistency within the prescription.
    Inconsistent fonts may indicate tampering.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with font consistency check results
    """
    result = {
        'consistent_fonts': True,
        'confidence': 0,
        'suspicious_flags': []
    }
    
    try:
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find text regions using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        # Filter out small components (noise)
        min_area = 20
        text_stats = [stat for stat in stats[1:] if stat[4] > min_area]  # stat[4] is the area
        
        if len(text_stats) < 5:
            result['suspicious_flags'].append("Too few text regions found for analysis")
            return result
            
        # Analyze height distribution of text components
        heights = [stat[3] for stat in text_stats]  # stat[3] is the height
        
        # Calculate statistics
        mean_height = np.mean(heights)
        std_height = np.std(heights)
        
        # Coefficient of variation (CV) - higher values indicate more variability
        cv_height = std_height / mean_height if mean_height > 0 else 0
        
        # Determine consistency based on CV
        # Typical CV for consistent fonts is below 0.3
        if cv_height > 0.4:
            result['consistent_fonts'] = False
            result['suspicious_flags'].append(f"Inconsistent font sizes detected (CV: {cv_height:.2f})")
            
        result['confidence'] = int(100 * (1 - min(1, cv_height / 0.5)))
        
    except Exception as e:
        result['suspicious_flags'].append(f"Error in font consistency check: {str(e)}")
    
    return result

def verify_prescription_signature(image_path: str) -> Dict[str, Any]:
    """
    Attempt to verify if a prescription has a valid signature.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with signature verification results
    """
    result = {
        'signature_detected': False,
        'confidence': 0,
        'location': None,
        'suspicious_flags': []
    }
    
    try:
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        potential_signatures = []
        height, width = img.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Typical signatures have specific characteristics
            if (0.2 < aspect_ratio < 5 and  # Not too narrow or wide
                area > 500 and  # Not too small
                area < (width * height * 0.1) and  # Not too large
                solidity < 0.8 and  # Not too solid (signatures are usually not solid shapes)
                y > height / 2):  # Usually in the bottom half of the prescription
                
                potential_signatures.append((contour, area, x, y, w, h))
        
        if potential_signatures:
            # Sort by area (largest first)
            potential_signatures.sort(key=lambda x: x[1], reverse=True)
            
            # Get the largest potential signature
            _, _, x, y, w, h = potential_signatures[0]
            
            result['signature_detected'] = True
            result['location'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            
            # Calculate confidence based on location and other factors
            # Bottom right is typical for signatures
            bottom_right_score = min(100, int(y / height * 100))  # Higher y is better
            
            # Size should be reasonable
            size_score = min(100, int((w * h) / (width * height * 0.05) * 100))
            
            # Calculate overall confidence
            result['confidence'] = min(100, int((bottom_right_score + size_score) / 2))
            
            if result['confidence'] < 50:
                result['suspicious_flags'].append("Signature detected but in unusual location or size")
                
        else:
            result['suspicious_flags'].append("No signature detected")
            
    except Exception as e:
        result['suspicious_flags'].append(f"Error in signature verification: {str(e)}")
    
    return result

def analyze_image_forensics(image_path: str) -> Dict[str, Any]:
    """
    Run comprehensive image forensics analysis on a prescription image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with all forensics analysis results
    """
    results = {
        'metadata': check_image_metadata(image_path),
        'copy_paste': detect_copy_paste_artifacts(image_path),
        'font_consistency': check_font_consistency(image_path),
        'signature': verify_prescription_signature(image_path),
        'is_suspicious': False,
        'confidence': 0,
        'suspicious_flags': []
    }
    
    # Collect all suspicious flags
    for key in ['metadata', 'copy_paste', 'font_consistency', 'signature']:
        if key in results and 'suspicious_flags' in results[key]:
            results['suspicious_flags'].extend(results[key]['suspicious_flags'])
    
    # Determine overall suspiciousness
    suspicion_factors = [
        results['metadata']['edited'],
        results['copy_paste']['artifacts_detected'],
        not results['font_consistency']['consistent_fonts'],
        not results['signature']['signature_detected']
    ]
    
    results['is_suspicious'] = any(suspicion_factors)
    
    # Calculate overall confidence
    confidence_factors = [
        100 if not results['metadata']['edited'] else 0,
        100 - results['copy_paste']['confidence'],
        results['font_consistency']['confidence'],
        results['signature']['confidence'] if results['signature']['signature_detected'] else 0
    ]
    
    # Average of valid confidence factors
    valid_factors = [f for f in confidence_factors if f > 0]
    results['confidence'] = int(sum(valid_factors) / len(valid_factors)) if valid_factors else 0
    
    return results
