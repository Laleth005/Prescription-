"""
Signature verification utilities for prescription analysis.

This module provides functions to detect, extract and verify signatures on prescriptions.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import os

def detect_signature_region(image) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Detect and extract a potential signature region from a prescription image.
    
    Args:
        image: OpenCV image (numpy array) of the prescription
        
    Returns:
        Tuple of (signature region image or None, analysis results dictionary)
    """
    result = {
        'signature_detected': False,
        'location': None,
        'confidence': 0,
        'dimensions': None,
        'analysis': {}
    }
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the bottom third of the image (where signatures typically are)
    height, width = binary.shape
    bottom_third = binary[int(height*2/3):height, :]
    
    contours, _ = cv2.findContours(bottom_third, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Get the bounding rectangle for top contours and check characteristics
    signature_candidates = []
    
    for contour in contours[:5]:  # Consider top 5 largest contours
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adjust y to account for bottom_third offset
        y += int(height*2/3)
        
        # Calculate contour area and density
        area = cv2.contourArea(contour)
        rect_area = w * h
        if rect_area == 0:
            continue
        
        density = area / rect_area
        aspect_ratio = w / h if h > 0 else 0
        
        # Skip very small regions
        if w < width * 0.05 or h < height * 0.01:
            continue
            
        # Skip very large regions
        if w > width * 0.9 or h > height * 0.3:
            continue
        
        # Signature characteristics - typically horizontal, medium density
        if 1.5 < aspect_ratio < 10 and 0.1 < density < 0.7:
            confidence = min(100, density * 100 + aspect_ratio * 5)
            
            signature_candidates.append({
                'contour': contour,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'density': density,
                'confidence': confidence
            })
    
    # Check if we found any signature candidates
    if signature_candidates:
        # Sort by confidence
        signature_candidates.sort(key=lambda c: c['confidence'], reverse=True)
        best_candidate = signature_candidates[0]
        
        # Extract the signature region
        x, y, w, h = best_candidate['x'], best_candidate['y'], best_candidate['width'], best_candidate['height']
        
        # Add padding to the region
        pad = 10
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(width, x + w + pad)
        y_end = min(height, y + h + pad)
        
        signature_region = gray[y_start:y_end, x_start:x_end]
        
        result['signature_detected'] = True
        result['location'] = (x, y, w, h)
        result['dimensions'] = (w, h)
        result['confidence'] = best_candidate['confidence']
        result['analysis'] = {
            'aspect_ratio': best_candidate['aspect_ratio'],
            'density': best_candidate['density'],
            'relative_size': (w * h) / (width * height)
        }
        
        return signature_region, result
    
    return None, result

def extract_signature_features(signature_image) -> Dict[str, Any]:
    """
    Extract features from a signature image that can be used for analysis.
    
    Args:
        signature_image: Grayscale image of the signature region
        
    Returns:
        Dictionary of signature features
    """
    if signature_image is None:
        return {'error': 'No signature image provided'}
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(signature_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Get signature density
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    density = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Get contours for more detailed analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate contour-based metrics
    contour_count = len(contours)
    
    # Calculate convex hull
    hull_area = 0
    contour_area = 0
    
    if contours:
        for contour in contours:
            contour_area += cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area += cv2.contourArea(hull)
    
    # Calculate solidity (area / hull area)
    solidity = contour_area / hull_area if hull_area > 0 else 0
    
    # Calculate average line thickness
    # This is a simple approximation using erosion
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    thickness_estimate = (np.sum(binary == 255) - np.sum(eroded == 255)) / max(1, np.sum(binary == 255))
    
    # Create skeleton to estimate stroke count
    skeleton = _create_skeleton(binary)
    branch_points = _find_branch_points(skeleton)
    
    # Calculate Fourier descriptors
    fourier_descriptors = []
    if contours and len(contours[0]) >= 10:  # Need enough points
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 10:
            fourier_descriptors = _calculate_fourier_descriptors(largest_contour)
    
    # Return feature dictionary
    return {
        'density': density,
        'contour_count': contour_count,
        'solidity': solidity,
        'thickness_estimate': thickness_estimate,
        'branch_points': len(branch_points),
        'signature_complexity': len(branch_points) * contour_count / 100,
        'fourier_descriptors': fourier_descriptors,
        'dimensions': signature_image.shape,
        'pixel_sum': white_pixels
    }

def _create_skeleton(binary_image) -> np.ndarray:
    """Create a skeleton of the binary image using morphological operations."""
    skeleton = np.zeros(binary_image.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_image.copy()
    
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
            
    return skeleton

def _find_branch_points(skeleton) -> List[Tuple[int, int]]:
    """Find branch points in a skeleton image."""
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    branch_points = zip(*np.where(filtered >= 13))
    return list(branch_points)

def _calculate_fourier_descriptors(contour, n_descriptors=10) -> List[float]:
    """Calculate Fourier descriptors for a contour."""
    # Convert contour to complex numbers (x + jy)
    contour = contour.squeeze()
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    
    # Apply Fourier transform
    fourier_result = np.fft.fft(complex_contour)
    
    # Take magnitude of coefficients and normalize
    magnitudes = np.abs(fourier_result)
    
    # Normalize by the DC component
    if magnitudes[0] > 0:
        magnitudes = magnitudes / magnitudes[0]
    
    # Return the first n_descriptors (skip DC component)
    return magnitudes[1:n_descriptors+1].tolist()

def compare_signatures(signature1_features, signature2_features) -> Dict[str, Any]:
    """
    Compare two signature feature sets to determine if they're from the same person.
    
    Args:
        signature1_features: Features from first signature
        signature2_features: Features from second signature
        
    Returns:
        Dictionary with comparison results
    """
    if 'error' in signature1_features or 'error' in signature2_features:
        return {'match': False, 'confidence': 0, 'error': 'Invalid signature features'}
    
    # Compare basic metrics
    density_diff = abs(signature1_features['density'] - signature2_features['density'])
    thickness_diff = abs(signature1_features['thickness_estimate'] - signature2_features['thickness_estimate'])
    complexity_diff = abs(signature1_features['signature_complexity'] - signature2_features['signature_complexity'])
    
    # Compare Fourier descriptors if available
    fourier_diff = 1.0  # Default to maximum difference
    if ('fourier_descriptors' in signature1_features and 
        'fourier_descriptors' in signature2_features and
        signature1_features['fourier_descriptors'] and
        signature2_features['fourier_descriptors']):
        
        # Get the minimum length of both descriptor sets
        min_length = min(len(signature1_features['fourier_descriptors']), 
                         len(signature2_features['fourier_descriptors']))
        
        if min_length > 0:
            # Calculate Euclidean distance
            descriptors1 = np.array(signature1_features['fourier_descriptors'][:min_length])
            descriptors2 = np.array(signature2_features['fourier_descriptors'][:min_length])
            
            fourier_diff = np.sqrt(np.sum((descriptors1 - descriptors2) ** 2)) / min_length
    
    # Calculate weighted score
    weights = {
        'density': 0.15,
        'thickness': 0.15,
        'complexity': 0.3,
        'fourier': 0.4
    }
    
    # Calculate similarity (0-1)
    density_sim = max(0, 1 - 2 * density_diff)
    thickness_sim = max(0, 1 - 5 * thickness_diff)
    complexity_sim = max(0, 1 - complexity_diff)
    fourier_sim = max(0, 1 - fourier_diff)
    
    # Overall similarity score
    similarity = (
        weights['density'] * density_sim +
        weights['thickness'] * thickness_sim +
        weights['complexity'] * complexity_sim +
        weights['fourier'] * fourier_sim
    )
    
    # Convert to percentage
    match_confidence = min(100, similarity * 100)
    
    return {
        'match': match_confidence > 70,
        'confidence': match_confidence,
        'metrics': {
            'density_similarity': density_sim * 100,
            'thickness_similarity': thickness_sim * 100,
            'complexity_similarity': complexity_sim * 100,
            'fourier_similarity': fourier_sim * 100
        }
    }

def verify_prescription_signature(prescription_image, reference_signatures=None) -> Dict[str, Any]:
    """
    Verify a signature on a prescription against reference signatures or general characteristics.
    
    Args:
        prescription_image: Image of the prescription
        reference_signatures: Optional list of reference signature images for comparison
        
    Returns:
        Dictionary with verification results
    """
    # Extract signature from prescription
    signature_region, detection_result = detect_signature_region(prescription_image)
    
    # If no signature detected, return early
    if not detection_result['signature_detected']:
        return {
            'verified': False,
            'confidence': 0,
            'reason': 'No signature detected on prescription',
            'detection_result': detection_result
        }
    
    # Extract features from the detected signature
    signature_features = extract_signature_features(signature_region)
    
    # If we have reference signatures, compare against them
    if reference_signatures:
        best_match = {
            'match': False,
            'confidence': 0
        }
        
        for ref_image in reference_signatures:
            ref_features = extract_signature_features(ref_image)
            comparison = compare_signatures(signature_features, ref_features)
            
            if comparison['confidence'] > best_match['confidence']:
                best_match = comparison
        
        return {
            'verified': best_match['match'],
            'confidence': best_match['confidence'],
            'signature_features': signature_features,
            'detection_result': detection_result,
            'comparison': best_match
        }
    
    # If no reference signatures, check general signature characteristics
    else:
        # Check general characteristics of a valid signature
        is_valid_signature = (
            # Not too small or too sparse
            signature_features['density'] > 0.05 and
            # Not too uniform (likely not a real signature)
            signature_features['contour_count'] > 1 and
            # Not too simple (likely not a real signature)
            signature_features['branch_points'] > 5
        )
        
        # Calculate confidence based on features
        confidence = min(80, 
                         40 + 
                         signature_features['contour_count'] * 2 +
                         signature_features['branch_points'] * 0.5)
        
        return {
            'verified': is_valid_signature,
            'confidence': confidence if is_valid_signature else 0,
            'signature_features': signature_features,
            'detection_result': detection_result,
            'note': 'No reference signatures provided, verification based on general characteristics only'
        }

def build_signature_database(signatures_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a database of signatures from a directory of signature images.
    
    Args:
        signatures_dir: Directory containing signature images
        
    Returns:
        Dictionary of signature name to features
    """
    database = {}
    
    # Check if directory exists
    if not os.path.exists(signatures_dir):
        return database
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for file_name in os.listdir(signatures_dir):
        file_path = os.path.join(signatures_dir, file_name)
        
        # Check if it's a file with valid extension
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in valid_extensions:
            try:
                # Load image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    # Extract features
                    features = extract_signature_features(image)
                    
                    # Store in database
                    name = os.path.splitext(file_name)[0]
                    database[name] = {
                        'features': features,
                        'file_path': file_path
                    }
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return database
