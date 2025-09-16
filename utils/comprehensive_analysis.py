"""
Comprehensive prescription analysis integrating multiple verification methods.

This module coordinates all the different analysis techniques to provide a complete
assessment of a prescription's authenticity.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

# Import specialized analysis modules
from utils.image_forensics import analyze_image_forensics
from utils.drug_analysis import analyze_prescription_medications
from utils.format_analysis import analyze_prescription_template
from utils.signature_verification import verify_prescription_signature
from utils.json_utils import convert_numpy_types

def extract_medicines_from_text(text: str) -> List[str]:
    """
    Extract medicine names from prescription text.
    
    Args:
        text: OCR extracted text from prescription
        
    Returns:
        List of medicine names
    """
    medicines = []
    
    # Look for common medicine patterns
    # Pattern 1: Numbered list (e.g., "1. Medicine Name 500mg")
    numbered_pattern = r'(?:^|\n)(?:\d+\.?\s+)([A-Za-z\s\-]+(?:\d+\s*(?:mg|ml|mcg))?)'
    
    # Pattern 2: Medicine with dosage (e.g., "Medicine Name 500mg")
    dosage_pattern = r'(?:^|\n)([A-Za-z\s\-]+)\s+(\d+\s*(?:mg|ml|mcg))'
    
    # Pattern 3: "Tab" or "Cap" prefix (e.g., "Tab Medicine Name 500mg")
    tab_pattern = r'(?:Tab|Cap|Inj|Syp)\.?\s+([A-Za-z\s\-]+(?:\s+\d+\s*(?:mg|ml|mcg))?)'
    
    # Extract using patterns
    for pattern in [numbered_pattern, dosage_pattern, tab_pattern]:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            medicine = match.group(1).strip()
            if medicine and len(medicine) > 3:  # Minimum length to avoid OCR errors
                medicines.append(medicine)
    
    # If no medicines found with patterns, try line-by-line extraction
    if not medicines:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line looks like a medicine entry
            if (len(line) > 5 and                                # Not too short
                len(line.split()) <= 6 and                       # Not too many words (likely not a medicine)
                re.search(r'[A-Za-z]{3,}', line) and             # Contains alphabetic characters
                not re.match(r'(patient|name|age|sex|dr|sign|hospital)', line.lower())):  # Not header/footer
                
                medicines.append(line)
    
    # Remove duplicates
    return list(set(medicines))

def extract_doctor_details(text: str) -> Dict[str, Any]:
    """
    Extract doctor information from prescription text.
    
    Args:
        text: OCR extracted text from prescription
        
    Returns:
        Dictionary with doctor details
    """
    result = {
        'name': None,
        'registration_number': None,
        'specialization': None,
        'hospital': None
    }
    
    # Look for doctor name
    dr_pattern = r'(?:Dr|DR|Doctor)\.?\s+([A-Z][A-Za-z\s\-\.]{2,})'
    dr_match = re.search(dr_pattern, text)
    if dr_match:
        result['name'] = dr_match.group(1).strip()
    
    # Look for registration number
    reg_pattern = r'(?:Reg|Registration|Regd)\.?\s*(?:No|Number|#)?\.?\s*[:\.]?\s*((?:[A-Z]+/)?[A-Z0-9]+/?[0-9]+)'
    reg_match = re.search(reg_pattern, text, re.IGNORECASE)
    if reg_match:
        result['registration_number'] = reg_match.group(1).strip()
    
    # Look for specialization
    specializations = [
        'Cardiologist', 'Dermatologist', 'Neurologist', 'Oncologist', 
        'Pediatrician', 'Psychiatrist', 'Surgeon', 'Ophthalmologist',
        'Gynecologist', 'Orthopedic', 'ENT', 'Dentist', 'General Physician'
    ]
    
    for spec in specializations:
        if spec.lower() in text.lower():
            result['specialization'] = spec
            break
    
    # Look for hospital/clinic name
    hospital_pattern = r'([A-Z][A-Za-z\s\.]{2,}(?:Hospital|Clinic|Medical|Healthcare|Center|Centre))'
    hospital_match = re.search(hospital_pattern, text)
    if hospital_match:
        result['hospital'] = hospital_match.group(1).strip()
    
    return result

def extract_patient_details(text: str) -> Dict[str, Any]:
    """
    Extract patient information from prescription text.
    
    Args:
        text: OCR extracted text from prescription
        
    Returns:
        Dictionary with patient details
    """
    result = {
        'name': None,
        'age': None,
        'gender': None,
        'date': None
    }
    
    # Look for patient name
    name_pattern = r'(?:Patient(?:\'s)?|Name)\s*[:\.]?\s*([A-Z][A-Za-z\s\-\.]{2,})'
    name_match = re.search(name_pattern, text, re.IGNORECASE)
    if name_match:
        result['name'] = name_match.group(1).strip()
    
    # Look for age
    age_pattern = r'(?:Age|Years)\s*[:\.]?\s*(\d{1,3})'
    age_match = re.search(age_pattern, text, re.IGNORECASE)
    if age_match:
        result['age'] = age_match.group(1).strip()
    
    # Look for gender
    gender_pattern = r'(?:Gender|Sex)\s*[:\.]?\s*([MF]|Male|Female)'
    gender_match = re.search(gender_pattern, text, re.IGNORECASE)
    if gender_match:
        result['gender'] = gender_match.group(1).strip()
    
    # Look for date
    date_patterns = [
        r'Date\s*[:\.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            result['date'] = date_match.group(1).strip()
            break
    
    return result

def calculate_prescription_age(date_text: Optional[str]) -> Dict[str, Any]:
    """
    Calculate the age of a prescription based on its date.
    
    Args:
        date_text: Date string from prescription
        
    Returns:
        Dictionary with prescription age analysis
    """
    if not date_text:
        return {
            'valid': False,
            'days_old': None,
            'expired': True,
            'reason': "No date found on prescription"
        }
    
    try:
        # Try multiple date formats
        date_formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
            '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%m-%d-%y'
        ]
        
        parsed_date = None
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_text, date_format)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            return {
                'valid': False,
                'days_old': None,
                'expired': True,
                'reason': f"Could not parse date: {date_text}"
            }
        
        # Calculate days since prescription
        days_old = (datetime.now() - parsed_date).days
        
        # Check if prescription is expired (typically valid for 6 months)
        is_expired = days_old > 180
        
        return {
            'valid': True,
            'date': parsed_date.strftime('%Y-%m-%d'),
            'days_old': days_old,
            'expired': is_expired,
            'reason': f"Prescription is {days_old} days old" + 
                      (", which exceeds the standard 6-month validity" if is_expired else "")
        }
        
    except Exception as e:
        return {
            'valid': False,
            'days_old': None,
            'expired': True,
            'reason': f"Error processing date: {str(e)}"
        }

def verify_registration_number(reg_number: Optional[str]) -> Dict[str, Any]:
    """
    Verify doctor's registration number format and validity.
    
    Args:
        reg_number: Registration number to verify
        
    Returns:
        Dictionary with verification results
    """
    if not reg_number:
        return {
            'valid': False,
            'confidence': 0,
            'reason': "No registration number found"
        }
    
    # Common formats for medical registration numbers in different countries
    patterns = {
        # Format: (pattern, description, confidence if matched)
        'india_mci': (r'^[A-Z]+/\d+/\d+$', "Medical Council of India", 90),
        'india_state': (r'^[A-Z]+/\d+$', "Indian State Medical Council", 85),
        'uk_gmc': (r'^[0-9]{7}$', "UK General Medical Council", 90),
        'us_npi': (r'^\d{10}$', "US National Provider Identifier", 90),
        'general_numeric': (r'^\d{5,10}$', "General numeric format", 60),
        'alphanumeric': (r'^[A-Z0-9]{5,15}$', "Alphanumeric format", 50)
    }
    
    # Check registration number against known patterns
    for pattern_name, (pattern, description, confidence) in patterns.items():
        if re.match(pattern, reg_number):
            return {
                'valid': True,
                'confidence': confidence,
                'format': pattern_name,
                'description': description,
                'number': reg_number
            }
    
    # If we get here, the format wasn't recognized but might still be valid
    return {
        'valid': False,
        'confidence': 30,
        'format': 'unknown',
        'description': "Format not recognized in common medical registration patterns",
        'number': reg_number
    }

def comprehensive_prescription_analysis(
    image_path: str, 
    ocr_text: str,
    reference_database: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a prescription image using multiple techniques.
    
    Args:
        image_path: Path to the prescription image
        ocr_text: OCR extracted text from the prescription
        reference_database: Optional reference data for verification
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return {'error': f"Could not load image: {image_path}"}
    
    # Initialize result dictionary
    result = {
        'prescription_id': os.path.basename(image_path),
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_verdict': {
            'is_genuine': False,
            'confidence': 0,
            'suspicious_flags': []
        }
    }
    
    # 1. Extract medicines, doctor and patient details from OCR text
    result['medicines'] = extract_medicines_from_text(ocr_text)
    result['doctor_details'] = extract_doctor_details(ocr_text)
    result['patient_details'] = extract_patient_details(ocr_text)
    
    # 2. Verify prescription date
    date_analysis = calculate_prescription_age(result['patient_details'].get('date'))
    result['date_analysis'] = date_analysis
    
    if date_analysis.get('expired', True):
        result['overall_verdict']['suspicious_flags'].append(
            "Prescription is expired or has invalid date"
        )
    
    # 3. Verify doctor's registration number
    reg_verification = verify_registration_number(result['doctor_details'].get('registration_number'))
    result['registration_verification'] = reg_verification
    
    if not reg_verification.get('valid', False):
        result['overall_verdict']['suspicious_flags'].append(
            "Invalid or missing doctor registration number"
        )
    
    # 4. Analyze prescription format
    format_analysis = analyze_prescription_template(ocr_text)
    result['format_analysis'] = format_analysis
    
    if format_analysis['analysis']['format_score'] < 60:
        result['overall_verdict']['suspicious_flags'].append(
            "Prescription format does not match standard medical formats"
        )
    
    # 5. Medication analysis
    if result['medicines']:
        med_analysis = analyze_prescription_medications(result['medicines'])
        result['medication_analysis'] = med_analysis
        
        # Add flags from medication analysis
        if med_analysis.get('is_suspicious', False):
            result['overall_verdict']['suspicious_flags'].extend(
                med_analysis.get('suspicious_flags', [])
            )
    
    # 6. Image forensics analysis
    forensics_result = analyze_image_forensics(image_path)
    result['image_forensics'] = forensics_result
    
    if forensics_result.get('is_manipulated', False):
        result['overall_verdict']['suspicious_flags'].append(
            "Image shows signs of digital manipulation"
        )
    
    # 7. Signature verification
    signature_result = verify_prescription_signature(image)
    result['signature_verification'] = signature_result
    
    if not signature_result.get('verified', False):
        result['overall_verdict']['suspicious_flags'].append(
            "Signature verification failed or no valid signature detected"
        )
    
    # Calculate overall confidence and verdict
    # Start with 100 and deduct based on issues found
    overall_confidence = 100
    
    # Format issues
    format_score = format_analysis['analysis']['format_score']
    overall_confidence -= max(0, (80 - format_score) * 0.5)
    
    # Registration issues
    if not reg_verification.get('valid', False):
        overall_confidence -= 25
    
    # Medication issues
    if 'medication_analysis' in result and result['medication_analysis'].get('is_suspicious', False):
        overall_confidence -= 15
    
    # Image manipulation issues
    if forensics_result.get('is_manipulated', False):
        overall_confidence -= 25
    
    # Signature issues
    if not signature_result.get('verified', False):
        overall_confidence -= 15
    
    # Date issues
    if date_analysis.get('expired', True):
        overall_confidence -= 10
    
    # Determine final verdict
    result['overall_verdict']['confidence'] = max(0, min(100, overall_confidence))
    result['overall_verdict']['is_genuine'] = overall_confidence >= 70
    
    # Convert all NumPy types to Python native types before returning
    return convert_numpy_types(result)

def extract_forensic_highlights(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key highlights from comprehensive analysis for display.
    
    Args:
        analysis_results: Full analysis results
        
    Returns:
        Dictionary with key highlights
    """
    # First, ensure all NumPy types are converted to Python native types
    analysis_results = convert_numpy_types(analysis_results)
    
    # Create highlights dictionary with converted values
    highlights = {
        'confidence_score': analysis_results['overall_verdict']['confidence'],
        'is_genuine': analysis_results['overall_verdict']['is_genuine'],
        'suspicious_flags': analysis_results['overall_verdict']['suspicious_flags'],
        'key_findings': []
    }
    
    # Add medication findings
    if 'medication_analysis' in analysis_results:
        med_analysis = analysis_results['medication_analysis']
        
        if med_analysis.get('drug_interactions', {}).get('interactions_found', False):
            interactions = med_analysis['drug_interactions']['interactions']
            highlights['key_findings'].append({
                'type': 'medication',
                'title': 'Drug Interactions Found',
                'description': f"{len(interactions)} potentially dangerous drug interactions detected"
            })
        
        controlled_substances = med_analysis.get('drug_interactions', {}).get('controlled_substances', [])
        if controlled_substances:
            highlights['key_findings'].append({
                'type': 'medication',
                'title': 'Controlled Substances',
                'description': f"{len(controlled_substances)} controlled substances found in prescription"
            })
    
    # Add forensic findings
    if 'image_forensics' in analysis_results:
        forensics = analysis_results['image_forensics']
        
        if forensics.get('metadata_suspicious', False):
            highlights['key_findings'].append({
                'type': 'forensic',
                'title': 'Suspicious Metadata',
                'description': "Image metadata indicates possible editing"
            })
            
        if forensics.get('ela_analysis', {}).get('suspicious', False):
            highlights['key_findings'].append({
                'type': 'forensic',
                'title': 'Error Level Analysis',
                'description': "Image shows signs of digital manipulation"
            })
            
        if forensics.get('font_consistency', {}).get('is_consistent', False) == False:
            highlights['key_findings'].append({
                'type': 'forensic',
                'title': 'Font Inconsistency',
                'description': "Multiple font types detected, possible tampering"
            })
    
    # Add format findings
    if 'format_analysis' in analysis_results:
        format_analysis = analysis_results['format_analysis']
        
        if format_analysis['analysis']['format_score'] < 60:
            highlights['key_findings'].append({
                'type': 'format',
                'title': 'Non-standard Format',
                'description': f"Prescription format score: {format_analysis['analysis']['format_score']}/100"
            })
            
        missing_regions = format_analysis['analysis'].get('regions_missing', [])
        if missing_regions:
            highlights['key_findings'].append({
                'type': 'format',
                'title': 'Missing Elements',
                'description': f"Prescription missing: {', '.join(missing_regions)}"
            })
    
    # Add signature findings
    if 'signature_verification' in analysis_results:
        sig_verify = analysis_results['signature_verification']
        
        if not sig_verify.get('verified', False):
            highlights['key_findings'].append({
                'type': 'signature',
                'title': 'Signature Issues',
                'description': "Signature verification failed or signature missing"
            })
    
    # Add registration findings
    if 'registration_verification' in analysis_results:
        reg_verify = analysis_results['registration_verification']
        
        if not reg_verify.get('valid', False):
            highlights['key_findings'].append({
                'type': 'registration',
                'title': 'Registration Issues',
                'description': reg_verify.get('reason', "Invalid doctor registration")
            })
    
    return highlights
