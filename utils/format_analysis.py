"""
Prescription format analysis utilities.

This module contains functions to analyze the format and structure of prescriptions
to determine if they match standard medical prescription formats.
"""

import re
from typing import Dict, Any, List, Optional

# Regions typically found on a legitimate prescription
PRESCRIPTION_REGIONS = [
    'header',       # Hospital/doctor name and contact
    'patient_info', # Patient name, age, gender
    'rx_symbol',    # The Rx symbol
    'medications',  # List of medications
    'directions',   # How to take medications
    'signature',    # Doctor's signature
    'registration', # Doctor's registration number
    'date',         # Date of prescription
]

# Standard header patterns for legitimate prescriptions
HEADER_PATTERNS = [
    r'(hospital|clinic|medical center|healthcare)',
    r'dr\.?\s+[a-z]+\s+[a-z]+',
    r'(m\.?b\.?b\.?s|md|mbchb|do)',
    r'(ph|tel|phone)\s*[:\.]?\s*[\d\+\- ]{7,}',
    r'(fax)\s*[:\.]?\s*[\d\+\- ]{7,}',
]

# Standard footer patterns
FOOTER_PATTERNS = [
    r'signature',
    r'(m\.?d|mbbs)',
    r'(reg|registration)\.?\s*(no|num|number)?\.?\s*[\d\w]+',
    r'(valid until|valid for|refill)',
]

def check_prescription_format(text: str) -> Dict[str, Any]:
    """
    Analyze a prescription's format to determine if it matches standard medical formats.
    
    Args:
        text: Extracted text from the prescription image
        
    Returns:
        Dictionary with format analysis results
    """
    lines = text.strip().split('\n')
    result = {
        'format_score': 0,
        'max_score': 100,
        'regions_found': [],
        'regions_missing': [],
        'suspicious_patterns': [],
    }
    
    # Check for header elements (doctor/hospital info)
    header_score = _analyze_header(lines[:5])
    result['header_score'] = header_score
    if header_score > 0:
        result['regions_found'].append('header')
    else:
        result['regions_missing'].append('header')
        result['suspicious_patterns'].append('Missing or invalid header (doctor/hospital information)')
    
    # Check for Rx symbol
    has_rx_symbol = _check_rx_symbol(text)
    if has_rx_symbol:
        result['regions_found'].append('rx_symbol')
    else:
        result['regions_missing'].append('rx_symbol')
        result['suspicious_patterns'].append('Missing Rx symbol (℞)')
    
    # Check for patient information
    patient_score = _analyze_patient_info(text)
    result['patient_score'] = patient_score
    if patient_score > 0:
        result['regions_found'].append('patient_info')
    else:
        result['regions_missing'].append('patient_info')
        result['suspicious_patterns'].append('Missing or incomplete patient information')
    
    # Check for medication list
    meds_score = _analyze_medication_section(text)
    result['medication_score'] = meds_score
    if meds_score > 0:
        result['regions_found'].append('medications')
    else:
        result['regions_missing'].append('medications')
        result['suspicious_patterns'].append('Missing or suspicious medication list format')
    
    # Check for doctor's signature indication
    signature_score = _check_signature_indicator(text)
    result['signature_score'] = signature_score
    if signature_score > 0:
        result['regions_found'].append('signature')
    else:
        result['regions_missing'].append('signature')
        result['suspicious_patterns'].append('No indication of doctor signature')
    
    # Check for date
    date_score = _check_date(text)
    result['date_score'] = date_score
    if date_score > 0:
        result['regions_found'].append('date')
    else:
        result['regions_missing'].append('date')
        result['suspicious_patterns'].append('Missing or invalid date')
    
    # Check for doctor registration number
    reg_score = _check_registration_number(text)
    result['registration_score'] = reg_score
    if reg_score > 0:
        result['regions_found'].append('registration')
    else:
        result['regions_missing'].append('registration')
        result['suspicious_patterns'].append('Missing or invalid doctor registration number')
    
    # Calculate overall format score
    # Different weights for different components
    weights = {
        'header': 15,
        'rx_symbol': 10,
        'patient_info': 15,
        'medications': 25,
        'signature': 15,
        'date': 10,
        'registration': 10,
    }
    
    total_score = 0
    for region in result['regions_found']:
        total_score += weights.get(region, 0)
    
    result['format_score'] = total_score
    
    # Add format verdict
    if total_score >= 80:
        result['format_verdict'] = 'Standard prescription format'
    elif total_score >= 60:
        result['format_verdict'] = 'Acceptable prescription format with minor issues'
    else:
        result['format_verdict'] = 'Non-standard prescription format - suspicious'
    
    return result

def _analyze_header(header_lines: List[str]) -> int:
    """
    Analyze the header of a prescription.
    
    Args:
        header_lines: First few lines of the prescription
        
    Returns:
        Score representing legitimacy of the header (0-15)
    """
    score = 0
    header_text = ' '.join(header_lines).lower()
    
    # Check for expected patterns in the header
    for pattern in HEADER_PATTERNS:
        if re.search(pattern, header_text, re.IGNORECASE):
            score += 3
    
    return min(score, 15)  # Cap at 15

def _check_rx_symbol(text: str) -> bool:
    """
    Check if the Rx symbol is present.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        True if Rx symbol found, False otherwise
    """
    # Look for actual Rx symbol or text representation
    return bool(re.search(r'℞|rx|prescription|script', text, re.IGNORECASE))

def _analyze_patient_info(text: str) -> int:
    """
    Check for patient information in the prescription.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        Score for patient information (0-15)
    """
    score = 0
    
    # Check for patient name
    if re.search(r'(patient|name)[:\s]+\w+', text, re.IGNORECASE):
        score += 5
    
    # Check for patient age/date of birth
    if re.search(r'(age|dob|date of birth|birth date)[:\s]+', text, re.IGNORECASE):
        score += 5
    
    # Check for patient gender
    if re.search(r'(gender|sex)[:\s]+(m|f|male|female)', text, re.IGNORECASE):
        score += 5
    
    return score

def _analyze_medication_section(text: str) -> int:
    """
    Analyze the medication section of the prescription.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        Score for medication section format (0-25)
    """
    score = 0
    
    # Check for numbered list of medications
    if re.search(r'(1|1\.)\s+\w+', text):
        score += 10
    
    # Check for medication with dosage
    if re.search(r'\d+\s*(mg|ml|mcg|g)', text, re.IGNORECASE):
        score += 10
    
    # Check for medication frequency
    if re.search(r'(once|twice|thrice|daily|weekly|monthly|every|q\.d|b\.i\.d|t\.i\.d|q\.i\.d)', text, re.IGNORECASE):
        score += 5
    
    return score

def _check_signature_indicator(text: str) -> int:
    """
    Check for signature indicator in the prescription.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        Score for signature section (0-15)
    """
    score = 0
    
    # Check for signature keywords
    if re.search(r'(sign|signature|signed)', text, re.IGNORECASE):
        score += 7
    
    # Check for doctor title with signature
    if re.search(r'(dr|doctor)\.?\s+\w+', text, re.IGNORECASE):
        score += 8
    
    return score

def _check_date(text: str) -> int:
    """
    Check for a valid date on the prescription.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        Score for date presence (0-10)
    """
    # Simple date pattern - can be expanded for more specific formats
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
        r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}',  # DD Month YYYY
        r'(date)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date: DD/MM/YYYY
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 10
    
    return 0

def _check_registration_number(text: str) -> int:
    """
    Check for doctor's registration number.
    
    Args:
        text: Extracted text from the prescription
        
    Returns:
        Score for registration number presence (0-10)
    """
    # Patterns for registration numbers
    reg_patterns = [
        r'(reg|registration)\.?\s*(no|num|number)?\.?\s*[\d\w]+',
        r'(reg|registration)\.?\s*(no|num|number)?\.?\s*[:]\s*[\d\w]+',
        r'doctor[\'s]?\s+(reg|registration)',
    ]
    
    for pattern in reg_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 10
    
    return 0

def analyze_prescription_template(text: str, hospital_templates: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Compare prescription to known hospital templates to check for legitimacy.
    
    Args:
        text: Extracted text from the prescription
        hospital_templates: Dictionary of hospital names to template signatures
        
    Returns:
        Dictionary with template analysis results
    """
    result = {
        'template_match': None,
        'confidence': 0,
        'analysis': check_prescription_format(text)
    }
    
    # If we have hospital templates to check against
    if hospital_templates:
        for hospital, template in hospital_templates.items():
            # Simple text similarity check (would be more sophisticated in production)
            similarity = _calculate_text_similarity(text, template)
            
            if similarity > result['confidence']:
                result['confidence'] = similarity
                result['template_match'] = hospital
    
    # Additional format checks
    suspicious_patterns = []
    
    # Check for consistent font/formatting
    lines = text.strip().split('\n')
    if len(lines) > 3:
        if len(lines[0]) > 50 and all(len(line) < 20 for line in lines[1:4]):
            suspicious_patterns.append("Inconsistent text formatting - possible tampering")
    
    # Check for standard prescription phrases
    common_phrases = [
        'take as directed', 'take with food', 'take on empty stomach',
        'refill', 'substitution permitted', 'no substitutions',
        'dispense as written', 'may refill'
    ]
    
    phrase_count = 0
    for phrase in common_phrases:
        if phrase in text.lower():
            phrase_count += 1
    
    if phrase_count == 0:
        suspicious_patterns.append("Missing standard prescription instructions")
    
    result['suspicious_patterns'] = suspicious_patterns + result['analysis']['suspicious_patterns']
    
    return result

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate a simple similarity score between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score from 0-100
    """
    # In a real implementation, this would use more sophisticated algorithms
    # like TF-IDF or word embeddings. This is a simplified version.
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0
    
    return (len(intersection) / len(union)) * 100
