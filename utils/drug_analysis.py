"""
Drug interaction analysis utilities for prescription verification.
"""
from typing import List, Dict, Any, Tuple
import re

# Mock database of drug interactions
# In a real application, this would be connected to a pharmaceutical database
DRUG_INTERACTIONS = {
    # Format: (drug1, drug2): {'severity': 'high|moderate|low', 'description': 'interaction details'}
    ('warfarin', 'aspirin'): {
        'severity': 'high',
        'description': 'Increased risk of bleeding when taken together'
    },
    ('lisinopril', 'potassium'): {
        'severity': 'moderate',
        'description': 'May increase potassium levels causing hyperkalemia'
    },
    ('simvastatin', 'erythromycin'): {
        'severity': 'high',
        'description': 'Increased risk of muscle damage and rhabdomyolysis'
    },
    ('fluoxetine', 'tramadol'): {
        'severity': 'high',
        'description': 'Increased risk of serotonin syndrome'
    },
    ('metformin', 'iodinated contrast'): {
        'severity': 'moderate',
        'description': 'Increased risk of lactic acidosis'
    },
    ('ciprofloxacin', 'dairy'): {
        'severity': 'low',
        'description': 'Reduced absorption of ciprofloxacin'
    },
    ('levothyroxine', 'calcium'): {
        'severity': 'low',
        'description': 'Reduced absorption of levothyroxine'
    },
    ('alprazolam', 'alcohol'): {
        'severity': 'high',
        'description': 'Severe central nervous system depression'
    },
    ('sildenafil', 'nitrates'): {
        'severity': 'high',
        'description': 'Severe hypotension that can be life-threatening'
    },
    ('clarithromycin', 'simvastatin'): {
        'severity': 'high',
        'description': 'Increased risk of myopathy and rhabdomyolysis'
    }
}

# Mock database of controlled substances
CONTROLLED_SUBSTANCES = {
    'morphine': {
        'schedule': 'II',
        'typical_dosage': '15-30 mg every 4 hours',
        'flags': ['requires special prescription', 'high abuse potential']
    },
    'oxycodone': {
        'schedule': 'II',
        'typical_dosage': '5-15 mg every 4-6 hours',
        'flags': ['requires special prescription', 'high abuse potential']
    },
    'fentanyl': {
        'schedule': 'II',
        'typical_dosage': '25-100 mcg patch every 72 hours',
        'flags': ['requires special prescription', 'high abuse potential', 'high overdose risk']
    },
    'adderall': {
        'schedule': 'II',
        'typical_dosage': '5-40 mg daily',
        'flags': ['requires special prescription', 'abuse potential']
    },
    'xanax': {
        'schedule': 'IV',
        'typical_dosage': '0.25-0.5 mg three times daily',
        'flags': ['controlled substance', 'moderate abuse potential']
    },
    'valium': {
        'schedule': 'IV',
        'typical_dosage': '2-10 mg 2-4 times daily',
        'flags': ['controlled substance', 'moderate abuse potential']
    },
    'ambien': {
        'schedule': 'IV',
        'typical_dosage': '5-10 mg at bedtime',
        'flags': ['controlled substance', 'sleep aid']
    },
    'ritalin': {
        'schedule': 'II',
        'typical_dosage': '20-30 mg daily',
        'flags': ['requires special prescription', 'abuse potential']
    }
}

# Mock database of common prescription drugs and their details
COMMON_DRUGS = {
    'lisinopril': {
        'class': 'ACE inhibitor',
        'common_use': 'hypertension',
        'typical_dosage': '10-40 mg once daily'
    },
    'metformin': {
        'class': 'biguanide',
        'common_use': 'type 2 diabetes',
        'typical_dosage': '500-1000 mg twice daily'
    },
    'atorvastatin': {
        'class': 'statin',
        'common_use': 'high cholesterol',
        'typical_dosage': '10-80 mg once daily'
    },
    'levothyroxine': {
        'class': 'thyroid hormone',
        'common_use': 'hypothyroidism',
        'typical_dosage': '50-200 mcg once daily'
    },
    'amoxicillin': {
        'class': 'penicillin antibiotic',
        'common_use': 'bacterial infections',
        'typical_dosage': '250-500 mg three times daily'
    },
    'omeprazole': {
        'class': 'proton pump inhibitor',
        'common_use': 'acid reflux',
        'typical_dosage': '20-40 mg once daily'
    },
    'paracetamol': {
        'class': 'analgesic',
        'common_use': 'pain relief',
        'typical_dosage': '500-1000 mg every 4-6 hours'
    },
    'ibuprofen': {
        'class': 'NSAID',
        'common_use': 'pain and inflammation',
        'typical_dosage': '200-400 mg every 4-6 hours'
    }
}

def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize drug names for consistent comparison.
    
    Args:
        drug_name: Raw drug name from prescription
        
    Returns:
        Normalized drug name
    """
    # Remove dose information
    drug_name = re.sub(r'\d+\s*mg|\d+\s*ml|\d+\s*mcg', '', drug_name)
    
    # Remove common suffixes
    drug_name = re.sub(r'tablet(s)?|capsule(s)?|injection|syrup|ointment|cream', '', drug_name)
    
    # Remove extra spaces and convert to lowercase
    return drug_name.strip().lower()

def identify_drug_from_text(drug_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Try to identify a drug from prescription text and get its information.
    
    Args:
        drug_text: Text describing the drug from prescription
        
    Returns:
        Tuple of (normalized_name, drug_info)
    """
    normalized = normalize_drug_name(drug_text)
    
    # Try to match with known drugs
    all_drugs = {**CONTROLLED_SUBSTANCES, **COMMON_DRUGS}
    
    # Find best match (simple contains logic)
    for drug_name in all_drugs:
        if drug_name in normalized:
            return drug_name, all_drugs[drug_name]
        
    # Try partial matching if no exact match
    for drug_name in all_drugs:
        # Check if significant part of drug name is in normalized text
        if len(drug_name) > 3 and drug_name[:4] in normalized:
            return drug_name, all_drugs[drug_name]
    
    # No match found
    return normalized, {}

def check_drug_interactions(medications: List[str]) -> Dict[str, Any]:
    """
    Check for potential drug interactions in a list of medications.
    
    Args:
        medications: List of medication descriptions from prescription
        
    Returns:
        Dictionary with interaction analysis
    """
    result = {
        'interactions_found': False,
        'interactions': [],
        'controlled_substances': [],
        'normalized_drugs': [],
        'suspicious_flags': []
    }
    
    # Normalize and identify drugs
    identified_drugs = []
    for med in medications:
        drug_name, drug_info = identify_drug_from_text(med)
        if drug_name:
            identified_drugs.append((drug_name, drug_info))
            result['normalized_drugs'].append(drug_name)
    
    # Check for controlled substances
    for drug_name, drug_info in identified_drugs:
        if drug_name in CONTROLLED_SUBSTANCES:
            result['controlled_substances'].append({
                'name': drug_name,
                'info': CONTROLLED_SUBSTANCES[drug_name]
            })
            
            # Check if prescription mentions dosage for controlled substances
            dosage_pattern = fr'{drug_name}.*?(\d+\s*mg|\d+\s*mcg)'
            if not re.search(dosage_pattern, ' '.join(medications), re.IGNORECASE):
                result['suspicious_flags'].append(f"Missing or unclear dosage for controlled substance: {drug_name}")
    
    # Check for interactions
    drug_names = [d[0] for d in identified_drugs]
    
    # Check all possible pairs of drugs for interactions
    for i, drug1 in enumerate(drug_names):
        for drug2 in drug_names[i+1:]:
            # Check both orders of the pair
            interaction = DRUG_INTERACTIONS.get((drug1, drug2)) or DRUG_INTERACTIONS.get((drug2, drug1))
            
            if interaction:
                result['interactions_found'] = True
                result['interactions'].append({
                    'drugs': [drug1, drug2],
                    'severity': interaction['severity'],
                    'description': interaction['description']
                })
                
                # Flag high severity interactions as suspicious
                if interaction['severity'] == 'high':
                    result['suspicious_flags'].append(
                        f"High-risk interaction between {drug1} and {drug2}: {interaction['description']}"
                    )
    
    return result

def analyze_prescription_medications(medications: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of medications in a prescription.
    
    Args:
        medications: List of medication descriptions from prescription
        
    Returns:
        Dictionary with medication analysis results
    """
    result = {
        'is_suspicious': False,
        'confidence': 100,  # Start with high confidence
        'drug_analysis': []
    }
    
    # Get drug interactions
    interactions = check_drug_interactions(medications)
    result['drug_interactions'] = interactions
    
    # Analyze individual drugs
    for med in medications:
        drug_name, drug_info = identify_drug_from_text(med)
        analysis = {
            'original_text': med,
            'normalized_name': drug_name,
            'identified': bool(drug_info),
            'is_controlled': drug_name in CONTROLLED_SUBSTANCES,
            'info': drug_info
        }
        
        # Check for suspicious patterns in medication text
        dosage_match = re.search(r'(\d+\s*mg|\d+\s*ml|\d+\s*mcg)', med)
        if not dosage_match and analysis['identified']:
            analysis['suspicious'] = True
            analysis['suspicious_reason'] = "No dosage information found"
            result['confidence'] -= 10
        
        result['drug_analysis'].append(analysis)
    
    # Set overall suspicious flag
    if interactions['suspicious_flags'] or result['confidence'] < 70:
        result['is_suspicious'] = True
        
    # Add interaction suspicious flags
    result['suspicious_flags'] = interactions['suspicious_flags']
    
    # Check for an unusual number of controlled substances
    if len(interactions['controlled_substances']) > 2:
        result['suspicious_flags'].append(f"Unusually high number of controlled substances ({len(interactions['controlled_substances'])})")
        result['is_suspicious'] = True
        result['confidence'] = max(10, result['confidence'] - 30)
    
    return result
