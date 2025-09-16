import os
import uuid
import cv2
import numpy as np
import pytesseract
import re
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import json
from flask.json import JSONEncoder

# Import our advanced analysis modules
from utils.drug_analysis import analyze_prescription_medications
from utils.format_analysis import analyze_prescription_template
from utils.image_forensics import analyze_image_forensics
from utils.signature_verification import verify_prescription_signature
from utils.comprehensive_analysis import comprehensive_prescription_analysis, extract_forensic_highlights
from utils.json_utils import convert_numpy_types

# Custom JSON encoder to handle NumPy types
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Set the path to tesseract executable (change this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

app = Flask(__name__)
app.secret_key = 'prescription_analysis_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.json_encoder = CustomJSONEncoder  # Use our custom JSON encoder

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Use the utility function from json_utils.py
make_serializable = convert_numpy_types

# Mock database of doctor registration numbers
# In a real application, this would be an actual database or API call
DOCTOR_REGISTRY = {
    "MCI-123456": {
        "name": "Dr. Rajesh Kumar",
        "specialization": "General Medicine",
        "hospital": "City General Hospital",
        "valid": True,
        "phone": "9876543210"
    },
    "DMC-789012": {
        "name": "Dr. Priya Sharma",
        "specialization": "Pediatrics",
        "hospital": "Children's Medical Center",
        "valid": True,
        "phone": "8765432109"
    },
    "IMC-345678": {
        "name": "Dr. Anand Singh",
        "specialization": "Cardiology",
        "hospital": "Heart Institute",
        "valid": True,
        "phone": "7654321098"
    },
    "KMC-901234": {
        "name": "Dr. Lakshmi Nair",
        "specialization": "Dermatology",
        "hospital": "Skin & Care Clinic",
        "valid": True,
        "phone": "6543210987"
    },
    "MMC-567890": {
        "name": "Dr. Vikram Reddy",
        "specialization": "Orthopedics",
        "hospital": "Joint & Bone Hospital",
        "valid": False,  # Example of an invalid registration
        "phone": "5432109876"
    }
}

# Medical councils in India that issue registration numbers
MEDICAL_COUNCILS = [
    "Medical Council of India (MCI)",
    "Delhi Medical Council (DMC)",
    "Karnataka Medical Council (KMC)",
    "Maharashtra Medical Council (MMC)",
    "Tamil Nadu Medical Council (TNMC)",
    "Andhra Pradesh Medical Council (APMC)",
    "Uttar Pradesh Medical Council (UPMC)",
    "West Bengal Medical Council (WBMC)",
    "Gujarat Medical Council (GMC)",
    "Punjab Medical Council (PMC)"
]

def verify_registration_number(reg_number):
    """
    Verify if a doctor's registration number is valid
    Returns a tuple of (is_valid, details)
    """
    # First, check our local mock database
    if reg_number in DOCTOR_REGISTRY:
        doctor_info = DOCTOR_REGISTRY[reg_number]
        return (doctor_info["valid"], doctor_info)
    
    # If not in our database, check if it follows expected patterns
    # Most Indian medical council registration numbers follow specific patterns
    council_prefixes = ["MCI-", "DMC-", "KMC-", "MMC-", "TNMC-", "APMC-", "UPMC-", "WBMC-", "GMC-", "PMC-"]
    
    # Check if the registration number starts with a valid prefix
    valid_prefix = any(reg_number.startswith(prefix) for prefix in council_prefixes)
    
    # Check if the registration number has the expected format (prefix + 6-8 digits)
    valid_format = bool(re.match(r'^[A-Z]{2,5}-\d{5,8}$', reg_number))
    
    if valid_prefix and valid_format:
        return (True, {"name": "Unknown", "valid": True, "verified_source": "Format validation only"})
    
    return (False, {"valid": False, "reason": "Invalid registration number format or unknown council"})

def verify_phone_number(phone_number):
    """
    Verify if a phone number is likely valid
    """
    # Check if it's a valid format for an Indian phone number
    if re.match(r'^[6-9]\d{9}$', phone_number):
        # In a real system, you would verify this against a database or API
        # For demo purposes, we'll consider it valid if it's in our mock database
        for doctor in DOCTOR_REGISTRY.values():
            if doctor["phone"] == phone_number:
                return (True, {"verified": True, "source": "Registry database"})
        
        # If not in database but valid format
        return (True, {"verified": "Format only", "note": "Number format is valid but not verified against registry"})
    
    return (False, {"verified": False, "reason": "Invalid phone number format"})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_image(image_path):
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Apply additional preprocessing to enhance OCR quality
        # Noise removal with median blur
        processed = cv2.medianBlur(threshold, 3)
        
        # Apply adaptive thresholding to improve text detection
        adaptive_threshold = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Use both regular and adaptive threshold results for better OCR
        text_regular = pytesseract.image_to_string(threshold)
        text_adaptive = pytesseract.image_to_string(adaptive_threshold)
        
        # Use the longer text as it likely contains more information
        text = text_regular if len(text_regular) > len(text_adaptive) else text_adaptive
        
        # Attempt to improve registration number detection
        # Use a custom OCR configuration optimized for alphanumeric codes
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:'
        reg_text = pytesseract.image_to_string(processed, config=config)
        
        # Combine the results, potentially adding more information
        combined_text = text + "\n" + reg_text
        
        return combined_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def analyze_prescription(text, image_path=None):
    """
    Analyze the prescription text to determine if it's fake or original
    and extract relevant information using both basic and advanced analysis.
    
    Args:
        text: OCR extracted text from prescription
        image_path: Optional path to the prescription image for forensic analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Convert text to lowercase for easier matching (keep original for some patterns)
    text_lower = text.lower()
    
    # Initialize result dictionary
    result = {
        'is_original': True,  # Default to original
        'confidence': 0,
        'flags': [],
        'details': {
            'hospital_name': '',
            'doctor_name': '',
            'patient_name': '',
            'date': '',
            'registration_number': '',
            'registration_verified': False,
            'phone_number': '',
            'phone_verified': False,
            'medical_council': '',
            'medicines': [],
            'warnings': []
        }
    }
    
    # If image path is provided, perform comprehensive analysis
    if image_path and os.path.exists(image_path):
        # Get the absolute path of the image
        abs_image_path = os.path.abspath(image_path)
        
        try:
            # Perform comprehensive analysis
            comprehensive_result = comprehensive_prescription_analysis(abs_image_path, text)
            
            # Extract highlights for display
            forensic_highlights = extract_forensic_highlights(comprehensive_result)
            
            # Add comprehensive analysis results to the result
            result['comprehensive_analysis'] = comprehensive_result
            result['forensic_highlights'] = forensic_highlights
            
            # Use comprehensive analysis verdict
            result['is_original'] = comprehensive_result['overall_verdict']['is_genuine']
            result['confidence'] = comprehensive_result['overall_verdict']['confidence']
            
            # Add suspicious flags
            result['flags'] = comprehensive_result['overall_verdict']['suspicious_flags']
            
            # Extract medicine info from comprehensive analysis
            if 'medicines' in comprehensive_result and comprehensive_result['medicines']:
                result['details']['medicines'] = comprehensive_result['medicines']
            
            # Extract doctor details
            if 'doctor_details' in comprehensive_result:
                doctor_details = comprehensive_result['doctor_details']
                if doctor_details.get('name'):
                    result['details']['doctor_name'] = doctor_details['name']
                if doctor_details.get('registration_number'):
                    result['details']['registration_number'] = doctor_details['registration_number']
                    
                    # Verify the registration number with our local database as well
                    is_valid, reg_details = verify_registration_number(doctor_details['registration_number'])
                    result['details']['registration_verified'] = is_valid
                
                if doctor_details.get('hospital'):
                    result['details']['hospital_name'] = doctor_details['hospital']
            
            # Extract patient details
            if 'patient_details' in comprehensive_result:
                patient_details = comprehensive_result['patient_details']
                if patient_details.get('name'):
                    result['details']['patient_name'] = patient_details['name']
                if patient_details.get('date'):
                    result['details']['date'] = patient_details['date']
            
            # If we have medication analysis, extract warnings
            if 'medication_analysis' in comprehensive_result:
                med_analysis = comprehensive_result['medication_analysis']
                warnings = []
                
                # Extract drug interaction warnings
                if 'drug_interactions' in med_analysis and med_analysis['drug_interactions'].get('interactions_found', False):
                    for interaction in med_analysis['drug_interactions']['interactions']:
                        warnings.append(f"Interaction ({interaction['severity']} risk): {interaction['description']}")
                
                # Add generic warnings
                if any('drowsy' in flag.lower() for flag in result['flags']):
                    warnings.append("May cause drowsiness")
                
                if any('alcohol' in flag.lower() for flag in result['flags']):
                    warnings.append("Avoid alcohol consumption")
                
                result['details']['warnings'] = warnings
                
                # Check for controlled substances
                if 'drug_interactions' in med_analysis and med_analysis['drug_interactions'].get('controlled_substances'):
                    for substance in med_analysis['drug_interactions']['controlled_substances']:
                        result['details'].setdefault('controlled_substances', []).append(
                            f"{substance['name'].title()} - {substance['info'].get('schedule', 'Unknown Schedule')}"
                        )
            
            # Return the combined result
            return result
        
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            # If comprehensive analysis fails, fall back to basic analysis
            pass
    
    # ---- Basic Analysis (Fallback) ----
    
    # Extract hospital name (usually at the top)
    hospital_pattern = re.search(r'([A-Za-z\s]+hospital|medical center|clinic|healthcare)', text_lower)
    if hospital_pattern:
        result['details']['hospital_name'] = hospital_pattern.group(0).title()
    
    # Extract doctor name (Dr. followed by name)
    doctor_pattern = re.search(r'dr\.?\s+([A-Za-z\s]+)', text_lower)
    if doctor_pattern:
        result['details']['doctor_name'] = "Dr. " + doctor_pattern.group(1).title().strip()
    
    # Extract patient name (usually preceded by "Patient:" or "Name:")
    patient_pattern = re.search(r'(patient|name)[:\s]+([A-Za-z\s]+)', text_lower)
    if patient_pattern:
        result['details']['patient_name'] = patient_pattern.group(2).title().strip()
    
    # Extract date (common date formats)
    date_pattern = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4}', text)
    if date_pattern:
        result['details']['date'] = date_pattern.group(0)
    
    # Extract doctor registration number
    # Look for common formats: MCI-123456, DMC/123456, Reg. No: 123456, etc.
    reg_patterns = [
        # Format: MCI-123456, DMC-789012
        r'([A-Z]{2,5}[-/]\d{5,8})',
        # Format: Registration No: 123456
        r'(?:registration|reg\.?|medical council).{0,5}(?:no\.?|number).{0,3}[:#]?\s*([A-Z0-9]{5,12})',
        # Format: 123456 (MCI), where MCI is the council
        r'(\d{5,8})\s*\([A-Z]{2,5}\)',
        # Format: Reg#123456
        r'reg#\s*([A-Z0-9]{5,12})'
    ]
    
    for pattern in reg_patterns:
        reg_match = re.search(pattern, text, re.IGNORECASE)
        if reg_match:
            # Extract and clean up the registration number
            reg_number = reg_match.group(1).strip().upper()
            
            # For registration numbers without prefix, try to find the medical council
            if re.match(r'^\d+$', reg_number):
                for council in MEDICAL_COUNCILS:
                    council_abbr = ''.join(word[0] for word in council.split() if '(' not in word)
                    if council_abbr in text or council.lower() in text_lower:
                        reg_number = f"{council_abbr}-{reg_number}"
                        result['details']['medical_council'] = council
                        break
            
            result['details']['registration_number'] = reg_number
            
            # Verify the registration number
            is_valid, reg_details = verify_registration_number(reg_number)
            result['details']['registration_verified'] = is_valid
            
            if not is_valid:
                result['flags'].append("Invalid or unverified doctor registration number")
                result['is_original'] = False
            
            break
    
    # Extract phone number
    phone_patterns = [
        r'(?:phone|tel|mobile|contact|call)(?::|.?no\.?|.?number)?\s*([6-9]\d{9})',
        r'([6-9]\d{9})',  # Standard 10-digit Indian mobile number
        r'\+91[- ]?(\d{10})'  # Format: +91 XXXXXXXXXX
    ]
    
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            phone_number = phone_match.group(1).replace(" ", "").replace("-", "")
            result['details']['phone_number'] = phone_number
            
            # Verify phone number
            is_valid, phone_details = verify_phone_number(phone_number)
            result['details']['phone_verified'] = is_valid
            
            if not is_valid:
                result['flags'].append("Invalid or suspicious phone number")
            
            break
    
    # Extract medicines (usually preceded by Rx or listed with numbers/bullets)
    medicine_lines = re.findall(r'(rx|tab\.?|cap\.?|inj\.?|syr\.?)[:\s]+([^\n]+)', text_lower)
    if not medicine_lines:
        # Try to find numbered list of medicines
        medicine_lines = re.findall(r'[\d\.\)]+\s+([A-Za-z0-9\s]+\d+\s*mg|\w+\s+\d+\s*mg)', text_lower)
    
    # Process medicines
    for med in medicine_lines:
        if isinstance(med, tuple):
            med = med[1]  # Get the medicine name from the tuple
        result['details']['medicines'].append(med.strip().title())
    
    # Look for common warnings
    warnings = []
    if any(word in text_lower for word in ['drowsy', 'drowsiness', 'sleepy']):
        warnings.append("May cause drowsiness")
    
    if any(word in text_lower for word in ['drive', 'driving', 'machinery', 'operate']):
        warnings.append("Avoid driving or operating heavy machinery")
    
    if any(word in text_lower for word in ['alcohol']):
        warnings.append("Avoid alcohol consumption")
        
    result['details']['warnings'] = warnings
    
    # Check for signs of fake prescription
    # 1. Missing essential elements
    if not result['details']['doctor_name']:
        result['flags'].append("Missing doctor name")
        result['is_original'] = False
    
    if not result['details']['patient_name']:
        result['flags'].append("Missing patient name")
        result['is_original'] = False
    
    if not result['details']['date']:
        result['flags'].append("Missing date")
        result['is_original'] = False
    
    if not result['details']['medicines']:
        result['flags'].append("No medicines found")
        result['is_original'] = False
    
    if not result['details']['registration_number']:
        result['flags'].append("Missing doctor registration number")
        result['is_original'] = False
    
    # 2. Check for suspicious terms
    suspicious_terms = ['morphine', 'oxycodone', 'vicodin', 'xanax', 'valium', 'adderall', 'ritalin']
    for term in suspicious_terms:
        if term in text_lower:
            # Check if it's a common prescription by checking for dosage information
            if not re.search(fr'{term}\s+\d+\s*mg', text_lower):
                result['flags'].append(f"Suspicious prescription of {term} without proper dosage")
                result['is_original'] = False
    
    # 3. Check consistency between doctor name and registration
    if result['details']['registration_number'] in DOCTOR_REGISTRY:
        registry_info = DOCTOR_REGISTRY[result['details']['registration_number']]
        doctor_name_lower = result['details']['doctor_name'].lower() if result['details']['doctor_name'] else ''
        registry_name_lower = registry_info['name'].lower()
        
        # Check if doctor names approximately match (allowing for minor OCR errors)
        name_similarity = False
        if doctor_name_lower and registry_name_lower:
            # Basic name similarity check
            doctor_tokens = set(doctor_name_lower.split())
            registry_tokens = set(registry_name_lower.split())
            common_tokens = doctor_tokens.intersection(registry_tokens)
            if len(common_tokens) >= 2 or (len(common_tokens) == 1 and len(doctor_tokens) == 1):
                name_similarity = True
        
        if not name_similarity and result['details']['doctor_name']:
            result['flags'].append("Doctor name doesn't match registration database")
            result['is_original'] = False
    
    # Calculate confidence
    # Base confidence on presence of key elements and validity of registration
    if result['is_original']:
        # Count present elements (max 6)
        present_elements = sum([
            bool(result['details']['hospital_name']),
            bool(result['details']['doctor_name']),
            bool(result['details']['patient_name']),
            bool(result['details']['date']),
            bool(result['details']['registration_number']) and result['details']['registration_verified'],
            bool(result['details']['medicines']),
            bool(result['details']['phone_number']) and result['details']['phone_verified']
        ])
        
        # Max elements is 7, so multiply by 100/7
        result['confidence'] = min(100, int(present_elements * (100/7)))
    else:
        # Reduce confidence based on number of flags
        result['confidence'] = max(0, 100 - len(result['flags']) * 20)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'prescription' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['prescription']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text from the uploaded prescription
            extracted_text = extract_text_from_image(filepath)
            
            # Analyze the prescription with advanced forensics
            analysis_result = analyze_prescription(extracted_text, filepath)
            
            # Make results serializable before storing in session
            serializable_result = make_serializable(analysis_result)
            
            # Store results in session
            session['extracted_text'] = extracted_text
            session['analysis_result'] = serializable_result
            session['image_path'] = filename
            
            return redirect(url_for('result'))
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, PDF)')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/result')
def result():
    if 'analysis_result' not in session:
        return redirect(url_for('index'))
        
    analysis_result = session['analysis_result']
    image_path = session.get('image_path', '')
    
    return render_template(
        'result.html',
        analysis=analysis_result,
        is_original=analysis_result['is_original'],
        confidence=analysis_result['confidence'],
        flags=analysis_result['flags'],
        image_path=image_path
    )

@app.route('/details')
def details():
    if 'analysis_result' not in session or 'extracted_text' not in session:
        return redirect(url_for('index'))
        
    analysis_result = session['analysis_result']
    extracted_text = session['extracted_text']
    image_path = session.get('image_path', '')
    
    # For debugging
    print("Analysis result structure:", analysis_result.keys())
    
    return render_template(
        'details.html',
        analysis=analysis_result,
        extracted_text=extracted_text,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
