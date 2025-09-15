import os
import uuid
import cv2
import numpy as np
import pytesseract
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename

# Set the path to tesseract executable (change this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

app = Flask(__name__)
app.secret_key = 'prescription_analysis_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(threshold)
        
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def analyze_prescription(text):
    """
    Analyze the prescription text to determine if it's fake or original
    and extract relevant information.
    """
    # Convert text to lowercase for easier matching
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
            'medicines': [],
            'warnings': []
        }
    }
    
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
    
    # 2. Check for suspicious terms
    suspicious_terms = ['morphine', 'oxycodone', 'vicodin', 'xanax', 'valium', 'adderall', 'ritalin']
    for term in suspicious_terms:
        if term in text_lower:
            # Check if it's a common prescription by checking for dosage information
            if not re.search(fr'{term}\s+\d+\s*mg', text_lower):
                result['flags'].append(f"Suspicious prescription of {term} without proper dosage")
                result['is_original'] = False
    
    # Calculate confidence
    if result['is_original']:
        # Count present elements
        present_elements = sum([
            bool(result['details']['hospital_name']),
            bool(result['details']['doctor_name']),
            bool(result['details']['patient_name']),
            bool(result['details']['date']),
            bool(result['details']['medicines'])
        ])
        result['confidence'] = min(100, present_elements * 20)
    else:
        # Inverse of the number of flags
        result['confidence'] = max(0, 100 - len(result['flags']) * 25)
    
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
            
            # Analyze the prescription
            analysis_result = analyze_prescription(extracted_text)
            
            # Store results in session
            session['extracted_text'] = extracted_text
            session['analysis_result'] = analysis_result
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
    
    return render_template(
        'details.html',
        analysis=analysis_result,
        extracted_text=extracted_text,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
