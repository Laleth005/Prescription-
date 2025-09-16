# Prescription Analyzer

A comprehensive web application that analyzes prescription images to verify their authenticity and extract key information using advanced forensic techniques.

## Features

- Upload prescription images for analysis
- Extract text from prescription images using OCR
- Verify prescription authenticity with multiple verification methods:
  - Image forensic analysis to detect manipulation
  - Format analysis to verify prescription layout
  - Drug interaction and controlled substance detection
  - Doctor registration verification
  - Signature verification
- View detailed prescription information including:
  - Hospital/clinic name
  - Doctor's name and credentials
  - Patient's name and information
  - Prescribed medicines with drug interaction warnings
  - Controlled substance alerts
  - Format compliance assessment
  - Image manipulation detection results

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd prescription-analyzer
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - For Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - For macOS: `brew install tesseract`
   - For Ubuntu: `sudo apt install tesseract-ocr`

4. Update the Tesseract path in `app.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Click "Upload My Prescription" to analyze a prescription image

## System Requirements

- Python 3.8+
- Tesseract OCR 4.0+
- OpenCV 4.5+
- Modern web browser (Chrome, Firefox, Edge, Safari)

## Advanced Analysis Modules

The system includes several specialized analysis modules:

1. **Image Forensics** (`utils/image_forensics.py`)
   - Metadata analysis for detecting edited images
   - Error Level Analysis (ELA) to identify manipulated regions
   - Font consistency checking to detect text alterations
   - Signature region detection and analysis

2. **Drug Analysis** (`utils/drug_analysis.py`)
   - Medication identification and normalization
   - Drug interaction detection between prescribed medicines
   - Controlled substance identification and verification
   - Dosage analysis for suspicious patterns

3. **Format Analysis** (`utils/format_analysis.py`) 
   - Prescription template verification
   - Layout analysis of key prescription regions
   - Standard medical prescription format validation
   - Consistency checks with known hospital templates

4. **Signature Verification** (`utils/signature_verification.py`)
   - Automatic signature region detection
   - Feature extraction and comparison
   - Signature validity assessment
   - Consistency analysis with known signatures

5. **Comprehensive Analysis** (`utils/comprehensive_analysis.py`)
   - Integration of all specialized analysis modules
   - Weighted scoring system for overall authenticity assessment
   - Detailed reporting with forensic highlights

## Technology Stack

- Backend: Python, Flask
- Text Extraction: Tesseract OCR, OpenCV
- Image Forensics: OpenCV, Error Level Analysis, Metadata extraction
- Drug Analysis: Medication pattern matching, interaction detection
- Format Analysis: Template verification, pattern recognition
- Signature Verification: Feature extraction, contour analysis
- Frontend: HTML, CSS, JavaScript, Responsive design

## License

[MIT License](LICENSE)
