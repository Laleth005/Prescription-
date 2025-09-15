# Prescription Analyzer

A web application that analyzes prescription images to verify their authenticity and extract key information.

## Features

- Upload prescription images for analysis
- Extract text from prescription images using OCR
- Verify prescription authenticity 
- View detailed prescription information including:
  - Hospital/clinic name
  - Doctor's name
  - Patient's name
  - Prescribed medicines
  - Warnings and precautions

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
- Modern web browser (Chrome, Firefox, Edge, Safari)

## Technology Stack

- Backend: Python, Flask
- Text Extraction: Tesseract OCR, OpenCV
- Frontend: HTML, CSS, JavaScript

## License

[MIT License](LICENSE)
