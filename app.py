from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np
import torch
import paddle
from PIL import Image
import tempfile
import uuid
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Create a folder to store uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize OCR engines and settings
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reader_easyocr = easyocr.Reader(['en'], gpu=True if device == 'cuda' else False)
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
reader_paddleocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=paddle.is_compiled_with_cuda())

def clean_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def process_image(image, use_tesseract=True, use_easyocr=True, use_paddleocr=True):
    image_np = np.array(image)
    processed_image = preprocess_image(image_np)
    
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_data = {
        'PyTesseract': [],
        'EasyOCR': [],
        'PaddleOCR': []
    }

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = max(0, x-5), max(0, y-5), min(w+10, image_np.shape[1]-x), min(h+10, image_np.shape[0]-y)
        table_image = image_np[y:y+h, x:x+w]

        if use_tesseract:
            try:
                tesseract_text = pytesseract.image_to_string(table_image)
                extracted_data['PyTesseract'].append(clean_text(tesseract_text))
            except Exception as e:
                extracted_data['PyTesseract'].append(f"Error: {str(e)}")

        if use_easyocr:
            try:
                easyocr_result = reader_easyocr.readtext(table_image)
                extracted_data['EasyOCR'].append(' '.join([text[1] for text in easyocr_result]))
            except Exception as e:
                extracted_data['EasyOCR'].append(f"Error: {str(e)}")

        if use_paddleocr:
            try:
                paddleocr_result = reader_paddleocr.ocr(table_image, cls=True)
                if paddleocr_result and paddleocr_result[0]:
                    extracted_data['PaddleOCR'].append(' '.join([
                        line[1][0] for line in paddleocr_result[0] if line is not None and isinstance(line, list) and len(line) > 1
                    ]))
                else:
                    extracted_data['PaddleOCR'].append("No text detected")
            except Exception as e:
                extracted_data['PaddleOCR'].append(f"Error: {str(e)}")

    # Join each list into a single string
    for key in extracted_data:
        extracted_data[key] = ' '.join(extracted_data[key])

    return extracted_data

def process_file(file):
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name

        images = convert_from_path(temp_pdf_path)
        os.unlink(temp_pdf_path)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        images = [Image.open(file.stream)]
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    all_extracted_data = {
        'PyTesseract': [],
        'EasyOCR': [],
        'PaddleOCR': []
    }
    image_paths = []

    for i, image in enumerate(images):
        extracted_data = process_image(image)
        for key in all_extracted_data:
            all_extracted_data[key].append(extracted_data[key])  # Append each result list to the respective key
        
        # Save the image
        if file_extension == '.pdf':
            image_filename = f"{uuid.uuid4()}.png"
        else:
            image_filename = f"{uuid.uuid4()}{file_extension}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)
        image_paths.append(image_filename)

    return all_extracted_data, image_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            try:
                extracted_data, image_paths = process_file(file)
                return render_template('results.html', results=extracted_data, image_paths=image_paths)
            except ValueError as e:
                return jsonify({'error': str(e)})
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')