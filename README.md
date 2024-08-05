# OCR Image and PDF Processing with Flask

This repository contains a Flask web application that processes images and PDF files to extract text using three OCR engines: PyTesseract, EasyOCR, and PaddleOCR.

## Features

- Extracts text from PDF documents and images (PNG, JPG, JPEG, TIFF, BMP)
- Supports text detection on various background colors
- Advanced image preprocessing for improved OCR accuracy:
  - Grayscale conversion
  - Adaptive thresholding
  - Denoising
  - Dilation and erosion for noise removal
- Utilizes multiple OCR engines:
  - Tesseract (via pytesseract)
  - EasyOCR
  - PaddleOCR
- Supports GPU acceleration for EasyOCR and PaddleOCR
- Display the extracted text from each OCR engine.
- View processed images and extracted text results in a web interface.

## Requirements

- Python 3.7 or later
- Flask
- PyTesseract
- EasyOCR
- PaddleOCR
- pdf2image
- OpenCV
- NumPy
- PIL (Pillow)
- Werkzeug

## Python Version

This project was developed using Python 3.12.3

## Installation

1. Clone the repository:

```bash
git clone https://github.com/barisacdr/OCR_flask_app.git
cd OCR_flask_app
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

or

```bash
pip install Flask PyTesseract EasyOCR PaddleOCR pdf2image OpenCV-Python-headless numpy Pillow Werkzeug
```

4. Download and install Tesseract-OCR from [here](https://github.com/tesseract-ocr/tesseract). Ensure that Tesseract is correctly installed and update the path to the `tesseract.exe` file in `app.py`:

```python
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
```

## Running the Application

1. Start the Flask application:

```bash
python app.py
```

2. Open a web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage

1. On the home page, upload an image file or a PDF file.
2. Click the "Process File" button.
3. View the processed images and extracted text from each OCR engine on the results page.

## Supported File Formats

The script supports the following file formats:
- PDF (.pdf)
- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tiff)
- BMP (.bmp)

## Image Preprocessing

The script now includes advanced image preprocessing steps to improve OCR accuracy:

1. Grayscale conversion
2. Adaptive thresholding
3. Denoising using fastNlMeansDenoising
4. Dilation and erosion to remove noise

These steps help in detecting text on various background colors and improve overall text recognition.

## Project Structure

- `app.py`: Main Flask application file.
- `templates/`: Contains HTML templates for the web pages.
  - `index.html`: Home page for file upload.
  - `results.html`: Results page to display processed images and extracted text.
- `static/uploads/`: Directory for storing uploaded images.

## Example

1. Upload an image or PDF file on the home page.
2. After processing, view the results page showing the processed images and extracted text from PyTesseract, EasyOCR, and PaddleOCR.

## Screenshots

### Home Page

![Home Page](screenshots/home_page.PNG)

### Results Page

![Results Page](screenshots/results_page.PNG)