# License Plate Detection API

A FastAPI backend service for detecting license plates in images and extracting text using OCR.

## Features

- 🚗 **License Plate Detection**: Uses YOLO model to detect license plates in images
- 📝 **OCR Text Extraction**: Extracts text from detected license plates using Tesseract OCR
- 🖼️ **Image Processing**: Saves detected images, cropped plates, and preprocessed images
- 🔍 **High Accuracy**: Includes confidence scores and preprocessing for better OCR results
- 📊 **JSON API**: RESTful API with detailed JSON responses

## API Endpoints

### `GET /`
- Returns API information and available endpoints

### `GET /health`
- Health check endpoint

### `POST /detect`
- Upload an image file to detect license plates
- **Input**: Image file (PNG, JPG, JPEG)
- **Output**: JSON response with detected plates and extracted text

### `GET /download/{filename}`
- Download processed image files

### `GET /docs`
- Interactive API documentation (Swagger UI)

### `GET /redoc`
- Alternative API documentation

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR:**
   ```bash
   # macOS with Homebrew
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

## Usage

### Start the Server
```bash
# Activate virtual environment
source .venv/bin/activate

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### Test the API

1. **Using the test script:**
   ```bash
   python test_api.py
   ```

2. **Using curl:**
   ```bash
   # Test API health
   curl http://localhost:8000/health
   
   # Upload image for detection
   curl -X POST -F "file=@Cars1.png" http://localhost:8000/detect
   ```

3. **Using Python requests:**
   ```python
   import requests
   
   # Upload image
   with open('Cars1.png', 'rb') as f:
       files = {'file': ('Cars1.png', f, 'image/png')}
       response = requests.post('http://localhost:8000/detect', files=files)
   
   result = response.json()
   print(f"Detected {result['total_plates_detected']} license plates")
   ```

### API Response Format

```json
{
  "success": true,
  "message": "Successfully detected 1 license plate(s)",
  "total_plates_detected": 1,
  "detected_image_url": "/static/Cars1_detected_uuid.png",
  "license_plates": [
    {
      "plate_number": 1,
      "extracted_text": "PPGMN112",
      "raw_ocr_output": "PPGMN112",
      "confidence": 0.77,
      "cropped_image_url": "/static/Cars1_cropped_1_uuid.png",
      "preprocessed_image_url": "/static/Cars1_preprocessed_1_uuid.png"
    }
  ],
  "unique_id": "uuid-string"
}
```

## File Structure

```
licence-plate-detection/
├── app.py                 # FastAPI application
├── main.py               # Original standalone script
├── test_api.py           # API test script
├── start_server.sh       # Server startup script
├── requirements.txt      # Python dependencies
├── license-plate-finetune-v1m.pt  # YOLO model
├── output/               # Generated images and results
├── Cars*.png            # Sample images
└── README.md            # This file
```

## Model Information

- **YOLO Model**: `license-plate-finetune-v1m.pt`
- **OCR Engine**: Tesseract 5.x
- **Supported Formats**: PNG, JPG, JPEG
- **Output**: Detected images with bounding boxes, cropped license plates, preprocessed images

## Development

### Interactive Documentation
Visit `http://localhost:8000/docs` for Swagger UI documentation where you can test the API endpoints directly.

### Environment Variables
You can customize the Tesseract path by modifying the `tesseract_cmd` in `app.py`:
```python
pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'
```

## Error Handling

The API includes comprehensive error handling:
- File validation (image types only)
- Model loading errors
- OCR processing errors
- File system errors

All errors return appropriate HTTP status codes and descriptive error messages.
