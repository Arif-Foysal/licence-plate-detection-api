from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import uuid
import easyocr
from PIL import Image
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
from pydantic import BaseModel

# Initialize EasyOCR reader (English only for better performance)
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA

# Initialize FastAPI app

app = FastAPI(
    title="License Plate Detection API",
    description="Backend API for detecting and extracting text from license plates",
    version="1.0.0"
)

# Load the YOLO model
model = YOLO('license-plate-finetune-v1m.pt')

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mount static files to serve images
app.mount("/static", StaticFiles(directory=output_dir), name="static")

# Response models
class LicensePlateResult(BaseModel):
    plate_number: int
    extracted_text: str
    raw_ocr_output: str
    confidence: float
    cropped_image_url: str
    preprocessed_image_url: str

class DetectionResponse(BaseModel):
    success: bool
    message: str
    total_plates_detected: int
    detected_image_url: str
    license_plates: List[LicensePlateResult]
    unique_id: str

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "License Plate Detection API",
        "version": "1.0.0",
        "description": "Backend API for detecting and extracting text from license plates",
        "endpoints": {
            "detect": "/detect - POST endpoint to upload image and detect license plates",
            "health": "/health - GET endpoint to check API health",
            "download": "/download/{filename} - GET endpoint to download processed images",
            "docs": "/docs - Interactive API documentation",
            "redoc": "/redoc - Alternative API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "License Plate Detection API is running"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_license_plates(file: UploadFile = File(...)):
    """
    Detect license plates in an uploaded image and extract text using OCR
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG)
    
    Returns:
        DetectionResponse with detected license plates and extracted text
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Generate unique ID for this request
        unique_id = str(uuid.uuid4())
        
        # Read and save the uploaded file temporarily
        contents = await file.read()
        temp_filename = f"temp_{unique_id}_{file.filename}"
        temp_path = os.path.join(output_dir, temp_filename)
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Get base filename without extension
        base_name = os.path.splitext(file.filename)[0]
        
        # Predict on the image
        results = model.predict(source=temp_path, save=False)
        result = results[0]
        
        # Load the original image for cropping
        original_image = cv2.imread(temp_path)
        
        # Plot the results on the image
        annotated_image = result.plot()
        
        # Save the annotated image
        detected_filename = f'{base_name}_detected_{unique_id}.png'
        detected_path = os.path.join(output_dir, detected_filename)
        cv2.imwrite(detected_path, annotated_image)
        
        license_plates = []
        
        # Extract and save cropped license plates
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Crop the license plate region
                cropped_plate = original_image[y1:y2, x1:x2]
                
                # Preprocess the cropped image for better OCR results
                gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                
                # Resize image for better OCR
                height, width = gray_plate.shape
                scale_factor = 3
                resized_plate = cv2.resize(gray_plate, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
                
                # Apply threshold
                _, thresh_plate = cv2.threshold(resized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract text using EasyOCR
                # EasyOCR works directly with numpy arrays
                ocr_results = reader.readtext(thresh_plate, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                
                # Combine all detected text
                extracted_text = ' '.join(ocr_results) if ocr_results else ''
                
                # Clean up the extracted text
                cleaned_text = ''.join(extracted_text.split()).upper()
                
                # Save the cropped license plate
                crop_filename = f'{base_name}_cropped_{i+1}_{unique_id}.png'
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_plate)
                
                # Save the preprocessed image
                debug_filename = f'{base_name}_preprocessed_{i+1}_{unique_id}.png'
                debug_path = os.path.join(output_dir, debug_filename)
                cv2.imwrite(debug_path, thresh_plate)
                
                # Create license plate result
                plate_result = LicensePlateResult(
                    plate_number=i+1,
                    extracted_text=cleaned_text,
                    raw_ocr_output=extracted_text,
                    confidence=float(conf),
                    cropped_image_url=f"/static/{crop_filename}",
                    preprocessed_image_url=f"/static/{debug_filename}"
                )
                license_plates.append(plate_result)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Create response
        response = DetectionResponse(
            success=True,
            message=f"Successfully detected {len(license_plates)} license plate(s)",
            total_plates_detected=len(license_plates),
            detected_image_url=f"/static/{detected_filename}",
            license_plates=license_plates,
            unique_id=unique_id
        )
        
        return response
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed image file"""
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/png'
    )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
