from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import uuid
import gc
from PIL import Image
import numpy as np
from typing import List, Optional
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="License Plate Detection API",
    description="Backend API for detecting and extracting text from license plates",
    version="1.0.0"
)

# Global variables for lazy loading
model = None
reader = None

def load_model():
    """Lazy load YOLO model"""
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO('license-plate-finetune-v1m.pt')
    return model

def load_ocr():
    """Lazy load OCR reader"""
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mount static files
app.mount("/static", StaticFiles(directory=output_dir), name="static")

# Response models (same as before)
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
    return {
        "message": "License Plate Detection API",
        "version": "1.0.0",
        "memory_optimized": True,
        "endpoints": {
            "detect": "/detect - POST endpoint to upload image and detect license plates",
            "health": "/health - GET endpoint to check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "License Plate Detection API is running"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_license_plates(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Generate unique ID
        unique_id = str(uuid.uuid4())
        
        # Load models only when needed
        model = load_model()
        reader = load_ocr()
        
        # Read uploaded file
        contents = await file.read()
        temp_filename = f"temp_{unique_id}_{file.filename}"
        temp_path = os.path.join(output_dir, temp_filename)
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        base_name = os.path.splitext(file.filename)[0]
        
        # Predict with optimized settings
        results = model.predict(source=temp_path, save=False, verbose=False)
        result = results[0]
        
        # Load original image
        original_image = cv2.imread(temp_path)
        
        # Create annotated image
        annotated_image = result.plot()
        
        # Save detected image
        detected_filename = f'{base_name}_detected_{unique_id}.png'
        detected_path = os.path.join(output_dir, detected_filename)
        cv2.imwrite(detected_path, annotated_image)
        
        license_plates = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                cropped_plate = original_image[y1:y2, x1:x2]
                
                # Simplified preprocessing
                gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                resized_plate = cv2.resize(gray_plate, (gray_plate.shape[1] * 2, gray_plate.shape[0] * 2))
                
                # OCR with memory cleanup
                ocr_results = reader.readtext(resized_plate, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                extracted_text = ' '.join(ocr_results) if ocr_results else ''
                cleaned_text = ''.join(extracted_text.split()).upper()
                
                # Save cropped image
                crop_filename = f'{base_name}_cropped_{i+1}_{unique_id}.png'
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_plate)
                
                # Save preprocessed image
                debug_filename = f'{base_name}_preprocessed_{i+1}_{unique_id}.png'
                debug_path = os.path.join(output_dir, debug_filename)
                cv2.imwrite(debug_path, resized_plate)
                
                plate_result = LicensePlateResult(
                    plate_number=i+1,
                    extracted_text=cleaned_text,
                    raw_ocr_output=extracted_text,
                    confidence=float(conf),
                    cropped_image_url=f"/static/{crop_filename}",
                    preprocessed_image_url=f"/static/{debug_filename}"
                )
                license_plates.append(plate_result)
        
        # Clean up
        os.remove(temp_path)
        del original_image, annotated_image
        gc.collect()
        
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
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='image/png')