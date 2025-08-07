from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import os
import uuid
import easyocr
from PIL import Image
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
import requests
import io

# Initialize EasyOCR reader (English only for better performance)
reader = easyocr.Reader(['en'], gpu=False)

# Initialize FastAPI app
app = FastAPI(
    title="License Plate OCR API",
    description="Lightweight OCR API for extracting text from license plate images",
    version="1.0.0"
)

# Response models
class OCRResult(BaseModel):
    extracted_text: str
    confidence: float
    processing_time: float

class OCRResponse(BaseModel):
    success: bool
    message: str
    result: Optional[OCRResult]

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "License Plate OCR API",
        "version": "1.0.0",
        "description": "Lightweight OCR API for extracting text from license plate images",
        "endpoints": {
            "ocr": "/ocr - POST endpoint to extract text from uploaded image",
            "health": "/health - GET endpoint to check API health"
        },
        "note": "This is a lightweight version without YOLO detection. Send pre-cropped license plate images."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "License Plate OCR API is running"}

@app.post("/ocr", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from an uploaded license plate image using OCR
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG) - should be a cropped license plate
    
    Returns:
        OCRResponse with extracted text and confidence
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        import time
        start_time = time.time()
        
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess the image for better OCR results
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image for better OCR (make it larger)
        height, width = gray.shape
        scale_factor = 3
        resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold to get better contrast
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text using EasyOCR
        ocr_results = reader.readtext(thresh, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # Get the result with highest confidence
        if ocr_results:
            # EasyOCR returns (bbox, text, confidence)
            best_result = max(ocr_results, key=lambda x: x[2])
            extracted_text = best_result[1].upper().replace(' ', '')
            confidence = float(best_result[2])
        else:
            extracted_text = ""
            confidence = 0.0
        
        processing_time = time.time() - start_time
        
        # Create response
        response = OCRResponse(
            success=True,
            message=f"Successfully extracted text: '{extracted_text}'" if extracted_text else "No text detected",
            result=OCRResult(
                extracted_text=extracted_text,
                confidence=confidence,
                processing_time=round(processing_time, 3)
            )
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
