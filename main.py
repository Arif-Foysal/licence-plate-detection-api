from ultralytics import YOLO
import cv2
import os
import uuid
import pytesseract
from PIL import Image
import numpy as np

# Set Tesseract path (for macOS with Homebrew installation)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Load the model
model = YOLO('license-plate-finetune-v1m.pt')

# Input image filename
input_image = 'Cars1.png'

# Predict on the image (disable automatic saving to avoid separate folders)
results = model.predict(source=input_image, save=False)

# Get the first result (since we're processing one image)
result = results[0]

# Load the original image for cropping
original_image = cv2.imread(input_image)

# Plot the results on the image
annotated_image = result.plot()

# Generate UUID for unique filenames
unique_id = str(uuid.uuid4())

# Get base filename without extension
base_name = os.path.splitext(input_image)[0]

# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the annotated image with new naming convention
output_filename = f'{output_dir}/{base_name}_detected_{unique_id}.png'
cv2.imwrite(output_filename, annotated_image)

# Extract and save cropped license plates
if result.boxes is not None:
    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
    
    # Crop and save each detected license plate
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        
        # Crop the license plate region
        cropped_plate = original_image[y1:y2, x1:x2]
        
        # Preprocess the cropped image for better OCR results
        # Convert to grayscale
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        
        # Apply some preprocessing to improve OCR accuracy
        # Resize image for better OCR (make it larger)
        height, width = gray_plate.shape
        scale_factor = 3
        resized_plate = cv2.resize(gray_plate, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold to get better contrast
        _, thresh_plate = cv2.threshold(resized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(thresh_plate)
        
        # Extract text using OCR with specific configuration for license plates
        # Using whitelist of characters commonly found in license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        extracted_text = pytesseract.image_to_string(pil_image, config=custom_config).strip()
        
        # Clean up the extracted text (remove spaces and special characters)
        cleaned_text = ''.join(extracted_text.split()).upper()
        
        # Save the cropped license plate with new naming convention
        crop_filename = f'{output_dir}/{base_name}_cropped_{unique_id}.png'
        cv2.imwrite(crop_filename, cropped_plate)
        
        # Also save the preprocessed image for debugging
        debug_filename = f'{output_dir}/{base_name}_preprocessed_{unique_id}.png'
        cv2.imwrite(debug_filename, thresh_plate)
        
        print(f"Cropped license plate {i+1} saved as: {crop_filename}")
        print(f"Preprocessed image saved as: {debug_filename}")
        print(f"Extracted text: '{cleaned_text}'")
        print(f"Raw OCR output: '{extracted_text}'")
        print(f"Confidence: {conf:.2f}")
        print("-" * 50)
else:
    print("No license plates detected in the image.")

print(f"Annotated image saved as: {output_filename}")
print(f"All files saved in output directory with unique UUIDs")
