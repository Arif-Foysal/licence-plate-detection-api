#!/usr/bin/env python3
"""
Test script for License Plate Detection API
"""
import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

def test_api_health():
    """Test the health endpoint"""
    print("🔍 Testing API health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_api_info():
    """Test the root endpoint for API info"""
    print("📋 Getting API information...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_license_plate_detection(image_path):
    """Test license plate detection"""
    print(f"🚗 Testing license plate detection with {image_path}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')}
            response = requests.post(f"{API_URL}/detect", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection successful!")
            print(f"Total plates detected: {result['total_plates_detected']}")
            print(f"Request ID: {result['unique_id']}")
            
            for i, plate in enumerate(result['license_plates'], 1):
                print(f"\n📝 License Plate {i}:")
                print(f"  Extracted Text: '{plate['extracted_text']}'")
                print(f"  Confidence: {plate['confidence']:.2f}")
                print(f"  Cropped Image: {API_URL}{plate['cropped_image_url']}")
                print(f"  Preprocessed Image: {API_URL}{plate['preprocessed_image_url']}")
            
            print(f"\n🖼️ Full detected image: {API_URL}{result['detected_image_url']}")
        else:
            print(f"❌ Error: {response.text}")
    
    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()

if __name__ == "__main__":
    print("🚀 License Plate Detection API Test")
    print("=" * 50)
    
    # Test API health
    test_api_health()
    
    # Test API info
    test_api_info()
    
    # Test detection with available images
    test_images = ["Cars1.png", "Cars2.png", "Cars7.png", "Cars8.png", "Cars15.png"]
    
    for image in test_images:
        try:
            test_license_plate_detection(image)
        except Exception as e:
            print(f"❌ Failed to test {image}: {str(e)}")
    
    print("🏁 Test completed!")
    print(f"📊 API Documentation: {API_URL}/docs")
    print(f"📚 Alternative Docs: {API_URL}/redoc")
