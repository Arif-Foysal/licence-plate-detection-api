# Deployment Guide

## Why Tesseract is a Problem for Vercel

Tesseract OCR requires system-level installation (`apt-get install tesseract-ocr` on Linux), which is **not possible on Vercel** because:

1. Vercel is a serverless platform with no persistent filesystem
2. You cannot install system packages with `apt-get`, `brew`, etc.
3. Vercel functions are stateless and containerized

## Solution: EasyOCR (No System Dependencies)

I've updated your app to use **EasyOCR** instead of Tesseract:

### ✅ Benefits of EasyOCR:
- Pure Python package (no system dependencies)
- Works on Vercel, Railway, Render, AWS Lambda, etc.
- Often more accurate than Tesseract for license plates
- GPU support (when available)

### 📦 Updated Requirements:
```
easyocr  # Replaces pytesseract
```

## Deployment Options

### 1. 🟢 **Recommended: Railway or Render** (Full App with YOLO)

**Railway**: https://railway.app/
**Render**: https://render.com/

**Steps:**
1. Push your code to GitHub
2. Connect Railway/Render to your repo
3. Use the provided `Dockerfile`
4. Deploy automatically

**Pros:**
- Supports large model files
- More memory and CPU
- Persistent storage
- Better for computer vision

### 2. 🟡 **Vercel** (Lightweight OCR Only)

Use `app_lightweight.py` - OCR only without YOLO detection.

**Limitations:**
- No YOLO detection (you need to send pre-cropped license plates)
- 50MB file size limit (can't include YOLO model)
- 10-second timeout limit
- Limited memory

**Steps:**
1. Rename `app_lightweight.py` to `app.py`
2. Update `vercel.json` 
3. Deploy to Vercel

### 3. 🔵 **AWS Lambda / Google Cloud Functions**

Similar to Vercel but with configurable limits.

### 4. 🟣 **Docker on VPS** (Full Control)

Use the provided `Dockerfile` on any VPS provider.

## Local Testing

Test the EasyOCR version locally:

```bash
# Install EasyOCR
pip install easyocr

# Start server
uvicorn app:app --reload

# Test
python test_api.py
```

## File Structure for Deployment

```
├── app.py                 # Full version (Railway/Render)
├── app_lightweight.py     # Lightweight version (Vercel)
├── requirements.txt       # Updated with easyocr
├── Dockerfile            # For Railway/Render/VPS
├── vercel.json           # For Vercel deployment
├── runtime.txt           # Python version for Vercel
└── README.md             # Documentation
```

## Performance Comparison

| Platform | YOLO Detection | OCR | Model Size Limit | Memory | Timeout |
|----------|----------------|-----|------------------|---------|---------|
| **Railway** | ✅ | ✅ | No limit | 8GB | No limit |
| **Render** | ✅ | ✅ | No limit | 4GB | No limit |
| **Vercel** | ❌ | ✅ | 50MB | 1GB | 10s |
| **AWS Lambda** | ⚠️ | ✅ | 250MB | 3GB | 15min |

## Recommendation

For your license plate detection use case:

1. **Best Choice**: Deploy on **Railway** or **Render** with the full app
2. **Alternative**: Use **Vercel** with the lightweight OCR-only version
3. **Hybrid**: Use Vercel for OCR + separate service for YOLO detection
