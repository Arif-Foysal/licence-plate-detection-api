#!/bin/bash
# Activate virtual environment and start FastAPI server
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
