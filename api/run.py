#!/usr/bin/env python3
"""
run.py - Script to run the Document Processing API

This script starts the FastAPI server for the Document Processing API.
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Get port from environment variable or use default
PORT = int(os.getenv("API_PORT", "8000"))

if __name__ == "__main__":
    print(f"Starting Document Processing API on port {PORT}...")
    print(f"API documentation will be available at http://localhost:{PORT}/docs")
    
    # Change to the api directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the app
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
