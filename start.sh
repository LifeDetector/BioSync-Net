#!/bin/bash
# BioSync-Net Render Startup Script

echo "Starting BioSync-Net Deployment..."

# Navigate to Backend directory
cd Backend

# Start the application using Uvicorn
# We use --host 0.0.0.0 and get the PORT from Render's environment variable
# $PORT is automatically provided by Render
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 30
