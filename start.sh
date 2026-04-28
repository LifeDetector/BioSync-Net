#!/bin/bash
# BioSync-Net Render Startup Script

echo "Starting BioSync-Net Deployment..."

# Navigate to Backend directory
cd Backend

# Start the application using Gunicorn with Uvicorn workers
# We use 1 worker to save RAM on Render Free Tier (512MB limit)
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT:-8000} --timeout 120
