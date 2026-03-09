#!/bin/bash

# Stillion AI - FLUX LoRA Training App
# Start script for DGX Spark

cd "$(dirname "$0")"

echo "================================================"
echo "  Stillion AI - FLUX LoRA Training"
echo "================================================"
echo ""
echo "Starting Gradio app on port 7865..."
echo "Access at: http://localhost:7865"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the app
python code/app.py
