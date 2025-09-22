#!/bin/bash
# ======================================================
# Emotion Recognition Setup and Run Script
# ======================================================

# 1. Create virtual environment if it doesn't exist
if [ ! -d "emotion-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv emotion-env
fi

# 2. Activate the environment
source emotion-env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install required packages
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Please add it to the project folder."
    exit 1
fi

# 5. Run real-time emotion recognition
echo "Starting real-time emotion recognition..."
python real_time_video_barchart.py

# 6. Deactivate environment after exit
deactivate
echo "Done!"
