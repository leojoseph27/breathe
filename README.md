# Respiratory Audio Disease Prediction System

This is a web application that uses a trained CNN model to predict respiratory diseases from audio samples. The system has a Flask backend that processes audio files and makes predictions, and a clean, user-friendly frontend for uploading and playing audio files.

## Features

- Upload respiratory audio files (WAV, MP3, M4A, FLAC)
- Play uploaded audio files directly in the browser
- Send audio files to backend for disease prediction
- Display prediction results with confidence scores
- Responsive design that works on desktop and mobile devices

## Components

### Backend (Flask)
- `/` - Main page serving the HTML frontend
- `/predict` - POST endpoint to receive audio files and return predictions

### Frontend
- HTML page with audio upload and playback functionality
- CSS styling for a clean, professional appearance
- JavaScript to handle file uploads and communicate with the backend

## Setup Instructions

### Option 1: Using Virtual Environment (Recommended)

**On Windows:**
1. Run the setup script:
   ```
   setup_venv.bat
   ```
   
**On Linux/Mac:**
1. Make the script executable and run it:
   ```
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

### Option 2: Manual Setup

1. Create and activate a virtual environment:
   ```
   python -m venv respiratory_ai_env
   # On Windows:
   respiratory_ai_env\Scripts\activate
   # On Linux/Mac:
   source respiratory_ai_env/bin/activate
   ```

2. Install the required dependencies:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Ensure you have the model file `respiratory_audio_cnn.h5` in the project root directory.

4. Run the Flask application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000` (or the IP address shown in the terminal).

## Technical Details

The backend processes audio files using the same feature extraction techniques used during training:
- Zero Crossing Rate (ZCR)
- Chroma STFT
- MFCC (Mel-Frequency Cepstral Coefficients)
- Root Mean Square (RMS)
- Mel Spectrogram

The model takes 2.5-second audio clips with a 0.6-second offset and predicts among 5 classes:
- Bronchial
- Asthma
- COPD
- Healthy
- Pneumonia

## Troubleshooting

If you encounter issues with the TensorFlow model loading due to version incompatibilities, the system will fall back to a mock prediction function that returns random results from the valid classes. To resolve this permanently:
1. Retrain and save the model with compatible versions, or
2. Upgrade/downgrade your TensorFlow installation to match the model's version