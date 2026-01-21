# Respiratory Audio Disease Prediction System

This is a web application that uses a trained CNN model to predict respiratory diseases from audio samples. The system has a Flask backend that processes audio files and makes predictions, and a clean, user-friendly frontend for uploading and playing audio files.

## Features

- Upload respiratory audio files (WAV, MP3, M4A, FLAC)
- Play uploaded audio files directly in the browser
- Send audio files to backend for disease prediction
- Display prediction results with confidence scores
- Responsive design that works on desktop and mobile devices
- Asthma risk assessment using clinical data
- Environmental health monitoring with Weatherstack API
- AI-powered medical verdict generation using Google Gemini

## Deployment Options

### Local Development

#### Option 1: Using Virtual Environment (Recommended)

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

#### Option 2: Manual Setup

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

4. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   WEATHERSTACK_API_KEY=your_weatherstack_api_key_here
   ```

5. Run the Flask application:
   ```
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5000`

### Deploy to Render

1. Fork this repository to your GitHub account
2. Sign up/in to [Render](https://render.com)
3. Click "New+" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Name**: breathe-app
   - **Region**: Oregon (or your preference)
   - **Branch**: main
   - **Root Directory**: Leave empty
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`

6. Add environment variables in the Render dashboard:
   - `GOOGLE_API_KEY` = your Google Gemini API key
   - `WEATHERSTACK_API_KEY` = your Weatherstack API key
   - `FLASK_DEBUG` = false

7. Click "Create Web Service"

8. Wait for the build and deployment to complete

Your application will be available at `https://your-app-name.onrender.com`

## Components

### Backend (Flask)
- `/` - Main page serving the HTML frontend
- `/predict` - POST endpoint to receive audio files and return predictions
- `/predict_asthma` - POST endpoint for asthma risk assessment
- `/weather` - GET endpoint for weather data
- `/generate_ai_verdict` - POST endpoint for AI medical verdict
- `/login` and `/logout` - Authentication routes

### Frontend
- Login page with session management
- Audio analysis interface
- Asthma detection form
- Environmental health monitoring
- AI doctor consultation page

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

## API Keys Required

1. **Google Gemini API Key**
   - Get from [Google AI Studio](https://aistudio.google.com/)
   - Used for AI doctor medical verdict generation

2. **Weatherstack API Key**
   - Get from [Weatherstack](https://weatherstack.com/)
   - Free tier available for basic weather data
   - Used for environmental health monitoring

## Troubleshooting

If you encounter issues with the TensorFlow model loading due to version incompatibilities, the system will fall back to a mock prediction function that returns random results from the valid classes. To resolve this permanently:
1. Retrain and save the model with compatible versions, or
2. Upgrade/downgrade your TensorFlow installation to match the model's version

For deployment issues:
- Check Render logs for build errors
- Verify all environment variables are set correctly
- Ensure model files are included in the repository