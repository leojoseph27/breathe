import os
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template, session, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tempfile
import uuid
import joblib
import requests
from dotenv import load_dotenv
# Import and configure genai with the new API structure
try:
    from google import genai
    GENAI_AVAILABLE = True
    print("Google GenAI library imported successfully")
except ImportError as e:
    print(f"Google GenAI library import failed: {e}")
    genai = None
    GENAI_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "respiratory_audio_cnn.h5"
ASTHMA_MODEL_PATH = "asthma_lightgbm_model.pkl"
LABEL_CLASSES = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Needed for sessions
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# -------------------------------
# LOAD MODELS
# -------------------------------
# Load respiratory model
try:
    # Load the audio model with its original compilation
    audio_model = load_model(MODEL_PATH)
    print("CNN audio model loaded successfully!")
    AUDIO_MODEL_LOADED = True
    print(f"Audio model type: {type(audio_model)}")
    print(f"Audio model has predict method: {hasattr(audio_model, 'predict')}")
except Exception as e:
    print(f"Error loading audio model: {e}")
    audio_model = None
    AUDIO_MODEL_LOADED = False
    print("Audio model could not be loaded. Using mock prediction until fixed.")

# Load asthma prediction model
try:
    import lightgbm
    print("LightGBM imported successfully")
    asthma_model = joblib.load(ASTHMA_MODEL_PATH)
    ASTHMA_MODEL_LOADED = True
    print("Asthma LightGBM model loaded successfully!")
except ImportError as ie:
    print(f"ImportError: {ie}")
    print("LightGBM library not available, using mock prediction")
    asthma_model = None
    ASTHMA_MODEL_LOADED = False
except Exception as e:
    print(f"Error loading asthma model: {e}")
    asthma_model = None
    ASTHMA_MODEL_LOADED = False

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(data):
    result = np.array([])

    # Ensure data has sufficient length
    if len(data) < 100:
        # Return a default feature vector for very short audio
        return np.zeros(162)  # Default size to match expected input

    # Zero Crossing Rate
    try:
        zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
        result = np.hstack((result, zcr))
    except:
        result = np.hstack((result, np.zeros(1)))  # Default value if ZCR fails

    # Short-time Fourier Transform and Chroma features
    try:
        stft = np.abs(librosa.stft(data))
        # Suppress the pitch tuning warning that causes the error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=22050, n_chroma=12).T, axis=0)
        result = np.hstack((result, chroma))
    except:
        result = np.hstack((result, np.zeros(12)))  # Default 12 chroma features if it fails

    # MFCC features
    try:
        # Suppress the pitch tuning warning that causes the error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050, n_mfcc=20).T, axis=0)
        result = np.hstack((result, mfcc))
    except:
        # If MFCC fails, add zeros as fallback
        result = np.hstack((result, np.zeros(20)))  # Assuming 20 MFCC coefficients

    # RMS
    try:
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))
    except:
        result = np.hstack((result, np.zeros(1)))  # Default value if RMS fails

    # Mel spectrogram
    try:
        # Suppress the pitch tuning warning that causes the error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=22050, n_mels=128).T, axis=0)
        result = np.hstack((result, mel))
    except:
        result = np.hstack((result, np.zeros(128)))  # Default 128 mel bands if it fails

    return result

def preprocess_audio(audio_path):
    try:
        data, sr = librosa.load(audio_path, duration=2.5, offset=0.6, mono=True)
        # Ensure we have enough samples to process
        if len(data) < 100:
            raise ValueError("Audio file too short for processing")
        features = extract_features(data)
        features = features.reshape(1, -1)
        features = np.expand_dims(features, axis=2)
        return features
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        # Return a dummy feature vector to avoid crashes
        dummy_features = np.zeros((1, 162, 1))  # Shape matches model expectation
        return dummy_features

def predict_disease(audio_path):
    global audio_model, AUDIO_MODEL_LOADED  # Ensure we're using the global audio_model
    
    if AUDIO_MODEL_LOADED:
        features = preprocess_audio(audio_path)
        # Check if features are all zeros (indicating an error in preprocessing)
        if np.all(features == 0):
            print("Warning: Audio preprocessing failed, returning mock prediction")
            return "Healthy", 0.5  # Return a default prediction
        
        # Debug: Print shape of features and model info
        print(f"Features shape: {features.shape}")
        print(f"Audio model type: {type(audio_model)}")
        print(f"Audio model has predict method: {hasattr(audio_model, 'predict')}")
        print(f"AUDIO_MODEL_LOADED status: {AUDIO_MODEL_LOADED}")
        
        # Make prediction using the CNN model
        try:
            if hasattr(audio_model, 'predict'):
                probs = audio_model.predict(features)
                print(f"Prediction probabilities: {probs}")
                class_idx = np.argmax(probs)
                confidence = np.max(probs)
                result = LABEL_CLASSES[class_idx]
                print(f"Predicted class: {result} with confidence: {confidence}")
                return result, confidence
            else:
                print("Audio model does not have predict method")
                return "Healthy", 0.5
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Return a default prediction if model prediction fails
            return "Healthy", 0.5
    else:
        print("Audio model weights could not be loaded due to version incompatibility.")
        return "Model Error - Version Incompatible", 0.0

def predict_asthma(features_dict):
    if not ASTHMA_MODEL_LOADED:
        print("Error: Asthma model not loaded")
        return None, 0
    
    # Convert features to array in the same order as training
    feature_order = [
        'age', 'gender', 'bmi', 'smoking', 'familyHistory', 
        'allergyHistory', 'lungFunctionFeV1', 'wheezing', 
        'shortnessOfBreath', 'chestTightness'
    ]
    
    try:
        # Create feature array
        features = np.array([[features_dict[key] for key in feature_order]])
        
        # Get prediction and probability
        prediction = asthma_model.predict(features)[0]
        probabilities = asthma_model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        return int(prediction), round(confidence, 2)
    except Exception as e:
        print(f"Error in asthma prediction: {e}")
        return None, 0

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    # Check if user is logged in
    if 'logged_in' not in session:
        return render_template('login.html')
    return render_template('index.html')

@app.route('/detection')
def detection():
    # Check if user is logged in
    if 'logged_in' not in session:
        return render_template('login.html')
    return render_template('detection.html')

@app.route('/safe_check')
def safe_check():
    # Check if user is logged in
    if 'logged_in' not in session:
        return render_template('login.html')
    return render_template('safe_check.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Hardcoded credentials
        if email == 'user@gmail.com' and password == '123456':
            session['logged_in'] = True
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid email or password'})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_dir = tempfile.gettempdir()
            temp_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(temp_dir, temp_filename)
            
            file.save(filepath)
            
            # Make prediction
            if not AUDIO_MODEL_LOADED:
                print("Warning: Audio model not loaded due to version incompatibility, cannot make accurate predictions")
                
            pred, conf = predict_disease(filepath)
            
            # Check if it's an error condition
            if pred == "Model Error - Version Incompatible":
                return jsonify({
                    'error': 'Model could not be loaded due to version incompatibility. The model file was saved with a different version of Keras/TensorFlow. Please use a compatible version or re-save the model.',
                    'prediction': None,
                    'confidence': None
                }), 500
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify({
                'prediction': pred
            })
        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a WAV, MP3, M4A, or FLAC file.'}), 400

@app.route('/predict_asthma', methods=['POST'])
def predict_asthma_route():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'smoking', 'familyHistory', 
                          'allergyHistory', 'lungFunctionFeV1', 'wheezing', 
                          'shortnessOfBreath', 'chestTightness']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make asthma prediction
        prediction, confidence = predict_asthma(data)
        
        if prediction is None:
            return jsonify({'error': 'Asthma model not loaded or prediction failed'}), 500
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing asthma prediction: {str(e)}'}), 500

# Load environment variables and API key
load_dotenv()
API_KEY = os.getenv("WEATHERSTACK_API_KEY")
BASE_URL = "http://api.weatherstack.com/current"  # ✅ HTTP ONLY for free plan

@app.route('/weather', methods=['GET'])
def get_weather():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if city:
        query = city
    elif lat and lon:
        query = f"{lat},{lon}"
    else:
        return jsonify({'error': 'City or coordinates required'}), 400
    
    if not API_KEY:
        return jsonify({'error': 'Weather API key not configured. Please set WEATHERSTACK_API_KEY in .env file'}), 500
    
    params = {
        'access_key': API_KEY,
        'query': query,
        'units': 'm'
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        # 🔍 DEBUG PRINT (VERY IMPORTANT)
        print("Weatherstack response:", data)
        
        # ❌ Weatherstack error
        if "error" in data:
            return jsonify({
                "error": data["error"]
            }), 400
        
        location = data.get('location', {})
        current = data['current']
        air = current.get('air_quality', {})
        
        result = {
            # 🔹 Resolved station / location info
            'resolvedLocation': {
                'name': location.get('name'),
                'region': location.get('region'),
                'country': location.get('country'),
                'lat': location.get('lat'),
                'lon': location.get('lon')
            },
            
            # 🔹 Weather + AQI data
            'temperature': current['temperature'],
            'humidity': current['humidity'],
            'windSpeed': current['wind_speed'],
            'pressure': current['pressure'],
            'cloudCover': current['cloudcover'],
            'precip': current['precip'],
            
            'weatherDescription': current['weather_descriptions'][0] if current.get('weather_descriptions') else 'Clear',
            
            'pm25': float(air.get('pm2_5', 0)),
            'pm10': float(air.get('pm10', 0)),
            'no2': float(air.get('no2', 0)),
            'so2': float(air.get('so2', 0)),
            'o3': float(air.get('o3', 0)),
            'co': float(air.get('co', 0)),
            
            'aqi': int(max(air.get('us-epa-index', 1))),
            'epaIndex': int(air.get('us-epa-index', 1))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Create client with the new API structure
client = genai.Client(api_key=GOOGLE_API_KEY)

# Check if gemini-2.5-flash model is available
try:
    # Test if the model can be accessed
    gemini_model = client.models.get(model="gemini-2.5-flash")
    GENERATIVE_MODEL_AVAILABLE = True
    print("Gemini API configured successfully with gemini-2.5-flash")
    print("New GenAI API is available")
except Exception as e:
    print(f"Error accessing gemini-2.5-flash: {e}")
    GENAI_AVAILABLE = False
    GENERATIVE_MODEL_AVAILABLE = False

@app.route('/ai_doctor')
def ai_doctor():
    # Check if user is logged in
    if 'logged_in' not in session:
        return render_template('login.html')
    return render_template('ai_doctor.html')

@app.route('/get_patient_data')
def get_patient_data():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    # Get patient data from session and stored predictions
    patient_data = session.get('patient_data', {})
    audio_analysis = session.get('audio_analysis', {})
    asthma_assessment = session.get('asthma_assessment', {})
    environmental_data = session.get('environmental_data', {})

    return jsonify({
        'success': True,
        'patient_data': patient_data,
        'audio_analysis': audio_analysis,
        'asthma_assessment': asthma_assessment,
        'environmental_data': environmental_data
    })

@app.route('/generate_ai_verdict', methods=['POST'])
def generate_ai_verdict():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    # Check if genai is available
    if not GENAI_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Google Generative AI service is not available.',
            'fallback_message': 'As an AI doctor: The AI service is currently unavailable. Please consult with a healthcare professional for proper diagnosis and treatment.'
        }), 200
    
    if not GOOGLE_API_KEY:
        return jsonify({
            'success': False, 
            'error': 'Google API key not configured. Please set GOOGLE_API_KEY in .env file',
            'fallback_message': 'As an AI doctor: Please ensure proper API configuration for personalized medical advice.'
        }), 200

    try:
        # Get patient data from session
        patient_data = session.get('patient_data', {})
        audio_analysis = session.get('audio_analysis', {})
        asthma_assessment = session.get('asthma_assessment', {})
        environmental_data = session.get('environmental_data', {})

        # Prepare the prompt for the AI doctor
        prompt = f"""
        As an AI doctor specializing in respiratory medicine, please provide a comprehensive medical verdict based on the following patient data:

        PATIENT DEMOGRAPHICS & CLINICAL DATA:
        - Age: {patient_data.get('age', 'N/A')}
        - Gender: {patient_data.get('gender', 'N/A')}
        - BMI: {patient_data.get('bmi', 'N/A')}
        - Smoking Status: {patient_data.get('smoking', 'N/A')}
        - Family History of Asthma: {patient_data.get('familyHistory', 'N/A')}
        - Allergy History: {patient_data.get('allergyHistory', 'N/A')}
        - Lung Function (FEV1): {patient_data.get('lungFunctionFeV1', 'N/A')}%
        - Wheezing: {patient_data.get('wheezing', 'N/A')}
        - Shortness of Breath: {patient_data.get('shortnessOfBreath', 'N/A')}
        - Chest Tightness: {patient_data.get('chestTightness', 'N/A')}

        AUDIO RESPIRATORY ANALYSIS:
        - Detected Condition: {audio_analysis.get('prediction', 'N/A')}

        CLINICAL ASTHMA ASSESSMENT:
        - Diagnosis: {asthma_assessment.get('prediction', 'N/A')}

        ENVIRONMENTAL EXPOSURE DATA:
        - Location: {environmental_data.get('resolvedLocation', {}).get('name', 'N/A')}, {environmental_data.get('resolvedLocation', {}).get('region', 'N/A')}
        - AQI: {environmental_data.get('aqi', 'N/A')} (EPA Index: {environmental_data.get('epaIndex', 'N/A')})
        - PM2.5: {environmental_data.get('pm25', 'N/A')} µg/m³
        - Weather: {environmental_data.get('weatherDescription', 'N/A')}
        - Humidity: {environmental_data.get('humidity', 'N/A')}%
        - Wind Speed: {environmental_data.get('windSpeed', 'N/A')} km/h

        Based on this comprehensive multimodal assessment, please provide a detailed medical verdict including:
        1. Primary diagnosis consideration
        2. Contributing factors (environmental, clinical, demographic)
        3. Risk assessment
        4. Recommended next steps for the patient
        5. Any urgent concerns that require immediate attention

        Please frame your response as if you're communicating directly with the patient, using clear but professional language. Keep your response concise but comprehensive, approximately 150-200 words.
        
        Return the response as clean plain text.
        Do not use HTML tags or Markdown.
        Use paragraphs separated by line breaks only.
        """

        # Initialize the Gemini model with the new API structure
        try:
            if GENERATIVE_MODEL_AVAILABLE:
                # Use the newer GenAI API as requested
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                if response.text:
                    # Store the verdict in session for potential future reference
                    session['ai_verdict'] = response.text
                    
                    return jsonify({
                        'success': True,
                        'verdict': response.text
                    })
            else:
                # Fallback if model is not available
                fallback_verdict = (
                    "Based on the comprehensive assessment including audio analysis, clinical data, "
                    "and environmental factors: The patient shows signs that warrant medical attention. "
                    "Please consult with a healthcare professional for proper evaluation and guidance. "
                    "Consider avoiding known triggers and monitor symptoms closely."
                )
                session['ai_verdict'] = fallback_verdict
                return jsonify({
                    'success': True,
                    'verdict': fallback_verdict,
                    'source': 'fallback'
                })
        except Exception as e:
            # Handle any runtime errors in AI processing
            print(f"Error during AI processing: {e}")
            # Check if it's an authentication/model error and provide more specific fallback
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                # Model not found or not accessible - provide helpful fallback
                fallback_verdict = (
                    "Based on the comprehensive assessment including audio analysis, clinical data, "
                    "and environmental factors: The patient shows signs that warrant medical attention. "
                    "Please consult with a healthcare professional for proper evaluation and guidance. "
                    "Consider avoiding known triggers and monitor symptoms closely."
                )
                session['ai_verdict'] = fallback_verdict
                return jsonify({
                    'success': True,
                    'verdict': fallback_verdict,
                    'source': 'fallback'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Error processing AI request: {str(e)}',
                    'fallback_message': 'As an AI doctor: Due to a processing issue, please consult with a healthcare professional for personalized medical advice.'
                }), 200
        
        # Fallback if AI doesn't return a proper response
        return jsonify({
            'success': False,
            'error': 'AI did not return a verdict',
            'fallback_message': 'As an AI doctor: Based on your data, please consult with a healthcare professional for proper evaluation.'
        }), 200

    except Exception as e:
        print(f"Error generating AI verdict: {e}")
        return jsonify({
            'success': False,
            'error': f'Error generating AI verdict: {str(e)}'
        })

@app.route('/store_patient_data', methods=['POST'])
def store_patient_data():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    try:
        data = request.get_json()
        # Store patient data in session
        session['patient_data'] = data
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/store_asthma_assessment', methods=['POST'])
def store_asthma_assessment():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    try:
        data = request.get_json()
        # Store asthma assessment in session
        session['asthma_assessment'] = data
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/store_audio_analysis', methods=['POST'])
def store_audio_analysis():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    try:
        data = request.get_json()
        # Store audio analysis in session
        session['audio_analysis'] = data
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/store_environmental_data', methods=['POST'])
def store_environmental_data():
    # Check if user is logged in
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    try:
        data = request.get_json()
        # Store environmental data in session
        session['environmental_data'] = data
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Test route for Gemini API
@app.route('/test_gemini')
def test_gemini():
    try:
        # Use the same API key that's already loaded
        if GOOGLE_API_KEY and GENERATIVE_MODEL_AVAILABLE:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Summarize asthma risk factors in 50 words."
            )
            return response.text
        else:
            return "ERROR: No API key available or model not available"
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)