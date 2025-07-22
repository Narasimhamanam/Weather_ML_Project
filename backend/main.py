from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import requests
import configparser
import os

app = FastAPI()

# === Allow CORS for React frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models and Encoders ===
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
rain_model = joblib.load(os.path.join(MODEL_DIR, 'rain_forest_model.joblib'))
temp_model = joblib.load(os.path.join(MODEL_DIR, 'temp_forest_model.joblib'))
encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.joblib'))

# === Load API Key ===
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'data', 'secrets.ini'))
API_KEY = config['openweathermap']['api_key']
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

# === Live Prediction Route ===
@app.get("/live-predict")
def live_predict(city: str):
    try:
        # Step 1: Fetch live data
        params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"City not found or API error: {data.get('message')}")

        # Step 2: Prepare feature values
        features = {
            'MinTemp': data['main']['temp_min'],
            'MaxTemp': data['main']['temp_max'],
            'Rainfall': data.get('rain', {}).get('1h', 0.0),
            'Evaporation': 3.0,  # Placeholder
            'Sunshine': 6.0,     # Placeholder
            'WindGustSpeed': data['wind']['speed'],
            'WindSpeed9am': data['wind']['speed'],
            'WindSpeed3pm': data['wind']['speed'],
            'Humidity9am': data['main']['humidity'],
            'Humidity3pm': data['main']['humidity'],
            'Pressure9am': data['main']['pressure'],
            'Pressure3pm': data['main']['pressure'],
            'Cloud9am': 4.0,     # Placeholder
            'Cloud3pm': 4.0,     # Placeholder
            'Temp9am': data['main']['temp'],
            'Temp3pm': data['main']['temp'],
            'WindGustDir': 'N',   # Placeholder categorical
            'WindDir9am': 'N',
            'WindDir3pm': 'N',
            'RainToday': 'No'
        }

        # Step 3: Encode and format input
        cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        values = [features[key] for key in features if key not in cat_cols]
        for col in cat_cols:
            encoder = encoders[col]
            val = features[col]
            if val not in encoder.classes_:
                val = encoder.classes_[0]
            values.append(encoder.transform([val])[0])

        input_arr = np.array(values).reshape(1, -1)

        # Step 4: Make Predictions
        rain_pred = rain_model.predict(input_arr)[0]
        rain_prob = rain_model.predict_proba(input_arr)[0][1]
        temp_pred = temp_model.predict(input_arr)[0]

        # Step 5: Return result
        return {
            "city": city,
            "rain_tomorrow": bool(rain_pred),
            "rain_probability": round(float(rain_prob), 3),
            "max_temperature_prediction": round(float(temp_pred), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Root Route ===
@app.get("/")
def root():
    return {
        "message": "✅ Weather Prediction API running!",
        "docs": "/docs",
        "example": "/live-predict?city=Guntur"
    }
