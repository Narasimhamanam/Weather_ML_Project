import joblib
import numpy as np
import os

# === Load Models ===
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

rain_model = joblib.load(os.path.join(MODEL_DIR, 'rain_forest_model.joblib'))
temp_model = joblib.load(os.path.join(MODEL_DIR, 'temp_forest_model.joblib'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))

# === Create Test Input (change values as needed) ===
# Must match features: num_cols + cat_cols
# Order:  MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed,
#         WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am,
#         Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm,
#         WindGustDir, WindDir9am, WindDir3pm, RainToday

# Example values (replace these with actual/real-time test values):
test_sample = [
    15.0,   # MinTemp
    25.0,   # MaxTemp
    1.2,    # Rainfall
    5.0,    # Evaporation
    8.0,    # Sunshine
    37,     # WindGustSpeed
    13,     # WindSpeed9am
    19,     # WindSpeed3pm
    70,     # Humidity9am
    50,     # Humidity3pm
    1012.5, # Pressure9am
    1010.2, # Pressure3pm
    5.0,    # Cloud9am
    4.0,    # Cloud3pm
    17.5,   # Temp9am
    24.0,   # Temp3pm
    'W',    # WindGustDir
    'NW',   # WindDir9am
    'SE',   # WindDir3pm
    'No'    # RainToday
]

# Encode categorical values
cat_start = 16
# Define column names in order
cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.joblib'))

for i, col in enumerate(cat_cols):
    value = test_sample[16 + i]
    
    # Handle unknowns: use 'most frequent' fallback
    encoder = encoders[col]
    if value not in encoder.classes_:
        print(f"[WARN] '{value}' not in training data for '{col}'; using first class as fallback.")
        value = encoder.classes_[0]
    
    test_sample[16 + i] = encoder.transform([value])[0]

# Convert to 2D array for prediction
X_input = np.array(test_sample).reshape(1, -1)

# === Make Predictions ===
rain_prob = rain_model.predict_proba(X_input)[0][1]
rain_prediction = rain_model.predict(X_input)[0]
temp_prediction = temp_model.predict(X_input)[0]

# === Display Results ===
print("\n🔍 TEST PREDICTION:")
print(f"➡️  Rain Tomorrow: {'Yes 🌧️' if rain_prediction == 1 else 'No ☀️'} (Confidence: {rain_prob:.2f})")
print(f"➡️  Predicted Max Temperature: {temp_prediction:.2f} °C")
