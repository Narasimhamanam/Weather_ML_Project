import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # ✅ Add this line!


# === 1. Load Dataset ===
# file_path = 'weather_australia.csv'
# print("[INFO] Loading dataset...")
# if not os.path.exists(file_path):
#     raise FileNotFoundError("Dataset not found! Make sure you've placed it in `data/weather_australia.csv`.")

df = pd.read_csv(r"C:\Users\naras\OneDrive\Desktop\MachineLearning\ML-Projects\Weather_predictor\model\weatherAUS.csv")

# === 2. Basic Data Cleaning ===
cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

print("[INFO] Handling missing values...")
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows with missing target
df = df.dropna(subset=['RainTomorrow'])
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# === 3. Encode Categorical Values ===
print("[INFO] Encoding categorical variables...")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save all encoders together
joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.joblib'))

# === 4. Feature Selection ===
features = num_cols + cat_cols
X = df[features]
y_rain = df['RainTomorrow']
y_temp = df['MaxTemp']

# === 5. Model Training ===
print("[INFO] Training models...")
X_train, X_test, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)
_, _, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)

# 1. Rain Prediction (Classification)
rain_model = RandomForestClassifier(n_estimators=10, random_state=42)
rain_model.fit(X_train, y_rain_train)
rain_preds = rain_model.predict(X_test)
accuracy = accuracy_score(y_rain_test, rain_preds)

# 2. Temperature Prediction (Regression)
temp_model = RandomForestRegressor(n_estimators=10, random_state=42)
temp_model.fit(X_train, y_temp_train)
temp_preds = temp_model.predict(X_test)
mae = mean_absolute_error(y_temp_test, temp_preds)

# === 6. Save Models ===
print("[INFO] Saving models to disk...")

# Save all encoders together
joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.joblib'))

# Save models
joblib.dump(rain_model, os.path.join(MODEL_DIR, 'rain_forest_model.joblib'))
joblib.dump(temp_model, os.path.join(MODEL_DIR, 'temp_forest_model.joblib'))

print("\n✅ Models & encoders saved successfully!")



# === 7. Report ===
print("\n✅ Training Complete!")
print(f"🌧️  Rain Prediction Accuracy: {accuracy:.2f}")
print(f"🌡️  Temperature Prediction MAE: {mae:.2f} °C")

print("\n✅ Models saved successfully!")
