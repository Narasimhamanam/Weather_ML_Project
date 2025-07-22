import requests
import configparser
import pandas as pd
from datetime import datetime
import os

# === Load API Key from secrets.ini ===
config = configparser.ConfigParser()
config.read('data/secrets.ini')

try:
    API_KEY = config['openweathermap']['api_key']
except KeyError:
    raise ValueError("API key not found. Check data/secrets.ini and ensure it's formatted correctly.")

BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

# === Define function to fetch live weather ===
def fetch_weather(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'
    }
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if response.status_code == 200:
            return {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'city': city,
                'temp': data['main']['temp'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'humidity': data['main']['humidity'],
                'rain': data.get('rain', {}).get('1h', 0),
                'weather_main': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed']
            }
        else:
            print(f"[ERROR] API responded with status {response.status_code}: {data}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network fault: {e}")

    return None

# === Fetch and save data ===
city = 'Guntur'
weather_data = fetch_weather(city)

if weather_data:
    df = pd.DataFrame([weather_data])
    file_path = 'data/weather_history.csv'

    # Create file if not exists
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

    print(f"✅ Weather data for {city} saved.")
else:
    print("❌ Failed to fetch weather data.")
