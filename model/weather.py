import requests
from datetime import datetime, timedelta

def get_hourly_weather_forecast():
    lat = 52.0786
    lon = -1.0169
    timezone = 'auto'
    
    # Get tomorrow's date
    tomorrow_date = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'timezone': timezone,
        'hourly': 'temperature_2m,precipitation,wind_speed_10m,wind_direction_10m',
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Extract hourly times and find indices for tomorrow's date
    times = data['hourly']['time']  # e.g., '2025-07-06T00:00'
    
    # Filter indices that belong to tomorrow only
    tomorrow_indices = [i for i, t in enumerate(times) if t.startswith(tomorrow_date)]

    # For demonstration, get all hourly data for tomorrow
    forecast_tomorrow = []
    for idx in tomorrow_indices:
        forecast_tomorrow.append({
            'time': times[idx],
            'temperature_2m': data['hourly']['temperature_2m'][idx],
            'precipitation': data['hourly']['precipitation'][idx],
            'wind_speed_10m': data['hourly']['wind_speed_10m'][idx],
            'wind_direction_10m': data['hourly']['wind_direction_10m'][idx],
            'humidity': data['hourly'].get('relative_humidity_2m')
        })

    return forecast_tomorrow

hourly_forecast = get_hourly_weather_forecast()
for hour_data in hourly_forecast:
    print(hour_data)
