import requests
import pandas as pd
import time
import threading 
base_url = 'https://api.openf1.org/v1/'

def session(year, session_name=''):
    url = base_url + 'sessions'
    params = {
        'year': year,
        'session_type': session_name,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    return response.json()

def multiple_sessions(years, session_names):
    all_data = []
    for year in years:
        print(f"Fetching data for year: {year}")
        for se in session_names:    
            print(f"Fetching session: {se} for year: {year}")
            data = session(year, se)
            all_data.extend(data)
            print(all_data)
    return all_data

def get_positions(session_key):
    url = base_url + 'position'
    params = {
        'session_key': session_key,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching positions: {response.status_code} - {response.text}")
    return response.json()

def get_driver_info(driver_number):
    url = base_url + 'drivers'
    params = {
        'driver_number': driver_number,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching driver info: {response.status_code} - {response.text}")
    return response.json()

def get_laps(session_key):
    url = base_url + 'laps'
    params = {
        'session_key': session_key,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching laps: {response.status_code} - {response.text}")
    return response.json()

def get_weather(session_key):
    url = base_url + 'weather'
    params = {
        'session_key': session_key,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching weather: {response.status_code} - {response.text}")
    return response.json()

def get_pitstops(session_key):
    url = base_url + 'pit'
    params = {
        'session_key': session_key,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching pitstops: {response.status_code} - {response.text}")
    return response.json()

def extract(years, session_names):
    print("Extracting session data...")
    session_data = multiple_sessions(years, session_names)
    session_df = pd.DataFrame([
    {k: item[k] for k in ['year', 'session_key', 'circuit_key', 'location', 'session_type','circuit_name'] if k in item}
    for item in session_data
])

    print(session_df)
    session_df.to_csv('session_df.csv', index=False)
    print("Session data extracted successfully.")

    all_positions = []
    print("Extracting position data...")
    for session_key in session_df['session_key']:
        data = get_positions(session_key)
        all_positions.extend(data)
        time.sleep(1)
    pos_df = pd.DataFrame([
        {k: v for k, v in item.items() if k in ['session_key', 'driver_number', 'position', 'date']}
        for item in all_positions
    ])
    pos_df['date'] = pd.to_datetime(pos_df['date'], format = 'mixed')
    print("Position data extracted successfully.")


    print("Extracting driver information...")
    driver_info = []
    for driver_number in pos_df['driver_number'].unique():
        info = get_driver_info(driver_number)
        driver_info.extend(info)
        time.sleep(1)
    driver_df = pd.DataFrame(driver_info)
    driver_df = driver_df[['driver_number', 'full_name', 'team_name']]
    print("Driver information extracted successfully.")

    

    print("Extracting lap data...")
    all_laps = []
    for session_key in session_df['session_key']:
        laps = get_laps(session_key)
        all_laps.extend(laps)
        time.sleep(1)
    laps_df = pd.DataFrame([
        {k: v for k, v in item.items() if k in ['session_key', 'driver_number', 'lap_number']}
        for item in all_laps])
    print("Lap data extracted successfully.")

    print("Extracting weather data...")
    weather_data = []
    for session_key in session_df['session_key']:
        weather = get_weather(session_key)
        weather_data.extend(weather)
        time.sleep(1)
    weather_df = pd.DataFrame({k: v for k, v in item.items() if k in ['session_key', 'air_temperature', 'track_temperature', 'wind_direction', 'wind_speed', 'humidity', 'rainfall','date']} for item in weather_data)
    
    pit=[]
    for session_key in session_df['session_key']:
        pit_data = get_pitstops(session_key)
        pit.extend(pit_data)
        time.sleep(1)
    pit_df = pd.DataFrame(pit)
    pit_df = pit_df[['session_key', 'driver_number', 'pit_duration', 'date', 'lap_number']]
    pit_df['date'] = pd.to_datetime(pit_df['date'], format = 'mixed')
    return {
        'session_df': session_df,
        'driver_df': driver_df,
        'pos_df': pos_df,
        'laps_df': laps_df,
        'weather_df': weather_df,
        'pit_df': pit_df
    }

def finalize_data(session_df, pos_df, driver_df, laps_df):
    starting_pos = (
        pos_df.sort_values('date')
        .groupby(['session_key', 'driver_number'])
        .first()
        .reset_index()
        .rename(columns={'position': 'starting_position'})
        [['session_key', 'driver_number', 'starting_position']]
    )
    final_pos = (
        pos_df.sort_values('date')     
        .groupby(['session_key', 'driver_number'])
        .last()
        .reset_index()
        .rename(columns={'position': 'final_position'})
        [['session_key', 'driver_number', 'final_position']]
    )

    lap_counts = (
        laps_df.groupby(['session_key', 'driver_number'])['lap_number']
        .max()
        .reset_index()
        .rename(columns={'lap_number': 'laps_completed'})
    )
    max_laps_per_session = (
        lap_counts.groupby('session_key')['laps_completed'] 
        .max()
        .reset_index()
        .rename(columns={'laps_completed': 'max_laps'})
    )
    lap_counts = pd.merge(lap_counts, max_laps_per_session, on='session_key', how='left')
    lap_counts['dnf'] = lap_counts['laps_completed'] < lap_counts['max_laps']

    final = pd.merge(final_pos, starting_pos, on=['session_key', 'driver_number'], how='left')
    final = pd.merge(final, session_df, on='session_key', how='left')
    final['position_change'] = final['starting_position'] - final['final_position']
    final = pd.merge(final, lap_counts[['session_key', 'driver_number', 'laps_completed', 'dnf']],
                     on=['session_key', 'driver_number'], how='left')
    
    return final.drop_duplicates()


def main():
    """Main function using DataFrame approach"""
    years = [2025]
    session_names = ['Race']
    
    # Extract all data as DataFrames
    extracted_data = extract(years, session_names)
    
    if not extracted_data:
        print("No data extracted. Exiting.")
        return
    
    print("\nExtracted data summary:")
    print(f"Sessions: {len(extracted_data['session_df'])}")
    print(f"Positions: {len(extracted_data['pos_df'])}")
    print(f"Drivers: {len(extracted_data['driver_df'])}")
    print(f"Laps: {len(extracted_data['laps_df'])}")

    # Calculate race results using DataFrames
    race_results_df = finalize_data(
        extracted_data['session_df'],
        extracted_data['pos_df'],
        extracted_data['driver_df'],
        extracted_data['laps_df']
    )
    
    print(f"Race Results: {len(race_results_df)}")
    print("\nSample race results:")
    print(race_results_df.head())

if __name__ == "__main__":
    main()