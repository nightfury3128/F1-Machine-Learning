import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import weather_df, results_df, drivers_df, laps_df, sessions_df, pitstops_df, positions_df

feature_cols = [
    'driver_number_x',
    'starting_position',
    'pit_duration',
    'circuit_key',
    'location',
    'session_type',
    'rainfall',
    'air_temperature',
    'track_temperature',
    'wind_direction',
    'wind_speed',
]

def data():
    dnf_data = pd.merge(results_df, pitstops_df, on='session_key', how='left')
    dnf_data = pd.merge(dnf_data, sessions_df, on='session_key', how='left')
    dnf_data = pd.merge(dnf_data, weather_df, on='session_key', how='left')
    return dnf_data

def train():
    dnf_data = data()
    dnf_data = dnf_data.dropna(subset=['driver_number_x', 'session_key', 'laps_completed', 'dnf'])
    dnf_data['dnf'] = dnf_data['dnf'].astype(int)

    X = dnf_data[feature_cols]
    y = dnf_data['dnf']

    # One-hot encode categorical columns
    X = pd.get_dummies(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.5, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Return model and training columns
    return model, X_train.columns

model, train_columns = train()

# Static race info
race_info = {
    'circuit_key': 'silverstone',
    'location': 'silverstone',
    'session_type': 'race'
}

# Static weather
weather_forecast = {
    'rainfall': 0.0,  
    'air_temperature': 18.5,
    'track_temperature': 18.5,  # assume track temp ~= air temp (or adjust)
    'wind_speed': 18.5,
    'wind_direction': 331,
}

drivers = [
    {'driver_number_x': 1, 'starting_position': 1, 'driver_name': 'M. Verstappen'},
    {'driver_number_x': 81, 'starting_position': 2, 'driver_name': 'O. Piastri'},
    {'driver_number_x': 4, 'starting_position': 3, 'driver_name': 'L. Norris'},
    {'driver_number_x': 63, 'starting_position': 4, 'driver_name': 'G. Russell'},
    {'driver_number_x': 44, 'starting_position': 5, 'driver_name': 'L. Hamilton'},
    {'driver_number_x': 16, 'starting_position': 6, 'driver_name': 'C. Leclerc'},
    {'driver_number_x': 12, 'starting_position': 7, 'driver_name': 'A.K. Antonelli'},
    {'driver_number_x': 87, 'starting_position': 8, 'driver_name': 'O. Bearman'},
    {'driver_number_x': 14, 'starting_position': 9, 'driver_name': 'F. Alonso'},
    {'driver_number_x': 10, 'starting_position': 10, 'driver_name': 'P. Gasly'},
    {'driver_number_x': 55, 'starting_position': 11, 'driver_name': 'C. Sainz Jr.'},
    {'driver_number_x': 22, 'starting_position': 12, 'driver_name': 'Y. Tsunoda'},
    {'driver_number_x': 6, 'starting_position': 13, 'driver_name': 'I. Hadjar'},
    {'driver_number_x': 23, 'starting_position': 14, 'driver_name': 'A. Albon'},
    {'driver_number_x': 31, 'starting_position': 15, 'driver_name': 'E. Ocon'},
    {'driver_number_x': 30, 'starting_position': 16, 'driver_name': 'L. Lawson'},
    {'driver_number_x': 5, 'starting_position': 17, 'driver_name': 'G. Bortoleto'},
    {'driver_number_x': 18, 'starting_position': 18, 'driver_name': 'L. Stroll'},
    {'driver_number_x': 27, 'starting_position': 19, 'driver_name': 'N. Hülkenberg'},
    {'driver_number_x': 43, 'starting_position': 20, 'driver_name': 'F. Colapinto'}
]



df = pd.DataFrame(drivers)

df['pit_duration'] = 0  # no pit duration yet
df['circuit_key'] = race_info['circuit_key']
df['location'] = race_info['location']
df['session_type'] = race_info['session_type']

df['rainfall'] = weather_forecast['rainfall']
df['air_temperature'] = weather_forecast['air_temperature']
df['track_temperature'] = weather_forecast['track_temperature']
df['wind_speed'] = weather_forecast['wind_speed']
df['wind_direction'] = weather_forecast['wind_direction']


X = df[feature_cols]

X = pd.get_dummies(X)

dnf_data = data()
dnf_data = dnf_data.dropna(subset=feature_cols + ['dnf'])
dnf_data['dnf'] = dnf_data['dnf'].astype(int)
X_train = dnf_data[feature_cols]
X_train = pd.get_dummies(X_train)

# Align columns: add missing columns with 0
X = X.reindex(columns=X_train.columns, fill_value=0)

# Predict
df['dnf_prediction'] = model.predict(X)

# Print who is predicted to DNF
dnf_drivers = df[df['dnf_prediction'] == 1]
print("Predicted to DNF:")
print(dnf_drivers[['driver_name', 'starting_position']])