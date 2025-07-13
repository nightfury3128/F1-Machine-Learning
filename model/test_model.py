from utils import get_table_df
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

SESSION_KEY_TO_TEST = '9693'  

# Query only the relevant data for the session_key
weather_df = get_table_df("weather", session_key=SESSION_KEY_TO_TEST).drop(columns=['id'])
sessions_df = get_table_df("sessions", session_key=SESSION_KEY_TO_TEST).drop(columns=['id'])
results_df = get_table_df("results", session_key=SESSION_KEY_TO_TEST).drop(columns=['id'])
positions_df = get_table_df("positions", session_key=SESSION_KEY_TO_TEST).drop(columns=['id'])
pit_df = get_table_df("pitstops", session_key=SESSION_KEY_TO_TEST).drop(columns=['id'])
drivers_df = get_table_df("drivers").drop(columns=['id'])

# Merge filtered dataframes
df = pd.merge(sessions_df, weather_df, on='session_key', how='left')
df = pd.merge(df, results_df, on='session_key', how='left')
df = pd.merge(df, drivers_df, on='driver_number', how='left')
df = pd.merge(df, positions_df, on=['session_key', 'driver_number'], how='left')

feature_cols = [
    'year', 'circuit_key', 'location', 'air_temperature', 'track_temperature',
    'wind_direction', 'wind_speed', 'humidity', 'rainfall',
    'starting_position', 'position_change', 'laps_completed',
    'team_name', 'full_name'
]

df = df.drop_duplicates()
df = df.drop_duplicates(subset=['driver_number'])  # Keep only one row per driver

df = df.dropna(subset=feature_cols + ['final_position'])

if df.empty:
    print(f"No data found for session_key: {SESSION_KEY_TO_TEST}")
    exit()

X = pd.get_dummies(df[feature_cols], drop_first=True)

# Load the columns used during training
model_columns = joblib.load("model_columns.pkl")
X = X.reindex(columns=model_columns, fill_value=0)

y = df['final_position'].apply(lambda x: 1 if x <= 3 else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = tf.keras.models.load_model("prediction.h5")

# Predict
y_pred_prob = model.predict(X_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Evaluate
accuracy = (y_pred == y.values).mean()
print(f"Test accuracy for session_key {SESSION_KEY_TO_TEST}: {accuracy:.2f}")
print("Predictions:", y_pred)
print("True labels:", y.values)

# Show driver name, predicted, and actual final position (per driver)
df_result = df.copy()
df_result['predicted_top3'] = y_pred
df_result['actual_top3'] = y.values
print("\nDriver results for session:")
for _, row in df_result.iterrows():
    print(f"Driver: {row['full_name']}, Team: {row['team_name']}, Final Position: {row['final_position']}, Predicted Top 3: {row['predicted_top3']}, Actual Top 3: {row['actual_top3']}")
print(f"Test accuracy for session_key {SESSION_KEY_TO_TEST}: {accuracy:.2f}")
print("Predictions:", y_pred)
print("True labels:", y.values)
