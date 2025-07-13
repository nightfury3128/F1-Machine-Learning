from utils import get_table_df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import joblib

# 1. Get first 10,000 sessions
sessions_df = get_table_df("sessions").drop(columns=['id']).sort_values("session_key").head(10000)
session_keys = sessions_df['session_key'].unique().tolist()

# 2. Get other tables and filter to these session keys
positions_df = get_table_df("positions").drop(columns=['id'])
positions_df = positions_df[positions_df['session_key'].isin(session_keys)]

weather_df = get_table_df("weather").drop(columns=['id'])
weather_df = weather_df[weather_df['session_key'].isin(session_keys)]

results_df = get_table_df("results").drop(columns=['id'])
results_df = results_df[results_df['session_key'].isin(session_keys)]

pitstops_df = get_table_df("pitstops").drop(columns=['id'])
pitstops_df = pitstops_df[pitstops_df['session_key'].isin(session_keys)]

drivers_df = get_table_df("drivers").drop(columns=['id'])

postions_df = get_table_df("positions").drop(columns=['id'])
postions_df = postions_df[postions_df['session_key'].isin(session_keys)]


df = pd.merge(sessions_df, weather_df, on='session_key', how='left')
df = pd.merge(df, results_df, on='session_key', how='left')
df = pd.merge(df, drivers_df, on='driver_number', how='left')
df = pd.merge(df, postions_df, on=['session_key','driver_number'], how='left')


feature_cols = [
    'year', 'circuit_key', 'location', 'air_temperature', 'track_temperature',
    'wind_direction', 'wind_speed', 'humidity', 'rainfall',
    'starting_position', 'position_change', 'laps_completed',
    'team_name', 'full_name'
]

df = df.drop_duplicates() # This is needed since I think the postions table has a lot of duplicates and I don't have the energy to fix it 

df.to_csv("merged_data.csv", index=False)

# Ignore one session_key completely from the training data for testing in the model with actual data for my own testing
SESSION_KEY_TO_IGNORE = '9693' 
df = df[df['session_key'] != SESSION_KEY_TO_IGNORE]

df = df.dropna(subset=feature_cols + ['final_position'])
print("Dataframe shape after merging:", df.shape)
X = pd.get_dummies(df[feature_cols], drop_first=True)
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Multi-class classification: predict final_position (shift to zero-based)
y = df['final_position'].astype(int) - 1
num_classes = y.nunique()
print(f"Number of position classes: {num_classes}")

print("Splitting start")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Splitting done")

# Build model for multi-class classification
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
print("Model built")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled")
print("Starting training")
# Train
model.fit(X_train, y_train, epochs=2, batch_size=100, validation_split=0.2)
print("Model trained")
# Save the model
model.save("prediction.h5")
print("Model saved to prediciton.h5")
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

print(df.drop_duplicates())