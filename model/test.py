from utils import get_table_df
import pandas as pd

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

# 3. Merge all tables on session_key and driver_number where appropriate
df = positions_df.merge(sessions_df, on='session_key', how='left')
df = df.merge(weather_df, on='session_key', how='left')
df = df.merge(results_df, on=['session_key', 'driver_number'], how='left')
df = df.merge(pitstops_df, on=['session_key', 'driver_number'], how='left')
df = df.merge(drivers_df, on='driver_number', how='left')

# 4. Save merged dataframe
df.to_csv("merged_positions_10000.csv", index=False)
print("Merged data saved to merged_positions_10000.csv")