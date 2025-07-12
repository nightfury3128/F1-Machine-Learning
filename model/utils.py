import pandas as pd
from supabase import create_client, Client

url = "https://azpzuawybizqujhlmlqu.supabase.co"
key =  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6cHp1YXd5Yml6cXVqaGxtbHF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3MzIxNDcsImV4cCI6MjA2NzMwODE0N30.7r_1VYLlJ6ysp5zFOFh6Rhx93sqToeS5qobF1fRmkVs"
supabase = create_client(url, key)

def get_table_df(table_name, batch_size=1000):
    all_data = []
    start = 0

    while True:
        print(f"Fetching data from {table_name} starting at index {start}")
        response = supabase.table(table_name).select("*").range(start, start + batch_size - 1).execute() #PAGINATION
        data_batch = response.data
        if not data_batch:
            break
        all_data.extend(data_batch)
        start += batch_size

    return pd.DataFrame(all_data)


results_df = get_table_df("results")
drivers_df = get_table_df("drivers")
laps_df = get_table_df("laps")
weather_df = get_table_df("weather")
sessions_df = get_table_df("sessions")
pitstops_df = get_table_df("pitstops")
positions_df = get_table_df("positions")

