import pandas as pd
from supabase import create_client, Client

url = "https://azpzuawybizqujhlmlqu.supabase.co"
key =  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6cHp1YXd5Yml6cXVqaGxtbHF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3MzIxNDcsImV4cCI6MjA2NzMwODE0N30.7r_1VYLlJ6ysp5zFOFh6Rhx93sqToeS5qobF1fRmkVs"
supabase = create_client(url, key)

def get_table_df(table_name, batch_size=1000, session_key=None):
    all_data = []
    start = 0
    while True:
        print(f"Fetching data from {table_name} starting at index {start}")
        if session_key:
            query = supabase.table(table_name).select("*").eq("session_key", session_key).range(start, start + batch_size - 1)
        else:
            query = supabase.table(table_name).select("*").range(start, start + batch_size - 1)
        response = query.execute()
        data_batch = response.data
        if not data_batch:
            break
        all_data.extend(data_batch)
        start += batch_size

    return pd.DataFrame(all_data)

