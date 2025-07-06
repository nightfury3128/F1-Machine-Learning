from supabase import create_client, Client
import pandas as pd

SUPABASE_URL = "https://azpzuawybizqujhlmlqu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6cHp1YXd5Yml6cXVqaGxtbHF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3MzIxNDcsImV4cCI6MjA2NzMwODE0N30.7r_1VYLlJ6ysp5zFOFh6Rhx93sqToeS5qobF1fRmkVs"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_table_df(table_name):
    response = supabase.table(table_name).select("*").execute()
    return pd.DataFrame(response.data)

results_df = get_table_df("results")
drivers_df = get_table_df("drivers")
laps_df = get_table_df("laps")
weather_df = get_table_df("weather")
sessions_df = get_table_df("sessions")
pitstops_df = get_table_df("pitstops")
positions_df = get_table_df("positions")