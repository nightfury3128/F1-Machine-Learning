import os 
import pandas as pd
import time
import numpy as np
from data import extract, finalize_data
from dotenv import load_dotenv
from supabase import create_client, Client
import json

# Load environment variables (create .env file for security)
load_dotenv()

url = "https://azpzuawybizqujhlmlqu.supabase.co"
key =  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6cHp1YXd5Yml6cXVqaGxtbHF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3MzIxNDcsImV4cCI6MjA2NzMwODE0N30.7r_1VYLlJ6ysp5zFOFh6Rhx93sqToeS5qobF1fRmkVs"

supabase: Client = create_client(url, key)

def clean_dataframe_for_json(df):
    """Clean DataFrame to make it JSON serializable with enhanced error handling"""
    df_clean = df.copy()
    
    # Convert datetime columns to ISO format strings
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            # Convert datetime to string, handling nulls properly
            df_clean[col] = df_clean[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None)
    
    # Replace problematic values more aggressively
    df_clean = df_clean.replace([np.inf, -np.inf, np.nan], None)
    
    # Handle each column individually with type conversion
    for col in df_clean.columns:
        # First, replace all NaN-like values with None
        df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
        
        # Convert data types to JSON-serializable formats
        if col in ['session_key', 'driver_number', 'position']:
            # Convert to nullable integers
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].apply(lambda x: int(x) if pd.notna(x) else None)
        elif col == 'date':
            # Ensure date is string
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace('nan', None)
        elif df_clean[col].dtype == 'object':
            # Handle string columns
            df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if x is not None and str(x) != 'nan' else None)
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            # Handle numeric columns
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].apply(lambda x: float(x) if pd.notna(x) else None)
        elif pd.api.types.is_bool_dtype(df_clean[col]):
            # Handle boolean columns
            df_clean[col] = df_clean[col].apply(lambda x: bool(x) if pd.notna(x) else None)
    
    return df_clean

def validate_json_record(record):
    """Validate that a record can be JSON serialized"""
    try:
        json.dumps(record)
        return True, None
    except (TypeError, ValueError) as e:
        return False, str(e)

def insert_sessions(session_df):
    """Insert session data into sessions table"""
    try:
        # Remove duplicates based on session_key before inserting
        session_df_unique = session_df.drop_duplicates(subset=['session_key'], keep='first')
        session_df_clean = clean_dataframe_for_json(session_df_unique)
        sessions_to_insert = session_df_clean.to_dict('records')
        
        # Insert in batches to avoid timeout
        batch_size = 100
        for i in range(0, len(sessions_to_insert), batch_size):
            batch = sessions_to_insert[i:i + batch_size]
            response = supabase.table('sessions').upsert(batch).execute()
            print(f"✓ Inserted {len(batch)} sessions (batch {i//batch_size + 1})")
        
        return True
    except Exception as e:
        print(f"✗ Error inserting sessions: {e}")
        return False

def insert_drivers(driver_df):
    """Insert driver data into drivers table"""
    try:
        # Remove duplicates and prepare data
        driver_df_unique = driver_df.drop_duplicates(subset=['driver_number'], keep='last')
        driver_df_clean = clean_dataframe_for_json(driver_df_unique)
        drivers_to_insert = driver_df_clean.to_dict('records')
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(drivers_to_insert), batch_size):
            batch = drivers_to_insert[i:i + batch_size]
            response = supabase.table('drivers').upsert(batch).execute()
            print(f"✓ Inserted {len(batch)} drivers (batch {i//batch_size + 1})")
        
        return True
    except Exception as e:
        print(f"✗ Error inserting drivers: {e}")
        return False

def check_table_schema(table_name):
    """Check the schema of a Supabase table"""
    try:
        # Try to get one record to see the expected structure
        response = supabase.table(table_name).select('*').limit(1).execute()
        print(f"✓ Table '{table_name}' exists and is accessible")
        if response.data:
            print(f"  Sample record keys: {list(response.data[0].keys())}")
        return True
    except Exception as e:
        print(f"✗ Error checking table '{table_name}': {e}")
        return False

def insert_positions(pos_df):
    """Insert position data into positions table with enhanced error handling"""
    try:
        # Check table schema first
        print("\n🔍 Checking positions table schema...")
        if not check_table_schema('positions'):
            return False
        
        # Clean DataFrame for JSON serialization
        pos_df_clean = clean_dataframe_for_json(pos_df)
        
        print(f"Total positions to insert: {len(pos_df_clean)}")
        print(f"Columns: {list(pos_df_clean.columns)}")
        
        # Debug: Check for problematic values
        print("\n🔍 Checking for problematic values...")
        for col in pos_df_clean.columns:
            null_count = pos_df_clean[col].isnull().sum()
            unique_count = pos_df_clean[col].nunique()
            print(f"  {col}: {null_count} nulls, {unique_count} unique values")
            
            # Check data types
            print(f"    Data type: {pos_df_clean[col].dtype}")
            
            # Check for any remaining problematic values
            sample_values = pos_df_clean[col].dropna().head(3).tolist()
            print(f"    Sample values: {sample_values}")
            print(f"    Sample value types: {[type(v).__name__ for v in sample_values]}")
        
        # Convert to records with extra validation and proper type conversion
        positions_records = []
        failed_records = 0
        
        for idx, row in pos_df_clean.iterrows():
            record = {}
            for col, value in row.items():
                # Enhanced value cleaning with proper type conversion
                if value is None or pd.isna(value):
                    record[col] = None
                elif col in ['session_key', 'driver_number', 'position']:
                    try:
                        if isinstance(value, str):
                            record[col] = int(value)
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            record[col] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            if np.isnan(value) or np.isinf(value):
                                record[col] = None
                            else:
                                record[col] = int(value)
                        else:
                            record[col] = int(value)
                    except (ValueError, TypeError):
                        print(f"⚠️  Could not convert {col} value '{value}' to int")
                        record[col] = None
                elif col == 'date':
                    # Ensure date is string
                    if isinstance(value, str):
                        record[col] = value
                    else:
                        record[col] = str(value) if value is not None else None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    record[col] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    if np.isnan(value) or np.isinf(value):
                        record[col] = None
                    else:
                        record[col] = float(value)
                elif isinstance(value, np.bool_):
                    record[col] = bool(value)
                elif isinstance(value, str):
                    if value.lower() in ['nan', 'none', 'null', '']:
                        record[col] = None
                    else:
                        record[col] = value.strip()
                else:
                    # Convert any other types to string
                    try:
                        record[col] = str(value) if value is not None else None
                    except:
                        record[col] = None
            
            # Validate JSON serializability
            is_valid, error = validate_json_record(record)
            if is_valid:
                positions_records.append(record)
            else:
                failed_records += 1
                if failed_records <= 5:  # Only print first 5 failures
                    print(f"⚠️  Record {idx} failed validation: {error}")
                    print(f"    Problematic record: {record}")
        
        print(f"✓ Prepared {len(positions_records)} valid records ({failed_records} failed)")
        
        if not positions_records:
            print("✗ No valid records to insert")
            return False
        
        # Debug: Show sample of cleaned records
        print("\n🔍 Sample of cleaned records:")
        for i, sample in enumerate(positions_records[:3]):
            print(f"  Record {i+1}: {sample}")
            print(f"    Types: {[(k, type(v).__name__) for k, v in sample.items()]}")
        
        # Try a single test record first to identify the issue
        print("\n🧪 Testing single record insertion...")
        test_record = positions_records[0]
        try:
            response = supabase.table('positions').insert([test_record]).execute()
            print("✓ Single record test successful")
            
            # Clean up test record
            if response.data:
                test_id = response.data[0].get('id')
                if test_id:
                    supabase.table('positions').delete().eq('id', test_id).execute()
                    print("✓ Test record cleaned up")
            
        except Exception as test_error:
            print(f"✗ Single record test failed: {test_error}")
            print(f"  Test record: {test_record}")
            
            # Try with upsert instead of insert
            print("\n🔄 Trying upsert instead of insert...")
            try:
                response = supabase.table('positions').upsert([test_record]).execute()
                print("✓ Upsert test successful")
                use_upsert = True
            except Exception as upsert_error:
                print(f"✗ Upsert test failed: {upsert_error}")
                print("❌ Cannot insert positions data - schema or constraint issue")
                return False
        else:
            use_upsert = False
        
        # Insert in very small batches with the working method
        batch_size = 500  # Even smaller batches
        total_batches = len(positions_records) // batch_size + (1 if len(positions_records) % batch_size > 0 else 0)
        
        successful_inserts = 0
        failed_batches = 0
        
        print(f"\n📤 Starting batch insertion using {'upsert' if use_upsert else 'insert'}...")
        
        for i in range(0, len(positions_records), batch_size):
            batch = positions_records[i:i + batch_size]
            
            try:
                if use_upsert:
                    response = supabase.table('positions').upsert(batch).execute()
                else:
                    response = supabase.table('positions').insert(batch).execute()
                
                successful_inserts += len(batch)
                print(f"✓ Processed {len(batch)} positions (batch {i//batch_size + 1}/{total_batches})")
                
            except Exception as batch_error:
                failed_batches += 1
                print(f"✗ Error in batch {i//batch_size + 1}: {str(batch_error)[:200]}...")
                
                # Try individual records for failed batch
                for j, record in enumerate(batch):
                    try:
                        if use_upsert:
                            response = supabase.table('positions').upsert([record]).execute()
                        else:
                            response = supabase.table('positions').insert([record]).execute()
                        successful_inserts += 1
                    except Exception as record_error:
                        print(f"  ✗ Individual record {j+1} failed: {str(record_error)[:100]}...")
                
            time.sleep(0.3)  # Moderate delay
        
        print(f"Positions insertion complete: {successful_inserts} successful, {failed_batches} failed batches")
        return successful_inserts > 0
        
    except Exception as e:
        print(f"✗ Error inserting positions: {e}")
        import traceback
        traceback.print_exc()
        return False

def clean_dataframe_for_json(df):
    """Clean DataFrame to make it JSON serializable"""
    df_clean = df.copy()
    
    # Convert datetime columns to ISO format strings
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            # Convert datetime to string, handling nulls properly
            df_clean[col] = df_clean[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None)
    
    # Replace NaN, infinity, and invalid values with None
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    
    # Handle fillna() with explicit method
    try:
        df_clean = df_clean.fillna(value=None)
    except Exception:
        # Alternative approach if fillna still fails
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: None if pd.isna(x) else x)
    
    # Convert numpy types to Python native types
    for col in df_clean.columns:
        if str(df_clean[col].dtype) == 'int64':
            df_clean[col] = df_clean[col].astype('Int64')
        elif str(df_clean[col].dtype) == 'float64':
            df_clean[col] = df_clean[col].astype('Float64')
        elif str(df_clean[col].dtype) == 'bool':
            df_clean[col] = df_clean[col].astype('boolean')
    
    return df_clean

def insert_laps(laps_df):
    """Insert lap data into laps table with enhanced error handling"""
    try:
        laps_df_clean = clean_dataframe_for_json(laps_df)
        
        # Check if we have valid sessions first
        session_keys = laps_df_clean['session_key'].unique()
        print(f"Attempting to insert laps for {len(session_keys)} sessions")
        
        # Convert to records with validation
        laps_records = []
        failed_records = 0
        
        for idx, row in laps_df_clean.iterrows():
            record = row.to_dict()
            
            # Clean record values
            for key, value in record.items():
                if pd.isna(value) or value is None:
                    record[key] = None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
                    else:
                        record[key] = float(value)
                elif isinstance(value, np.bool_):
                    record[key] = bool(value)
                elif isinstance(value, str):
                    if value.lower() in ['nan', 'none', 'null', '']:
                        record[key] = None
                    else:
                        record[key] = value.strip()
            
            # Validate JSON serializability
            is_valid, error = validate_json_record(record)
            if is_valid:
                laps_records.append(record)
            else:
                failed_records += 1
                if failed_records <= 5:
                    print(f"⚠️  Lap record {idx} failed validation: {error}")
        
        print(f"✓ Prepared {len(laps_records)} valid lap records ({failed_records} failed)")
        
        # Insert in batches
        batch_size = 100  # Smaller batch size for safety
        for i in range(0, len(laps_records), batch_size):
            batch = laps_records[i:i + batch_size]
            response = supabase.table('laps').insert(batch).execute()
            print(f"✓ Inserted {len(batch)} laps (batch {i//batch_size + 1})")
            time.sleep(0.3)
        
        return True
    except Exception as e:
        print(f"✗ Error inserting laps: {e}")
        import traceback
        traceback.print_exc()
        return False

def insert_results(results_df):
    """Insert race results into results table with enhanced JSON handling"""
    try:
        # Remove columns that don't exist in the database table
        columns_to_keep = ['session_key', 'driver_number', 'starting_position', 'final_position', 
                          'position_change', 'laps_completed', 'dnf']
        
        # Check which columns actually exist in the DataFrame
        available_columns = [col for col in columns_to_keep if col in results_df.columns]
        results_filtered = results_df[available_columns].copy()
        
        print(f"Available columns for race results: {available_columns}")
        print(f"Total race results to insert: {len(results_filtered)}")
        
        # Enhanced cleaning for race results
        results_clean = clean_dataframe_for_json(results_filtered)
        
        # Convert to records with enhanced validation
        cleaned_results = []
        failed_records = 0
        
        for idx, row in results_clean.iterrows():
            record = {}
            for key, value in row.items():
                # Enhanced value cleaning with strict type conversion
                if value is None or pd.isna(value):
                    record[key] = None
                elif key in ['session_key', 'driver_number', 'starting_position', 'final_position', 
                           'position_change', 'laps_completed']:
                    # Integer fields
                    try:
                        if isinstance(value, str) and value.lower() in ['nan', 'none', 'null', '']:
                            record[key] = None
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            record[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            if np.isnan(value) or np.isinf(value):
                                record[key] = None
                            else:
                                record[key] = int(value)
                        else:
                            record[key] = int(float(value))
                    except (ValueError, TypeError, OverflowError):
                        print(f"⚠️  Could not convert {key} value '{value}' to int")
                        record[key] = None
                elif key == 'dnf':
                    # Boolean field
                    try:
                        if isinstance(value, str):
                            if value.lower() in ['true', '1', 'yes']:
                                record[key] = True
                            elif value.lower() in ['false', '0', 'no']:
                                record[key] = False
                            else:
                                record[key] = None
                        elif isinstance(value, (np.bool_, bool)):
                            record[key] = bool(value)
                        elif isinstance(value, (np.integer, np.floating)):
                            if pd.isna(value):
                                record[key] = None
                            else:
                                record[key] = bool(value)
                        else:
                            record[key] = bool(value) if value is not None else None
                    except (ValueError, TypeError):
                        print(f"⚠️  Could not convert {key} value '{value}' to bool")
                        record[key] = None
                else:
                    # Handle any other fields as strings
                    try:
                        if isinstance(value, str) and value.lower() in ['nan', 'none', 'null', '']:
                            record[key] = None
                        else:
                            record[key] = str(value) if value is not None else None
                    except:
                        record[key] = None
            
            # Validate JSON serializability before adding
            is_valid, error = validate_json_record(record)
            if is_valid:
                cleaned_results.append(record)
            else:
                failed_records += 1
                if failed_records <= 5:  # Only print first 5 failures
                    print(f"⚠️  Race result record {idx} failed validation: {error}")
                    print(f"    Problematic record: {record}")
        
        print(f"✓ Prepared {len(cleaned_results)} valid race result records ({failed_records} failed)")
        
        if not cleaned_results:
            print("✗ No valid race result records to insert")
            return False
        
        # Debug: Show sample of cleaned records
        print("\n🔍 Sample of cleaned race result records:")
        for i, sample in enumerate(cleaned_results[:2]):
            print(f"  Record {i+1}: {sample}")
            print(f"    Types: {[(k, type(v).__name__) for k, v in sample.items()]}")
        
        # Test single record insertion first
        print("\n🧪 Testing single race result record insertion...")
        test_record = cleaned_results[0]
        try:
            response = supabase.table('results').insert([test_record]).execute()
            print("✓ Single race result record test successful")
            
            # Clean up test record
            if response.data:
                test_id = response.data[0].get('id')
                if test_id:
                    supabase.table('results').delete().eq('id', test_id).execute()
                    print("✓ Test race result record cleaned up")
            use_upsert = False
            
        except Exception as test_error:
            print(f"✗ Single record test failed: {test_error}")
            
            # Try with upsert instead
            print("\n🔄 Trying upsert for race results...")
            try:
                response = supabase.table('results').upsert([test_record]).execute()
                print("✓ Upsert test successful")
                use_upsert = True
            except Exception as upsert_error:
                print(f"✗ Upsert test failed: {upsert_error}")
                print("❌ Cannot insert race results - schema or constraint issue")
                return False
        
        # Insert in smaller batches with better error handling
        batch_size = 500  # Very small batches to avoid JSON issues
        total_batches = len(cleaned_results) // batch_size + (1 if len(cleaned_results) % batch_size > 0 else 0)
        
        successful_inserts = 0
        failed_batches = 0
        
        print(f"\n📤 Starting race results batch insertion using {'upsert' if use_upsert else 'insert'}...")
        
        for i in range(0, len(cleaned_results), batch_size):
            batch = cleaned_results[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            # Double-check batch JSON validity
            batch_valid = True
            for j, record in enumerate(batch):
                is_valid, error = validate_json_record(record)
                if not is_valid:
                    print(f"⚠️  Batch {batch_number}, record {j+1} invalid: {error}")
                    batch_valid = False
                    break
            
            if not batch_valid:
                failed_batches += 1
                print(f"✗ Skipping batch {batch_number} due to invalid records")
                continue
            
            try:
                if use_upsert:
                    response = supabase.table('results').upsert(batch).execute()
                else:
                    response = supabase.table('results').insert(batch).execute()
                
                successful_inserts += len(batch)
                print(f"✓ Processed {len(batch)} race results (batch {batch_number}/{total_batches})")
                
            except Exception as batch_error:
                failed_batches += 1
                print(f"✗ Error in race results batch {batch_number}: {str(batch_error)[:200]}...")
                
                # Try individual records for failed batch
                individual_success = 0
                for j, record in enumerate(batch):
                    try:
                        if use_upsert:
                            response = supabase.table('results').upsert([record]).execute()
                        else:
                            response = supabase.table('results').insert([record]).execute()
                        individual_success += 1
                        successful_inserts += 1
                    except Exception as record_error:
                        print(f"  ✗ Individual race result record {j+1} failed: {str(record_error)[:100]}...")
                        print(f"    Record: {record}")
                
                print(f"  ✓ Saved {individual_success}/{len(batch)} individual records from failed batch")
            
            time.sleep(0.5)  # Longer delay for race results
        
        print(f"Race results insertion complete: {successful_inserts} successful, {failed_batches} failed batches")
        return successful_inserts > 0
        
    except Exception as e:
        print(f"✗ Error inserting race results: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def insert_weather(weather_df):
    """Insert weather data into Supabase with proper error handling"""
    if weather_df.empty:
        print("No weather data to insert.")
        return False

    try:
        # Clean the DataFrame for JSON serialization
        weather_clean = clean_dataframe_for_json(weather_df)
        weather_records = weather_clean.to_dict('records')
        
        # Validate records
        valid_records = []
        failed_records = 0
        
        for idx, record in enumerate(weather_records):
            is_valid, error = validate_json_record(record)
            if is_valid:
                valid_records.append(record)
            else:
                failed_records += 1
                if failed_records <= 3:
                    print(f"⚠️  Weather record {idx} failed validation: {error}")
        
        if not valid_records:
            print("✗ No valid weather records to insert")
            return False
        
        print(f"✓ Prepared {len(valid_records)} valid weather records ({failed_records} failed)")
        
        # Insert in batches
        batch_size = 100
        successful_inserts = 0
        
        for i in range(0, len(valid_records), batch_size):
            batch = valid_records[i:i + batch_size]
            try:
                response = supabase.table('weather').insert(batch).execute()
                successful_inserts += len(batch)
                print(f"✓ Inserted {len(batch)} weather records (batch {i//batch_size + 1})")
            except Exception as batch_error:
                print(f"✗ Error inserting weather batch {i//batch_size + 1}: {batch_error}")
                
                # Try individual records
                for j, record in enumerate(batch):
                    try:
                        response = supabase.table('weather').insert([record]).execute()
                        successful_inserts += 1
                    except Exception as record_error:
                        print(f"  ✗ Individual weather record {j+1} failed: {str(record_error)[:100]}...")
            
            time.sleep(0.2)
        
        print(f"Weather insertion complete: {successful_inserts} successful records")
        return successful_inserts > 0
        
    except Exception as e:
        print(f"✗ Error inserting weather records: {e}")
        import traceback
        traceback.print_exc()
        return False


def insert_pitstops(pitstops_df):
    """Insert pitstop data into Supabase with proper JSON serialization"""
    if pitstops_df.empty:
        print("No pitstop data to insert.")
        return False

    try:
        # Clean the DataFrame for JSON serialization
        pitstops_clean = clean_dataframe_for_json(pitstops_df)
        
        print(f"Total pitstop records to insert: {len(pitstops_clean)}")
        print(f"Pitstop columns: {list(pitstops_clean.columns)}")
        
        # Convert to records with enhanced validation
        pitstop_records = []
        failed_records = 0
        
        for idx, row in pitstops_clean.iterrows():
            record = {}
            for col, value in row.items():
                # Enhanced value cleaning with proper type conversion
                if value is None or pd.isna(value):
                    record[col] = None
                elif col in ['session_key', 'driver_number', 'lap', 'lap_number', 'stop_number']:
                    # Integer fields - including lap_number
                    try:
                        if isinstance(value, str):
                            if value.lower() in ['nan', 'none', 'null', '']:
                                record[col] = None
                            else:
                                record[col] = int(float(value))  # Convert string to int via float
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            record[col] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            if np.isnan(value) or np.isinf(value):
                                record[col] = None
                            else:
                                record[col] = int(value)
                        else:
                            record[col] = int(float(value))
                    except (ValueError, TypeError, OverflowError):
                        print(f"⚠️  Could not convert pitstop {col} value '{value}' to int")
                        record[col] = None
                elif col == 'pit_duration':
                    # Float field
                    try:
                        if isinstance(value, str):
                            if value.lower() in ['nan', 'none', 'null', '']:
                                record[col] = None
                            else:
                                record[col] = float(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            if np.isnan(value) or np.isinf(value):
                                record[col] = None
                            else:
                                record[col] = float(value)
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            record[col] = float(value)
                        else:
                            record[col] = float(value)
                    except (ValueError, TypeError, OverflowError):
                        print(f"⚠️  Could not convert pit_duration value '{value}' to float")
                        record[col] = None
                elif col in ['time', 'date']:
                    # Time/date fields - convert to string
                    try:
                        if isinstance(value, str):
                            record[col] = value
                        elif hasattr(value, 'strftime'):
                            record[col] = value.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            record[col] = str(value) if value is not None else None
                    except:
                        record[col] = str(value) if value is not None else None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    record[col] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    if np.isnan(value) or np.isinf(value):
                        record[col] = None
                    else:
                        record[col] = float(value)
                elif isinstance(value, np.bool_):
                    record[col] = bool(value)
                elif isinstance(value, str):
                    if value.lower() in ['nan', 'none', 'null', '']:
                        record[col] = None
                    else:
                        record[col] = value.strip()
                else:
                    # Convert any other types to string
                    try:
                        record[col] = str(value) if value is not None else None
                    except:
                        record[col] = None
            
            # Validate JSON serializability
            is_valid, error = validate_json_record(record)
            if is_valid:
                pitstop_records.append(record)
            else:
                failed_records += 1
                if failed_records <= 3:
                    print(f"⚠️  Pitstop record {idx} failed validation: {error}")
                    print(f"    Problematic record: {record}")
        
        print(f"✓ Prepared {len(pitstop_records)} valid pitstop records ({failed_records} failed)")
        
        if not pitstop_records:
            print("✗ No valid pitstop records to insert")
            return False
        
        # Debug: Show sample of cleaned records
        print("\n🔍 Sample of cleaned pitstop records:")
        for i, sample in enumerate(pitstop_records[:2]):
            print(f"  Record {i+1}: {sample}")
            print(f"    Types: {[(k, type(v).__name__) for k, v in sample.items()]}")
        
        # Test single record insertion first
        print("\n🧪 Testing single pitstop record insertion...")
        test_record = pitstop_records[0]
        try:
            response = supabase.table('pitstops').insert([test_record]).execute()
            print("✓ Single pitstop record test successful")
            
            # Clean up test record
            if response.data:
                test_id = response.data[0].get('id')
                if test_id:
                    supabase.table('pitstops').delete().eq('id', test_id).execute()
                    print("✓ Test pitstop record cleaned up")
            use_upsert = False
            
        except Exception as test_error:
            print(f"✗ Single pitstop record test failed: {test_error}")
            
            # Try with upsert instead
            print("\n🔄 Trying upsert for pitstops...")
            try:
                response = supabase.table('pitstops').upsert([test_record]).execute()
                print("✓ Upsert test successful")
                use_upsert = True
            except Exception as upsert_error:
                print(f"✗ Upsert test failed: {upsert_error}")
                print("❌ Cannot insert pitstops - schema or constraint issue")
                return False
        
        # Insert in very small batches
        batch_size = 25  # Very small batches to avoid JSON issues
        successful_inserts = 0
        total_batches = len(pitstop_records) // batch_size + (1 if len(pitstop_records) % batch_size > 0 else 0)
        
        for i in range(0, len(pitstop_records), batch_size):
            batch = pitstop_records[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            # Double-check batch JSON validity before sending
            batch_valid = True
            for j, record in enumerate(batch):
                is_valid, error = validate_json_record(record)
                if not is_valid:
                    print(f"⚠️  Batch {batch_number}, record {j+1} invalid: {error}")
                    batch_valid = False
                    break
            
            if not batch_valid:
                print(f"✗ Skipping batch {batch_number} due to invalid records")
                continue
            
            try:
                if use_upsert:
                    response = supabase.table('pitstops').upsert(batch).execute()
                else:
                    response = supabase.table('pitstops').insert(batch).execute()
                
                successful_inserts += len(batch)
                print(f"✓ Inserted {len(batch)} pitstop records (batch {batch_number}/{total_batches})")
                
            except Exception as batch_error:
                print(f"✗ Error inserting pitstop batch {batch_number}: {str(batch_error)[:200]}...")
                
                # Try individual records
                individual_success = 0
                for j, record in enumerate(batch):
                    try:
                        if use_upsert:
                            response = supabase.table('pitstops').upsert([record]).execute()
                        else:
                            response = supabase.table('pitstops').insert([record]).execute()
                        individual_success += 1
                        successful_inserts += 1
                    except Exception as record_error:
                        print(f"  ✗ Individual pitstop record {j+1} failed: {str(record_error)[:100]}...")
                        print(f"    Record: {record}")
                
                print(f"  ✓ Saved {individual_success}/{len(batch)} individual records from failed batch")
            
            time.sleep(0.5)  # Longer delay
        
        print(f"Pitstop insertion complete: {successful_inserts} successful records")
        return successful_inserts > 0
        
    except Exception as e:
        print(f"✗ Error inserting pitstop records: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_connection():
    """Test Supabase connection"""
    try:
        response = supabase.table('sessions').select('*').limit(1).execute()
        print("✓ Supabase connection successful!")
        return True
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return False

def check_existing_sessions(session_keys):
    """Check which sessions already exist in the database"""
    try:
        if not session_keys:
            return []
        
        # Query database for all the provided session keys
        response = supabase.table('sessions').select('session_key').in_('session_key', session_keys).execute()
        
        # Extract session_keys from response
        existing_keys = [record.get('session_key') for record in response.data]
        print(f"Found {len(existing_keys)} existing sessions in database")
        return existing_keys
    except Exception as e:
        print(f"⚠️ Error checking existing sessions: {e}")
        return []

def main():
    """Main function to extract data and insert into database"""
    print("🏁 Starting F1 Data Pipeline...")
    
    # Test connection first
    if not test_connection():
        print("Please check your Supabase configuration.")
        return
    
    # Check all table schemas
    print("\n🔍 Checking database schema...")
    tables = ['sessions', 'drivers', 'positions', 'laps', 'results']
    for table in tables:
        check_table_schema(table)
    
    # Extract all data using data.py
    years = [2025,2024,2023]
    session_names = ['Race', 'Qualifying']
    
    print("\n📊 Extracting data...")
    extracted_data = extract(years, session_names)
    
    if not extracted_data:
        print("No data extracted. Exiting.")
        return
    
    print(f"✓ Extracted {len(extracted_data['session_df'])} sessions")
    print(f"✓ Extracted {len(extracted_data['pos_df'])} positions")
    print(f"✓ Extracted {len(extracted_data['driver_df'])} drivers")
    print(f"✓ Extracted {len(extracted_data['laps_df'])} laps")
    
    # Check for existing sessions before inserting
    print("\n🔍 Checking for existing sessions in the database...")
    existing_session_keys = check_existing_sessions(extracted_data['session_df']['session_key'].tolist())
    
    # Filter out sessions that already exist
    new_sessions = extracted_data['session_df'][~extracted_data['session_df']['session_key'].isin(existing_session_keys)]
    print(f"✓ Found {len(new_sessions)} new sessions to insert")
    
    if new_sessions.empty:
        print("No new sessions to insert. Skipping session insertion.")
    else:
        print (new_sessions.head())
        new_sessions.to_csv('session_df.csv', index=False)
        
        # Insert data into database in proper order
        print("\n💾 Inserting data into Supabase...")
        
        # 1. Insert sessions first (required for foreign keys)
        if insert_sessions(new_sessions):
            print("✓ Sessions inserted successfully")
        else:
            print("✗ Failed to insert sessions. Stopping pipeline.")
            return
    
    # Filter related data to only include new sessions
    if not new_sessions.empty:
        new_session_keys = new_sessions['session_key'].tolist()
        filtered_pos_df = extracted_data['pos_df'][extracted_data['pos_df']['session_key'].isin(new_session_keys)]
        filtered_laps_df = extracted_data['laps_df'][extracted_data['laps_df']['session_key'].isin(new_session_keys)]
        filtered_weather_df = extracted_data['weather_df'][extracted_data['weather_df']['session_key'].isin(new_session_keys)] if 'weather_df' in extracted_data else pd.DataFrame()
        filtered_pit_df = extracted_data['pit_df'][extracted_data['pit_df']['session_key'].isin(new_session_keys)] if 'pit_df' in extracted_data else pd.DataFrame()
    else:
        # If all sessions already exist, skip further processing
        print("All sessions already exist in the database. Skipping further processing.")
        return
    
    # 2. Insert drivers
    if insert_drivers(extracted_data['driver_df']):
        print("✓ Drivers inserted successfully")
    
    # 3. Insert positions (depends on sessions)
    if insert_positions(filtered_pos_df):
        print("✓ Positions inserted successfully")
    else:
        print("⚠️  Positions insertion failed, but continuing with other data...")
    
    # 4. Insert laps (depends on sessions)
    if insert_laps(filtered_laps_df):
        print("✓ Laps inserted successfully")
    
    # 5. Insert weather data
    if not filtered_weather_df.empty and insert_weather(filtered_weather_df):
        print("✓ Weather data inserted successfully")
    
    # 6. Insert pitstops
    if not filtered_pit_df.empty and insert_pitstops(filtered_pit_df):
        print("✓ Pitstops inserted successfully")

    # 7. Calculate and insert race results
    print("\n🏆 Calculating race results...")
    try:
        results_df = finalize_data(
            new_sessions,
            filtered_pos_df,
            extracted_data['driver_df'],
            filtered_laps_df
        )
        
        if insert_results(results_df):
            print("✓ Race results inserted successfully")
    except Exception as e:
        print(f"✗ Error in race results calculation: {e}")
    
    print("\n🎉 F1 Data Pipeline completed!")

""""

I made a mistake while importing the race results and I did not want to import everything again so well here you go, in case you are messing around with the database and want to import more training data

years = [2025]
session_names = ['Race']
    
print("\n📊 Extracting data...")
extracted_data = extract(years, session_names)
insert_results(results_df = finalize_data(
            extracted_data['session_df'],
            extracted_data['pos_df'],
            extracted_data['driver_df'],
            extracted_data['laps_df']
        ))
        """

if __name__ == "__main__":
    main()