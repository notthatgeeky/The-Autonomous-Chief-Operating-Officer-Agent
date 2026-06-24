import pandas as pd
import sqlite3
import os

DB_NAME = "ecommerce.db"
CSV_FILE = "raw_superstore_data.csv"

def extract_data(filepath):
    """Loads the raw CSV data."""
    print(f"Extracting data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin1') # latin1 is often needed for this specific Kaggle dataset
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Please download it from Kaggle and place it in this directory.")
        return None

def transform_data(df):
    """Cleans and formats the raw DataFrame."""
    print("Transforming data...")
    
    # 1. Standardize column names (lowercase, replace spaces with underscores)
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # 2. Handle Date Formats
    # The Superstore dataset usually has 'order_date' and 'ship_date'
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%Y', errors='coerce')
    if 'ship_date' in df.columns:
        df['ship_date'] = pd.to_datetime(df['ship_date'], format='%d/%m/%Y', errors='coerce')
        
    # 3. Handle Missing Values
    # Fill missing postal codes with a placeholder (e.g., '00000')
    if 'postal_code' in df.columns:
        df['postal_code'] = df['postal_code'].fillna('00000').astype(str)
        
    # Drop rows where critical information like 'sales' or 'order_date' is missing
    df = df.dropna(subset=['sales', 'order_date'])
    
    # 4. Data Type Conversions
    # Ensure Sales and Profit are floats
    if 'sales' in df.columns:
        df['sales'] = df['sales'].astype(float)
    if 'profit' in df.columns:
        df['profit'] = df['profit'].astype(float)
        
    print(f"Transformation complete. {len(df)} rows ready for loading.")
    return df

def load_data(df, db_name):
    """Loads the cleaned DataFrame into an SQLite database."""
    print(f"Loading data into SQLite database: {db_name}...")
    
    # Create a connection to the SQLite database
    # If the file doesn't exist, sqlite3 will create it automatically
    conn = sqlite3.connect(db_name)
    
    try:
        # Write the data to a table named 'historical_sales'
        # if_exists='replace' will overwrite the table if you run the script multiple times
        df.to_sql('historical_sales', conn, if_exists='replace', index=False)
        print("Data loaded successfully into 'historical_sales' table.")
        
    except Exception as e:
        print(f"An error occurred during database loading: {e}")
        
    finally:
        # Always close the connection
        conn.close()

if __name__ == "__main__":
    print("--- Starting ETL Pipeline ---")
    
    raw_df = extract_data(CSV_FILE)
    
    if raw_df is not None:
        clean_df = transform_data(raw_df)
        load_data(clean_df, DB_NAME)
        
    print("--- Pipeline Execution Finished ---")