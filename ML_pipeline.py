import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

DB_NAME = "ecommerce.db"

def load_data_from_db():
    """Reads historical sales data from the SQLite database."""
    print("Loading data from database...")
    conn = sqlite3.connect(DB_NAME)
    
    # We only need the date and the sales amount for a time-series forecast
    query = "SELECT order_date, sales FROM historical_sales"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert string dates back to actual datetime objects
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

def prepare_features(df):
    """Aggregates data by day and creates mathematical features for the ML model."""
    print("Engineering features for the ML model...")
    
    # 1. Aggregate: Sum up all individual transactions to get total daily sales
    daily_sales = df.groupby('order_date')['sales'].sum().reset_index()
    
    # 2. Feature Engineering: ML models don't understand "dates", they understand numbers.
    # We extract the year, month, day, and day of the week as separate numerical columns.
    daily_sales['year'] = daily_sales['order_date'].dt.year
    daily_sales['month'] = daily_sales['order_date'].dt.month
    daily_sales['day'] = daily_sales['order_date'].dt.day
    daily_sales['dayofweek'] = daily_sales['order_date'].dt.dayofweek
    
    return daily_sales

def train_model(df):
    """Trains a Random Forest algorithm to predict sales based on the date features."""
    print("Training the Machine Learning model...")
    
    # X contains the features (the patterns the model learns from)
    # y contains the target (what we want to predict)
    X = df[['year', 'month', 'day', 'dayofweek']]
    y = df['sales']
    
    # Initialize and train the algorithm
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model training complete.")
    return model

def generate_future_predictions(model, last_date, days_to_predict=30):
    """Uses the trained model to predict the next 30 days of sales."""
    print(f"Generating predictions for the next {days_to_predict} days...")
    
    # Create a list of the next 30 dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_df = pd.DataFrame({'forecast_date': future_dates})
    
    # Extract the exact same features we used during training
    future_df['year'] = future_df['forecast_date'].dt.year
    future_df['month'] = future_df['forecast_date'].dt.month
    future_df['day'] = future_df['forecast_date'].dt.day
    future_df['dayofweek'] = future_df['forecast_date'].dt.dayofweek
    
    # Ask the model to predict the sales for these new dates
    predictions = model.predict(future_df[['year', 'month', 'day', 'dayofweek']])
    
    # Clean up the output DataFrame
    future_df['predicted_sales'] = predictions
    # Keep only the date and the prediction
    final_forecast = future_df[['forecast_date', 'predicted_sales']] 
    
    return final_forecast

def save_predictions_to_db(forecast_df):
    """Saves the predicted future sales into a new table in the database."""
    print("Saving predictions to the database...")
    conn = sqlite3.connect(DB_NAME)
    
    # Convert forecast_date to string before saving to SQL
    forecast_df['forecast_date'] = forecast_df['forecast_date'].astype(str)
    
    # Save to a new table. if_exists='replace' ensures we can re-run this safely.
    forecast_df.to_sql('future_predictions', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Predictions saved successfully to the 'future_predictions' table!")

if __name__ == "__main__":
    print("--- Starting ML Forecasting Pipeline ---")
    
    # Step 1: Load Data
    historical_df = load_data_from_db()
    
    if historical_df.empty:
        print("Error: The historical_sales table is empty. Run etl_pipeline.py first.")
    else:
        # Step 2: Prepare the data
        prepared_df = prepare_features(historical_df)
        
        # Step 3: Train the model
        trained_model = train_model(prepared_df)
        
        # Step 4: Predict the future (next 30 days)
        last_recorded_date = prepared_df['order_date'].max()
        forecast = generate_future_predictions(trained_model, last_recorded_date)
        
        # Step 5: Save back to Database
        save_predictions_to_db(forecast)
        
    print("--- ML Pipeline Execution Finished ---")