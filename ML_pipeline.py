import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

DB_NAME = "ecommerce.db"

def load_data_from_db():
    print("Loading data from database...")
    conn = sqlite3.connect(DB_NAME)
    
    query = "SELECT order_date, sales FROM historical_sales"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

def prepare_features(df):
    print("Engineering features for the ML model...")
    
    daily_sales = df.groupby('order_date')['sales'].sum().reset_index()
    
    daily_sales['year'] = daily_sales['order_date'].dt.year
    daily_sales['month'] = daily_sales['order_date'].dt.month
    daily_sales['day'] = daily_sales['order_date'].dt.day
    daily_sales['dayofweek'] = daily_sales['order_date'].dt.dayofweek
    
    return daily_sales

def train_model(df):
    print("Training the Machine Learning model...")
    
    X = df[['year', 'month', 'day', 'dayofweek']]
    y = df['sales']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model training complete.")
    return model

def generate_future_predictions(model, last_date, days_to_predict=30):
    print(f"Generating predictions for the next {days_to_predict} days...")
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_df = pd.DataFrame({'forecast_date': future_dates})
    
    future_df['year'] = future_df['forecast_date'].dt.year
    future_df['month'] = future_df['forecast_date'].dt.month
    future_df['day'] = future_df['forecast_date'].dt.day
    future_df['dayofweek'] = future_df['forecast_date'].dt.dayofweek
    
    predictions = model.predict(future_df[['year', 'month', 'day', 'dayofweek']])
    
    future_df['predicted_sales'] = predictions
    final_forecast = future_df[['forecast_date', 'predicted_sales']] 
    
    return final_forecast

def save_predictions_to_db(forecast_df):
    print("Saving predictions to the database...")
    conn = sqlite3.connect(DB_NAME)
    
    forecast_df['forecast_date'] = forecast_df['forecast_date'].astype(str)
    
    forecast_df.to_sql('future_predictions', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Predictions saved successfully to the 'future_predictions' table!")

if __name__ == "__main__":
    print("--- Starting ML Forecasting Pipeline ---")
    
    historical_df = load_data_from_db()
    
    if historical_df.empty:
        print("Error: The historical_sales table is empty. Run etl_pipeline.py first.")
    else:
        prepared_df = prepare_features(historical_df)
        
        trained_model = train_model(prepared_df)
        
        last_recorded_date = prepared_df['order_date'].max()
        forecast = generate_future_predictions(trained_model, last_recorded_date)
        
        save_predictions_to_db(forecast)
        
    print("--- ML Pipeline Execution Finished ---")