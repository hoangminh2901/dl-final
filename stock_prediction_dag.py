"""
Stock Prediction Pipeline DAG

This Apache Airflow DAG orchestrates the automated stock data ingestion,
transformation, model retraining, and prediction workflow.

Prerequisites:
- Airflow installed
- Required connections configured in Airflow

Usage:
    Place this file in your Airflow dags folder (e.g., ~/airflow/dags)
    and trigger via Airflow UI or CLI.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Automated stock data ingestion and prediction pipeline',
    schedule_interval='0 8 * * *',  # Run daily at 8 AM
    catchup=False,
    tags=['ml', 'stock', 'prediction'],
)

def extract_stock_data(**context):
    """
    Extract stock data from external source (e.g., Airbyte or API).
    """
    import pandas as pd
    
    # Example: Load from local CSV (replace with Airbyte/External API)
    df = pd.read_csv('/path/to/raw_data/stocks.csv')
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='raw_data', value=df.to_json())
    
    print(f"Extracted {len(df)} rows of stock data")
    return "Data extraction complete"

def transform_data(**context):
    """
    Transform raw data: feature engineering, cleaning, and scaling.
    Uses dbt or direct Python transformations.
    """
    import pandas as pd
    
    # Pull raw data from XCom
    raw_data = context['task_instance'].xcom_pull(key='raw_data', task_ids='extract_data')
    df = pd.read_json(raw_data)
    
    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Save transformed data
    df.to_csv('/path/to/processed_data/stocks_transformed.csv', index=False)
    
    print(f"Transformed {len(df)} rows with new features")
    return "Data transformation complete"

def train_model(**context):
    """
    Retrain the prediction model with latest data.
    """
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Load processed data
    df = pd.read_csv('/path/to/processed_data/stocks_transformed.csv')
    
    # Prepare features (similar to notebook)
    feature_columns = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
    window_size = 30
    
    X, y = [], []
    for i in range(1, len(df) - window_size - 1):
        X.append(df[feature_columns].iloc[i:i+window_size].values)
        y.append(df['Open'].iloc[i+window_size])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Normalize (simplified)
    X_train = X_train / X_train.max()
    y_train = y_train / y_train.max()
    
    # Build model (Conv1D)
    model = tf.keras.Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(window_size, 6), padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Save model
    model.save('/path/to/models/stock_model.keras')
    
    print("Model training complete")
    return "Model training complete"

def make_predictions(**context):
    """
    Run inference on latest data and save predictions.
    """
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    
    # Load model
    model = tf.keras.models.load_model('/path/to/models/stock_model.keras')
    
    # Load latest data
    df = pd.read_csv('/path/to/processed_data/stocks_transformed.csv')
    
    # Prepare last window
    feature_columns = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
    window_size = 30
    
    last_window = df[feature_columns].tail(window_size).values
    last_window = last_window / last_window.max()
    
    # Predict
    pred = model.predict(np.array([last_window]))
    
    # Save prediction
    result = {'date': str(df['Date'].iloc[-1]), 'predicted_price': float(pred[0][0])}
    pd.DataFrame([result]).to_csv('/path/to/output/predictions.csv', index=False)
    
    print(f"Prediction: {pred[0][0]}")
    return "Prediction complete"

# Task definitions
t1 = PythonOperator(
    task_id='extract_data',
    python_callable=extract_stock_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t4 = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag,
)

# Task dependencies
t1 >> t2 >> t3 >> t4