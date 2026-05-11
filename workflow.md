# AI Engineering Workflow for Stock Prediction

## Overview
This document outlines the industry-standard MLOps workflow for deploying and maintaining the stock price prediction model in production.

## Architecture Diagram
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│   Airbyte ETL    │────▶│   PostgreSQL    │
│   (Stock APIs)  │     │   (Ingestion)    │     │   (Raw Storage) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌────────┴────────┐
│   FastAPI       │◀────│   Airflow DAG    │◀────│   dbt           │
│   (Prediction)  │     │   (Orchestration)│     │   (Transform)  │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Streamlit     │
│   (Frontend)    │
└─────────────────┘
```

## Component Details

### 1. Data Ingestion (Airbyte)
- **Purpose**: Automated data extraction from stock APIs
- **Configuration**: 
  - Source: Alpha Vantage / Yahoo Finance / Vietnam Exchange
  - Destination: PostgreSQL / MongoDB
  - Schedule: Daily at 8:00 AM
- **Connectors**: 
  - `source-yahoo-finance` for US stocks
  - `source-vietnam-exchange` for VN stocks

### 2. Data Transformation (dbt)
- **Purpose**: Clean and engineer features from raw data
- **Transformations**:
  - Moving averages (7-day, 30-day)
  - RSI calculation
  - Bollinger Bands
  - Volume normalization
  - Missing value imputation
- **Models**:
  - `staging.stock_raw` - Raw data staging
  - `marts.stock_features` - Feature engineering

### 3. Orchestration (Apache Airflow)
- **Purpose**: Orchestrate the entire ML pipeline
- **DAG Tasks**:
  1. `extract_data` - Pull latest stock data
  2. `transform_data` - Run dbt transformations
  3. `train_model` - Retrain Conv1D model if needed
  4. `make_predictions` - Generate predictions

### 4. Model Serving (FastAPI)
- **Purpose**: REST API for real-time predictions
- **Endpoints**:
  - `POST /v1/models/stock_model:predict` - Predict next day price
  - `GET /health` - Health check
- **Model Format**: Keras `.keras` format

### 5. Web Interface (Streamlit)
- **Purpose**: User-friendly dashboard for predictions
- **Features**:
  - Historical stock data visualization
  - Next-day price prediction
  - Interactive charts
  - CSV upload for batch predictions

## Deployment Steps

### Step 1: Set up Infrastructure
```bash
# Start Airbyte
docker run -p 8000:8000 airbyte/airbyte

# Start PostgreSQL
docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres

# Start Airflow
pip install apache-airflow
airflow standalone
```

### Step 2: Configure Connections
1. Airbyte: Configure stock API sources
2. Airflow: Set up PostgreSQL connection `postgres_default`
3. dbt: Configure profile for warehouse connection

### Step 3: Deploy Model
```bash
# Copy model to deployment directory
cp stock_model.keras /path/to/models/

# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8501
```

### Step 4: Launch Frontend
```bash
# Start Streamlit
streamlit run app.py
```

## Monitoring & Maintenance

### Health Checks
- API endpoint: `GET /health`
- Airflow UI: `http://localhost:8080`
- Streamlit: `http://localhost:8501`

### Model Retraining
- Triggered manually or via Airflow schedule
- Model versioning in `/path/to/models/versions/`
- A/B testing via multiple FastAPI instances

### Logging
- FastAPI: `/var/log/stock-prediction/api.log`
- Airflow: `/var/log/airflow/dag.log`
- Streamlit: Built-in logging

## Security Considerations
- API authentication via API keys
- Rate limiting on prediction endpoint
- HTTPS in production
- Secrets management via Airflow variables or HashiCorp Vault