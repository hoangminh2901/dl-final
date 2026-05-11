# Stock Price Prediction Project

A deep learning project for time-series stock price prediction using CNN (Conv1D) models, deployed as a REST API with a Streamlit frontend.

## Project Structure

```
final-project/
├── data/                      # Stock data
│   ├── AAPL.csv              # Apple stock data
│   ├── ACB-VNINDEX-History.csv
│   └── companies.csv
├── stock_model.keras         # Trained model
├── train_model.py            # Model training script
├── api.py                    # FastAPI server
├── app.py                    # Streamlit dashboard
├── stock_prediction_dag.py  # Airflow DAG
├── workflow.md               # MLOps workflow documentation
└── README.md
```

## Quick Start

### 1. Train Model
```bash
cd final-project
source .venv/bin/activate
python train_model.py
```
This saves `stock_model.keras`.

### 2. Start FastAPI (Backend)
```bash
uvicorn api:app --host 0.0.0.0 --port 8501
```
- API: http://localhost:8501
- Health: http://localhost:8501/health

### 3. Start Streamlit (Frontend)
```bash
streamlit run app.py
```
- Dashboard: http://localhost:8501

## API Usage

**Endpoint**: `POST /v1/models/stock_model:predict`

```json
{
  "instances": [[[0.1, 0.2, 0.3, 0.15, 0.14, 0.13], ...]]  // 30 timesteps x 6 features
}
```

Response:
```json
{
  "predictions": [[0.85]]
}
```

## Task 5: Deployment Components

### Task 5.1: Model Deployment (REST API)
- **FastAPI** (`api.py`) serves the trained model
- **Endpoint**: `/v1/models/stock_model:predict`
- See: `workflow.md` for production deployment

### Task 5.2: Model as SaaS (Web Interface)
- **Streamlit** (`app.py`) provides user-friendly dashboard
- Features: historical data, prediction, visualization

### Task 5.3: AI Engineering Workflow
- **Apache Airflow** DAG (`stock_prediction_dag.py`)
- **dbt** for data transformations
- **Airbyte** for data ingestion
- Full workflow documented in `workflow.md`

## Dependencies

```
tensorflow
fastapi
uvicorn
streamlit
pandas
numpy
scikit-learn
apache-airflow
```

## Data Format

Input features: `['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']`
Window size: 30 days
Output: Predicted next-day Open price

## Notes

- Model trained on last 1000 days of AAPL data
- Uses Conv1D architecture for time-series
- Normalization applied per-window