from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import List, Any

app = FastAPI(title="Stock Price Prediction API")

# Load the trained model globally
try:
    model = tf.keras.models.load_model("stock_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictRequest(BaseModel):
    instances: List[Any]

@app.post("/v1/models/stock_model:predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    try:
        # Convert instances to numpy array
        input_data = np.array(request.instances, dtype=np.float32)
        
        # The model expects shape (batch_size, window_size, num_features)
        # e.g., (batch_size, 30, 6)
        
        # Perform prediction
        predictions = model.predict(input_data)
        
        # Return as list of lists
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
