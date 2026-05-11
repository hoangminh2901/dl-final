import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

st.title("Stock Price Prediction Dashboard")

st.markdown("""
This dashboard connects to our deployed FastAPI model service to predict the next day's stock price based on the last 30 days of data.
""")

API_URL = "http://127.0.0.1:8501/v1/models/stock_model:predict"

@st.cache_data
def load_data():
    df = pd.read_csv("data/AAPL.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data()

st.subheader("Recent Stock Data (AAPL)")
st.dataframe(df.tail(10))

if st.button("Predict Next Day Price"):
    with st.spinner("Calling API..."):
        # Prepare the last 30 days as input
        feature_columns = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
        last_30_days = df[feature_columns].tail(30).values
        
        # We need to normalize the input just like the training data
        # For simplicity in this demo, we'll apply a quick min-max based on this window
        window_min = np.min(last_30_days, axis=0)
        window_max = np.max(last_30_days, axis=0)
        denom = window_max - window_min
        denom[denom == 0] = 1 # Avoid division by zero
        
        normalized_window = (last_30_days - window_min) / denom
        
        payload = {"instances": [normalized_window.tolist()]}
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                pred_norm = response.json()["predictions"][0][0]
                
                # Denormalize based on the 'Open' column (index 1)
                open_min = window_min[1]
                open_max = window_max[1]
                predicted_price = pred_norm * (open_max - open_min) + open_min
                
                st.success(f"Predicted Open Price for Next Day: ${predicted_price:.2f}")
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['Date'].tail(30), df['Open'].tail(30), label='Historical Open Price', marker='o')
                
                # Next day date
                next_day = df['Date'].iloc[-1] + pd.Timedelta(days=1)
                ax.plot([df['Date'].iloc[-1], next_day], [df['Open'].iloc[-1], predicted_price], 
                        color='red', linestyle='--', marker='x', label='Predicted Next Day')
                
                ax.set_title("Stock Price Trend")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid()
                st.pyplot(fig)
                
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
