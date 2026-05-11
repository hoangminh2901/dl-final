import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def normalize_multifeature(X, y):
    X_norm = X.copy().astype(np.float64)
    y_norm = y.copy().astype(np.float64)
    for i in range(len(X)):
        for f in range(X.shape[2]):
            min_val = np.min(X[i, :, f])
            max_val = np.max(X[i, :, f])
            denom = max_val - min_val
            if denom == 0:
                X_norm[i, :, f] = 0
            else:
                X_norm[i, :, f] = (X[i, :, f] - min_val) / denom

        open_min = np.min(X[i, :, 1])
        open_max = np.max(X[i, :, 1])
        open_denom = open_max - open_min
        if open_denom == 0:
            y_norm[i] = 0
        else:
            y_norm[i] = (y[i] - open_min) / open_denom
    return X_norm, y_norm

def prepare_data(df, feature_columns, window_size):
    X_data = []
    y_data = []
    for i in range(1, len(df) - window_size - 1):
        data_feature = []
        for j in range(window_size):
            data_feature.append(df[feature_columns].iloc[i + j].values)
        data_label = df['Open'].iloc[i + window_size]
        X_data.append(np.array(data_feature))
        y_data.append(np.array(data_label))
    return np.array(X_data), np.array(y_data)

df = pd.read_csv('data/AAPL.csv')
feature_columns = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
window_size = 30

# Using a subset for faster training
df = df.tail(1000).reset_index(drop=True)

X, y = prepare_data(df, feature_columns, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train_norm, y_train_norm = normalize_multifeature(X_train, y_train)

num_features = len(feature_columns)

model = tf.keras.Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(window_size, num_features), padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print("Training model...")
model.fit(X_train_norm, y_train_norm, epochs=5, batch_size=32, verbose=1)

model.save("stock_model.keras")
print("Model saved to stock_model.keras")
