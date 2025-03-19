import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

def get_moving_average_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Moving Average"""
    ma_window = 20
    ma = df['Close'].rolling(window=ma_window).mean().iloc[-1]
    return np.array([ma] * prediction_days)

def get_linear_regression_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Linear Regression"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(df), len(df) + prediction_days).reshape(-1, 1)
    predictions = model.predict(future_X)
    return predictions

def get_arima_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using ARIMA"""
    model = ARIMA(df['Close'], order=(5,1,0))
    results = model.fit()
    predictions = results.forecast(steps=prediction_days)
    return predictions

def get_neural_network_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Neural Network"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    # Prepare sequences
    sequence_length = 60
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
        
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Create and train model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)
    
    # Generate predictions
    last_sequence = scaled_data[-sequence_length:]
    predictions = []
    
    for _ in range(prediction_days):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def get_svr_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Support Vector Regression"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    svr = SVR(kernel='rbf', C=1000.0, gamma=0.1)
    svr.fit(X, y)
    
    future_X = np.arange(len(df), len(df) + prediction_days).reshape(-1, 1)
    predictions = svr.predict(future_X)
    return predictions
