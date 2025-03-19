import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
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
    """Generate predictions using basic time series prediction"""
    # Simplified version without ARIMA
    last_value = df['Close'].iloc[-1]
    trend = df['Close'].diff().mean()
    predictions = np.array([last_value + trend * i for i in range(1, prediction_days + 1)])
    return predictions

def get_neural_network_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using a simple prediction method"""
    # Simplified version without neural network
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Use last value and average change
    last_value = df['Close'].iloc[-1]
    avg_change = df['Close'].diff().mean()
    predictions = np.array([last_value + avg_change * i for i in range(1, prediction_days + 1)])
    return predictions

def get_svr_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Support Vector Regression"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    svr = SVR(kernel='rbf', C=1000.0, gamma=0.1)
    svr.fit(X, y)

    future_X = np.arange(len(df), len(df) + prediction_days).reshape(-1, 1)
    predictions = svr.predict(future_X)
    return predictions