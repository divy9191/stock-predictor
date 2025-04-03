import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def prepare_data(df: pd.DataFrame, window: int = 60):
    """Prepare data for model training"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    X = []
    y = []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i])

    X = np.array(X).reshape(len(X), window)
    y = np.array(y)

    return X, y, scaler

def get_moving_average_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Moving Average"""
    ma_window = 20
    ma = df['Close'].rolling(window=ma_window).mean().iloc[-1]
    return np.array([ma] * prediction_days)

def get_linear_regression_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Linear Regression"""
    X, y, scaler = prepare_data(df)
    model = LinearRegression()
    model.fit(X, y)

    # Prepare last window data for prediction
    last_window = scaler.transform(df[['Close']].tail(60))
    predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(last_window[-60:].reshape(1, -1))
        predictions.append(next_pred[0])
        last_window = np.append(last_window[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

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

def get_random_forest_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Random Forest"""
    X, y, scaler = prepare_data(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_window = scaler.transform(df[['Close']].tail(60))
    predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(last_window[-60:].reshape(1, -1))
        predictions.append(next_pred[0])
        last_window = np.append(last_window[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_extra_trees_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Extra Trees"""
    X, y, scaler = prepare_data(df)
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_window = scaler.transform(df[['Close']].tail(60))
    predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(last_window[-60:].reshape(1, -1))
        predictions.append(next_pred[0])
        last_window = np.append(last_window[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_knn_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using KNN"""
    X, y, scaler = prepare_data(df)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)

    last_window = scaler.transform(df[['Close']].tail(60))
    predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(last_window[-60:].reshape(1, -1))
        predictions.append(next_pred[0])
        last_window = np.append(last_window[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_xgboost_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using XGBoost"""
    X, y, scaler = prepare_data(df)
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)

    last_window = scaler.transform(df[['Close']].tail(60))
    predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(last_window[-60:].reshape(1, -1))
        predictions.append(next_pred[0])
        last_window = np.append(last_window[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()