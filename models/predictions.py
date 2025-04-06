import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import warnings

# Suppress statsmodels warnings during forecasting
warnings.filterwarnings("ignore")

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
    """Generate predictions using ARIMA time series forecasting
    
    Args:
        df (pd.DataFrame): Historical stock data
        prediction_days (int): Number of days to predict ahead
        
    Returns:
        np.ndarray: Array of predicted prices
    """
    try:
        # Use only the 'Close' price for ARIMA modeling
        close_prices = df['Close'].astype(float)
        
        # Fit ARIMA model (p,d,q) = (5,1,0) which often works well for financial data
        # p: lag order (AR), d: degree of differencing (I), q: order of moving average (MA)
        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=prediction_days)
        predictions = forecast.values
        
        return predictions
    except Exception as e:
        # Fallback if ARIMA fails (financial data can be complex)
        st.warning(f"ARIMA model failed, using trend-based forecast: {str(e)}")
        last_value = df['Close'].iloc[-1]
        trend = df['Close'].diff().mean()
        predictions = np.array([last_value + trend * i for i in range(1, prediction_days + 1)])
        return predictions
        
def get_sarima_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Seasonal ARIMA (SARIMA) time series forecasting
    
    Args:
        df (pd.DataFrame): Historical stock data
        prediction_days (int): Number of days to predict ahead
        
    Returns:
        np.ndarray: Array of predicted prices
    """
    try:
        # Use only the 'Close' price for SARIMA modeling
        close_prices = df['Close'].astype(float)
        
        # SARIMA model with parameters (p,d,q)x(P,D,Q,s)
        # For financial data with weekly seasonality (s=5 for trading days)
        model = SARIMAX(close_prices, 
                       order=(1, 1, 1),               # Non-seasonal part (p,d,q)
                       seasonal_order=(1, 1, 1, 5))   # Seasonal part (P,D,Q,s)
        model_fit = model.fit(disp=False)
        
        # Generate forecast
        forecast = model_fit.forecast(steps=prediction_days)
        predictions = forecast.values
        
        return predictions
    except Exception as e:
        # Fallback if SARIMA fails
        st.warning(f"SARIMA model failed, using trend-based forecast: {str(e)}")
        return get_arima_prediction(df, prediction_days)
        
def get_exponential_smoothing_prediction(df: pd.DataFrame, prediction_days: int) -> np.ndarray:
    """Generate predictions using Holt-Winters Exponential Smoothing
    
    Args:
        df (pd.DataFrame): Historical stock data
        prediction_days (int): Number of days to predict ahead
        
    Returns:
        np.ndarray: Array of predicted prices
    """
    try:
        # Use only the 'Close' price for Exponential Smoothing
        close_prices = df['Close'].astype(float)
        
        # Holt-Winters model for time series with trend and seasonality
        model = ExponentialSmoothing(
            close_prices, 
            trend='add',           # Additive trend
            seasonal='add',        # Additive seasonality
            seasonal_periods=5     # Weekly seasonality (5 trading days)
        )
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(prediction_days)
        predictions = forecast.values
        
        return predictions
    except Exception as e:
        # Fallback if exponential smoothing fails
        st.warning(f"Exponential Smoothing model failed, using trend-based forecast: {str(e)}")
        return get_arima_prediction(df, prediction_days)

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