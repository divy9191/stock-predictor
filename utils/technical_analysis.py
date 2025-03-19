import pandas as pd
import numpy as np

def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
