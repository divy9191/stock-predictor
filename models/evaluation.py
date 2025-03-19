import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(df: pd.DataFrame, predictions: dict) -> dict:
    """
    Calculate performance metrics for each model
    
    Args:
        df (pd.DataFrame): Historical data
        predictions (dict): Dictionary of model predictions
    
    Returns:
        dict: Dictionary of performance metrics
    """
    metrics = {
        'Model': [],
        'MSE': [],
        'MAE': [],
        'R2 Score': []
    }
    
    actual = df['Close'].values[-len(predictions['Moving Average']):]
    
    for model_name, pred in predictions.items():
        metrics['Model'].append(model_name)
        metrics['MSE'].append(mean_squared_error(actual, pred))
        metrics['MAE'].append(mean_absolute_error(actual, pred))
        metrics['R2 Score'].append(r2_score(actual, pred))
    
    return metrics
