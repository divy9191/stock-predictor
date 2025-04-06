import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance with caching
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period for historical data
    
    Returns:
        pd.DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Enhanced period options for longer historical data
        if period == "max":
            # Get maximum available data
            df = stock.history(period="max")
        else:
            # For specific periods, use start and end dates for more precision
            end_date = datetime.now()
            
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=2*365)
            elif period == "5y":
                start_date = end_date - timedelta(days=5*365)
            elif period == "10y":
                start_date = end_date - timedelta(days=10*365)
            else:
                # Default to standard period parameter for other cases
                df = stock.history(period=period)
                if df.empty:
                    st.error(f"No data found for symbol {symbol}")
                    return None
                return df
            
            df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for symbol {symbol}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
