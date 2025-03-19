import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from utils.data_fetcher import fetch_stock_data
from models.predictions import (
    get_linear_regression_prediction,
    get_random_forest_prediction,
    get_extra_trees_prediction,
    get_knn_prediction,
    get_xgboost_prediction
)
from models.evaluation import calculate_metrics

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

st.title("Stock Market Prediction App")

# Sidebar inputs
st.sidebar.header("Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
period = st.sidebar.selectbox(
    "Select Time Period",
    options=["1y", "2y", "5y"],
    index=0
)

prediction_days = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=30,
    value=7
)

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    options=[
        "Linear Regression",
        "Random Forest",
        "Extra Trees",
        "KNN",
        "XGBoost"
    ]
)

if st.sidebar.button("Analyze"):
    try:
        # Fetch data
        df = fetch_stock_data(stock_symbol, period)

        if df is not None and not df.empty:
            # Display stock info
            stock = yf.Ticker(stock_symbol)
            info = stock.info

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${info['currentPrice']:.2f}")
            with col2:
                st.metric("Market Cap", f"${info['marketCap']:,.0f}")
            with col3:
                st.metric("Volume", f"{info['volume']:,}")

            # Plot historical data
            st.subheader("Historical Price Data")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Generate prediction for selected model
            prediction_funcs = {
                "Linear Regression": get_linear_regression_prediction,
                "Random Forest": get_random_forest_prediction,
                "Extra Trees": get_extra_trees_prediction,
                "KNN": get_knn_prediction,
                "XGBoost": get_xgboost_prediction
            }

            with st.spinner(f'Generating {selected_model} predictions...'):
                predictions = prediction_funcs[selected_model](df, prediction_days)

                # Display predictions
                st.subheader(f"{selected_model} Predictions")
                pred_fig = go.Figure()

                # Plot historical data
                pred_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name="Historical",
                    line=dict(color='black')
                ))

                # Plot predictions
                pred_dates = [df.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
                pred_fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions,
                    name=f"{selected_model} Prediction",
                    line=dict(color='blue', dash='dash')
                ))

                st.plotly_chart(pred_fig, use_container_width=True)

                # Display prediction values
                st.subheader("Predicted Values")
                pred_df = pd.DataFrame({
                    'Date': pred_dates,
                    'Predicted Price': predictions
                })
                st.dataframe(pred_df.set_index('Date'))

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Usage Instructions:
1. Enter the stock symbol (e.g., AAPL for Apple)
2. Select the historical data period
3. Choose the number of days to predict
4. Select a prediction model
5. Click 'Analyze' to see predictions
""")