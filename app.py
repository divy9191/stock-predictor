import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.data_fetcher import fetch_stock_data
from models.predictions import (
    get_moving_average_prediction,
    get_linear_regression_prediction,
    get_arima_prediction,
    get_neural_network_prediction,
    get_svr_prediction
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

            # Generate predictions
            predictions = {
                "Moving Average": get_moving_average_prediction(df, prediction_days),
                "Linear Regression": get_linear_regression_prediction(df, prediction_days),
                "ARIMA": get_arima_prediction(df, prediction_days),
                "Neural Network": get_neural_network_prediction(df, prediction_days),
                "SVR": get_svr_prediction(df, prediction_days)
            }

            # Display predictions
            st.subheader("Predictions Comparison")
            pred_fig = go.Figure()
            
            # Plot historical data
            pred_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                name="Historical",
                line=dict(color='black')
            ))

            # Plot predictions
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for (name, pred), color in zip(predictions.items(), colors):
                pred_dates = [df.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
                pred_fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred,
                    name=name,
                    line=dict(color=color, dash='dash')
                ))

            st.plotly_chart(pred_fig, use_container_width=True)

            # Display metrics
            st.subheader("Model Performance Metrics")
            metrics = calculate_metrics(df, predictions)
            
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df)

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Usage Instructions:
1. Enter the stock symbol (e.g., AAPL for Apple)
2. Select the historical data period
3. Choose the number of days to predict
4. Click 'Analyze' to see predictions
""")
