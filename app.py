import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from utils.data_fetcher import fetch_stock_data
from utils.news_fetcher import fetch_stock_news
from utils.technical_analysis import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi
)
from models.predictions import (
    get_linear_regression_prediction,
    get_random_forest_prediction,
    get_extra_trees_prediction,
    get_knn_prediction,
    get_xgboost_prediction
)

st.set_page_config(page_title="IntelliTrade", layout="wide")
st.title("IntelliTrade - Stock Market Prediction App")

# Sidebar inputs
st.sidebar.header("Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
period = st.sidebar.selectbox(
    "Select Time Period",
    options=["1y", "2y", "5y"],
    index=0
)

# Technical Analysis Settings
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show SMA", value=True)
sma_period = st.sidebar.number_input("SMA Period", min_value=5, max_value=200, value=20)

show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
bollinger_std = st.sidebar.number_input("Bollinger Std Dev", min_value=1, max_value=4, value=2)

show_macd = st.sidebar.checkbox("Show MACD", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)

# Prediction Settings
st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=30,
    value=7
)
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
            company_name = info.get('longName', stock_symbol)

            # Create two columns for stock info and latest news
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Stock Information")
                metric1, metric2, metric3 = st.columns(3)
                with metric1:
                    st.metric("Current Price", f"${info['currentPrice']:.2f}")
                with metric2:
                    st.metric("Market Cap", f"${info['marketCap']:,.0f}")
                with metric3:
                    st.metric("Volume", f"{info['volume']:,}")

            with col2:
                st.subheader("Latest News")
                news = fetch_stock_news(stock_symbol, company_name)
                if news:
                    for article in news[:3]:
                        with st.expander(article['title']):
                            st.write(article['description'])
                            st.markdown(f"[Read more]({article['url']})")
                            st.caption(f"Source: {article['source']} | {article['publishedAt']}")

            # Technical Analysis Section
            st.subheader("Technical Analysis")

            # Create main price chart with technical indicators
            fig = go.Figure()

            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ))

            # Add SMA
            if show_sma:
                sma = calculate_sma(df, window=sma_period)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=sma,
                    name=f"SMA ({sma_period})",
                    line=dict(color='orange')
                ))

            # Add Bollinger Bands
            if show_bollinger:
                sma, upper, lower = calculate_bollinger_bands(df, window=sma_period, num_std=bollinger_std)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=upper,
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=lower,
                    name='Lower Band',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))

            st.plotly_chart(fig, use_container_width=True)

            # Add MACD and RSI in separate charts if selected
            if show_macd:
                macd, signal, histogram = calculate_macd(df)
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=macd,
                    name='MACD',
                    line=dict(color='blue')
                ))
                macd_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=signal,
                    name='Signal',
                    line=dict(color='orange')
                ))
                macd_fig.add_trace(go.Bar(
                    x=df.index,
                    y=histogram,
                    name='Histogram'
                ))
                macd_fig.update_layout(title="MACD")
                st.plotly_chart(macd_fig, use_container_width=True)

            if show_rsi:
                rsi = calculate_rsi(df)
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=rsi,
                    name='RSI',
                    line=dict(color='purple')
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.update_layout(title="RSI")
                st.plotly_chart(rsi_fig, use_container_width=True)

            # Predictions Section
            st.subheader("Price Predictions")
            prediction_funcs = {
                "Linear Regression": get_linear_regression_prediction,
                "Random Forest": get_random_forest_prediction,
                "Extra Trees": get_extra_trees_prediction,
                "KNN": get_knn_prediction,
                "XGBoost": get_xgboost_prediction
            }

            with st.spinner(f'Generating {selected_model} predictions...'):
                predictions = prediction_funcs[selected_model](df, prediction_days)
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
3. Choose technical indicators to display
4. Select prediction model and days
5. Click 'Analyze' to see analysis and predictions
""")