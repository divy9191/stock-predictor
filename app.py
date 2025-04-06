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
    get_xgboost_prediction,
    get_arima_prediction,
    get_sarima_prediction,
    get_exponential_smoothing_prediction
)

st.set_page_config(
    page_title="IntelliTrade", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# IntelliTrade\nA comprehensive stock prediction platform powered by multiple machine learning models."
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f2f6;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .model-button {
        margin-top: 0.5rem;
        width: 100%;
    }
    .watchlist-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>IntelliTrade</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Advanced Stock Market Prediction Platform</p>", unsafe_allow_html=True)

# Initialize session state variables
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Handle redirection from watchlist if needed
if 'stock_for_analysis' in st.session_state and 'mode_for_redirect' in st.session_state:
    # Save the redirection parameters (will be used later in the app logic)
    redirect_symbol = st.session_state.stock_for_analysis
    redirect_mode = st.session_state.mode_for_redirect
    # Clear the session state to prevent infinite redirects
    del st.session_state.stock_for_analysis
    del st.session_state.mode_for_redirect
else:
    redirect_symbol = None
    redirect_mode = None

# Sidebar organization
with st.sidebar:
    st.markdown("<div class='sidebar-header'>📊 IntelliTrade</div>", unsafe_allow_html=True)
    
    # Navigation section
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Mode",
        options=["Stock Analysis", "Model Comparison", "Watchlist"],
        key="app_mode_radio",
        label_visibility="collapsed"
    )
    
    # Create an expander for stock selection
    with st.expander("🔍 Stock Selection", expanded=True):
        stock_symbol = st.text_input("Enter Stock Symbol", value="AAPL")
        period = st.selectbox(
            "Historical Data Period",
            options=["1y", "2y", "5y", "10y", "max"],
            index=0,
            help="Select the period of historical data to analyze. 'max' retrieves all available data."
        )
    
    # Create an expander for technical indicators
    with st.expander("📈 Technical Indicators", expanded=app_mode == "Stock Analysis"):
        show_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
        if show_sma:
            sma_period = st.number_input("SMA Period", min_value=5, max_value=200, value=20)
        
        show_bollinger = st.checkbox("Bollinger Bands", value=False)
        if show_bollinger:
            bollinger_std = st.number_input("Bollinger Std Dev", min_value=1, max_value=4, value=2)
        
        col1, col2 = st.columns(2)
        with col1:
            show_macd = st.checkbox("MACD", value=False)
        with col2:
            show_rsi = st.checkbox("RSI", value=False)
    
    # Prediction settings in an expander
    with st.expander("🔮 Prediction Settings", expanded=app_mode != "Watchlist"):
        prediction_days = st.slider(
            "Forecast Horizon (Days)",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # Initialize model selection variables
        selected_model = "Linear Regression"  # Default value
        selected_models = ["Linear Regression", "Random Forest", "XGBoost"]  # Default values
        
        # Model selection based on mode
        if app_mode == "Stock Analysis":
            selected_model = st.selectbox(
                "Prediction Model",
                options=[
                    "Linear Regression",
                    "Random Forest", 
                    "Extra Trees",
                    "KNN",
                    "XGBoost",
                    "ARIMA",
                    "SARIMA",
                    "Exponential Smoothing"
                ]
            )
        elif app_mode == "Model Comparison":
            selected_models = st.multiselect(
                "Models to Compare",
                options=[
                    "Linear Regression",
                    "Random Forest",
                    "Extra Trees", 
                    "KNN",
                    "XGBoost",
                    "ARIMA",
                    "SARIMA",
                    "Exponential Smoothing"
                ],
                default=["Linear Regression", "Random Forest", "XGBoost"]
            )
    
    # Watchlist management
    if app_mode == "Watchlist":
        with st.expander("➕ Manage Watchlist", expanded=True):
            new_stock = st.text_input("Add Stock Symbol")
            if st.button("➕ Add to Watchlist", use_container_width=True) and new_stock:
                if new_stock not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_stock)
                    st.success(f"Added {new_stock} to watchlist!")
                else:
                    st.info(f"{new_stock} is already in your watchlist")
                    
            if st.session_state.watchlist and st.button("🗑️ Clear Watchlist", use_container_width=True):
                st.session_state.watchlist = []
                st.info("Watchlist cleared")
    
    # Action button at the bottom
    st.markdown("### 🚀 Actions")
    if app_mode == "Stock Analysis":
        analyze_button = st.button("Analyze Stock", use_container_width=True, type="primary")
    elif app_mode == "Model Comparison":
        analyze_button = st.button("Compare Models", use_container_width=True, type="primary")
    else:
        analyze_button = False

# Import required libraries for PDF
import io
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64

# Function to create a PDF report
def create_pdf_report(stock_symbol, company_name, current_price, predictions, model_name, period):
    buffer = io.BytesIO()
    
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, f"IntelliTrade Stock Analysis Report: {company_name} ({stock_symbol})")
    
    # General information
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 80, "Stock Information")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 100, f"Current Price: ${current_price:.2f}")
    pdf.drawString(50, height - 120, f"Analysis Period: {period}")
    pdf.drawString(50, height - 140, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Prediction information
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 180, f"Price Predictions ({model_name})")
    
    y_pos = height - 200
    pdf.setFont("Helvetica", 10)
    for i, (date, price) in enumerate(zip(predictions["Date"], predictions["Predicted Price"])):
        date_str = date.strftime("%Y-%m-%d")
        pdf.drawString(50, y_pos - (i * 20), f"{date_str}: ${price:.2f}")
    
    # Add footer
    pdf.setFont("Helvetica-Italic", 8)
    pdf.drawString(50, 30, "Generated by IntelliTrade Stock Prediction App")
    pdf.drawString(50, 20, "Disclaimer: This report is for informational purposes only and should not be considered investment advice.")
    
    pdf.save()
    buffer.seek(0)
    return buffer

# Apply any redirection from watchlist
if redirect_symbol is not None and redirect_mode is not None:
    stock_symbol = redirect_symbol
    app_mode = redirect_mode

if app_mode == "Stock Analysis":
    # Display description and instructional tab
    with st.expander("ℹ️ About Stock Analysis Mode", expanded=False):
        st.markdown("""
        **In this mode, you can:**
        * Analyze a single stock with technical indicators
        * Generate price predictions using your selected model
        * Export the analysis as a PDF report
        
        Use the sidebar to select your stock and configure the analysis settings.
        """)
    
    # Show placeholder content before analysis
    if not 'analyze_button' in locals() or not analyze_button:
        st.markdown("<h2 class='sub-header'>Stock Analysis Dashboard</h2>", unsafe_allow_html=True)
        st.info("👈 Enter a stock symbol and click 'Analyze Stock' in the sidebar to begin")
        
        # Placeholder for the chart area
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("""
            ### How to use this feature:
            1. **Enter a stock symbol** in the sidebar (e.g., AAPL, MSFT, GOOGL)
            2. **Choose the time period** for historical data analysis
            3. **Toggle technical indicators** you want to see
            4. **Select a prediction model** for price forecasting
            5. **Click the 'Analyze Stock' button** to view your analysis
            """)
            
    # When analyze button is clicked
    if 'analyze_button' in locals() and analyze_button:
        try:
            with st.spinner("Fetching stock data and generating analysis..."):
                # Fetch data
                df = fetch_stock_data(stock_symbol, period)
                if df is not None and not df.empty:
                    # Display stock info
                    stock = yf.Ticker(stock_symbol)
                    info = stock.info
                    company_name = info.get('longName', stock_symbol)
                    
                    # Stock Header Section
                    st.markdown(f"<h2 class='sub-header'>{company_name} ({stock_symbol})</h2>", unsafe_allow_html=True)
                    
                    # Stock Overview Card
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # Create three columns for stock metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Current Price", f"${info['currentPrice']:.2f}", 
                                 f"{info.get('regularMarketChangePercent', 0):.2f}%")
                    with metric_cols[1]:
                        st.metric("Market Cap", f"${info['marketCap']:,.0f}")
                    with metric_cols[2]:
                        st.metric("Volume", f"{info['volume']:,}")
                    with metric_cols[3]:
                        st.metric("Analysis Period", period)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🔮 Predictions", "📰 News"])
                    
                    with tab1:
                        # Main Price Chart
                        st.markdown("<h3>Price History</h3>", unsafe_allow_html=True)
                        
                        # Create main price chart
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"
                        ))
                        
                        # Layout improvements
                        fig.update_layout(
                            title=f"{company_name} Stock Price - {period}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            template="plotly_white",
                            height=500,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Technical Analysis Section
                        st.markdown("<h3>Technical Indicators</h3>", unsafe_allow_html=True)
                        
                        # Create technical analysis chart
                        tech_fig = go.Figure()
                        
                        # Add candlestick chart as base
                        tech_fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"
                        ))
                        
                        # Add selected technical indicators
                        # Add SMA
                        if show_sma:
                            sma = calculate_sma(df, window=sma_period)
                            tech_fig.add_trace(go.Scatter(
                                x=df.index,
                                y=sma,
                                name=f"SMA ({sma_period})",
                                line=dict(color='orange')
                            ))
                        
                        # Add Bollinger Bands
                        if show_bollinger:
                            sma, upper, lower = calculate_bollinger_bands(df, window=sma_period, num_std=bollinger_std)
                            tech_fig.add_trace(go.Scatter(
                                x=df.index,
                                y=upper,
                                name='Upper Band',
                                line=dict(color='gray', dash='dash')
                            ))
                            tech_fig.add_trace(go.Scatter(
                                x=df.index,
                                y=lower,
                                name='Lower Band',
                                line=dict(color='gray', dash='dash'),
                                fill='tonexty'
                            ))
                        
                        # Layout improvements
                        tech_fig.update_layout(
                            title=f"Technical Analysis - {stock_symbol}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            template="plotly_white",
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(tech_fig, use_container_width=True)
                        
                        # Add MACD and RSI in separate charts if selected
                        indicator_cols = st.columns(2)
                        
                        with indicator_cols[0]:
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
                                macd_fig.update_layout(
                                    title="MACD",
                                    height=300,
                                    template="plotly_white"
                                )
                                st.plotly_chart(macd_fig, use_container_width=True)
                            else:
                                st.info("Enable MACD in the sidebar to view this indicator")
                        
                        with indicator_cols[1]:
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
                                rsi_fig.update_layout(
                                    title="RSI",
                                    height=300,
                                    template="plotly_white"
                                )
                                st.plotly_chart(rsi_fig, use_container_width=True)
                            else:
                                st.info("Enable RSI in the sidebar to view this indicator")
                    
                    with tab3:
                        # Predictions Section
                        st.markdown("<h3>Price Predictions</h3>", unsafe_allow_html=True)
                        
                        # Set up prediction functions
                        prediction_funcs = {
                            "Linear Regression": get_linear_regression_prediction,
                            "Random Forest": get_random_forest_prediction,
                            "Extra Trees": get_extra_trees_prediction,
                            "KNN": get_knn_prediction,
                            "XGBoost": get_xgboost_prediction,
                            "ARIMA": get_arima_prediction,
                            "SARIMA": get_sarima_prediction,
                            "Exponential Smoothing": get_exponential_smoothing_prediction
                        }
                        
                        with st.spinner(f'Generating {selected_model} predictions...'):
                            # Calculate predictions
                            predictions = prediction_funcs[selected_model](df, prediction_days)
                            pred_dates = [df.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
                            
                            # Create prediction dataframe
                            pred_df = pd.DataFrame({
                                'Date': pred_dates,
                                'Predicted Price': predictions
                            })
                            
                            # Display model information card
                            st.markdown(f"""
                            <div class='card'>
                                <h4>Model: {selected_model}</h4>
                                <p>Forecast horizon: {prediction_days} days</p>
                                <p>Training data period: {period}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create prediction chart
                            pred_fig = go.Figure()
                            
                            # Plot historical data (last 30 days for better visibility)
                            pred_fig.add_trace(go.Scatter(
                                x=df.index[-30:],
                                y=df['Close'][-30:],
                                name="Historical",
                                line=dict(color='black')
                            ))
                            
                            # Plot predictions
                            pred_fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=predictions,
                                name=f"{selected_model} Prediction",
                                line=dict(color='blue', dash='dash')
                            ))
                            
                            # Layout improvements
                            pred_fig.update_layout(
                                title=f"{stock_symbol} Price Prediction - {selected_model}",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                template="plotly_white",
                                height=400
                            )
                            
                            st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Display prediction values
                            st.subheader("Predicted Values")
                            
                            # Format the table with styling
                            st.dataframe(
                                pred_df.set_index('Date').style.format({
                                    'Predicted Price': '${:.2f}'
                                }),
                                use_container_width=True
                            )
                            
                            # Generate PDF Report
                            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                            st.subheader("Export Analysis")
                            
                            if st.button("📄 Generate PDF Report", use_container_width=True):
                                with st.spinner("Creating PDF report..."):
                                    pdf_buffer = create_pdf_report(
                                        stock_symbol, 
                                        company_name, 
                                        info['currentPrice'], 
                                        pred_df, 
                                        selected_model,
                                        period
                                    )
                                    
                                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                                    href = f'<a href="data:application/pdf;base64,{b64}" download="{stock_symbol}_analysis.pdf" style="text-decoration:none;"><button style="padding:10px 20px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; width:100%;">Download PDF Report</button></a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                    st.success("PDF report generated successfully!")
                    
                    with tab4:
                        # News Section
                        st.markdown("<h3>Latest News</h3>", unsafe_allow_html=True)
                        
                        # Fetch news
                        news = fetch_stock_news(stock_symbol, company_name)
                        
                        if news:
                            for i, article in enumerate(news[:5]):  # Show 5 news articles
                                with st.expander(f"{i+1}. {article['title']}", expanded=i==0):
                                    st.markdown(f"**{article['description']}**")
                                    st.markdown(f"[Read full article]({article['url']})")
                                    st.caption(f"Source: {article['source']} | {article['publishedAt']}")
                        else:
                            st.info("No recent news found for this stock")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            st.info("Please check your stock symbol and try again")

elif app_mode == "Model Comparison":
    # Display description and instructional tab
    with st.expander("ℹ️ About Model Comparison Mode", expanded=False):
        st.markdown("""
        **In this mode, you can:**
        * Compare predictions from multiple models side-by-side
        * See how different algorithms forecast the same stock
        * Export a comprehensive comparison report
        
        Use the sidebar to select multiple models (at least 2) for comparison.
        """)
    
    # Show placeholder content before analysis
    if not 'analyze_button' in locals() or not analyze_button:
        st.markdown("<h2 class='sub-header'>Model Comparison Dashboard</h2>", unsafe_allow_html=True)
        st.info("👈 Select at least 2 models in the sidebar and click 'Compare Models' to begin")
        
        # Placeholder for the chart area
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("""
            ### How to use this feature:
            1. **Enter a stock symbol** in the sidebar
            2. **Select multiple prediction models** (at least 2)
            3. **Choose the forecast horizon** (prediction days)
            4. **Click the 'Compare Models' button** to run the comparison
            """)
    
    # When compare button is clicked
    if 'analyze_button' in locals() and analyze_button:
        if len(selected_models) < 2:
            st.warning("⚠️ Please select at least 2 models to compare in the sidebar.")
        else:
            try:
                with st.spinner("Generating multi-model comparison..."):
                    # Fetch data
                    df = fetch_stock_data(stock_symbol, period)
                    if df is not None and not df.empty:
                        stock = yf.Ticker(stock_symbol)
                        info = stock.info
                        company_name = info.get('longName', stock_symbol)
                        
                        # Stock Header Section
                        st.markdown(f"<h2 class='sub-header'>Model Comparison: {company_name} ({stock_symbol})</h2>", unsafe_allow_html=True)
                        
                        # Stock Overview Card
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        
                        # Create columns for stock metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Current Price", f"${info['currentPrice']:.2f}", 
                                     f"{info.get('regularMarketChangePercent', 0):.2f}%")
                        with metric_cols[1]:
                            st.metric("Models Compared", f"{len(selected_models)}")
                        with metric_cols[2]:
                            st.metric("Forecast Days", f"{prediction_days}")
                        with metric_cols[3]:
                            st.metric("Data Period", period)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Create tabs for different comparison views
                        tab1, tab2, tab3 = st.tabs(["📊 Comparison Chart", "📋 Prediction Table", "🏆 Model Performance"])
                        
                        # Set up prediction functions
                        prediction_funcs = {
                            "Linear Regression": get_linear_regression_prediction,
                            "Random Forest": get_random_forest_prediction,
                            "Extra Trees": get_extra_trees_prediction,
                            "KNN": get_knn_prediction,
                            "XGBoost": get_xgboost_prediction,
                            "ARIMA": get_arima_prediction,
                            "SARIMA": get_sarima_prediction,
                            "Exponential Smoothing": get_exponential_smoothing_prediction
                        }
                        
                        # Calculate predictions for each model
                        with st.spinner(f'Generating predictions for {len(selected_models)} models...'):
                            pred_dates = [df.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
                            
                            # Create a DataFrame to hold all predictions
                            all_predictions = pd.DataFrame({'Date': pred_dates})
                            all_predictions.set_index('Date', inplace=True)
                            
                            # Colors for consistent model representation
                            model_colors = {
                                "Linear Regression": "#1f77b4",  # blue
                                "Random Forest": "#ff7f0e",      # orange
                                "Extra Trees": "#2ca02c",        # green
                                "KNN": "#d62728",                # red
                                "XGBoost": "#9467bd",            # purple
                                "ARIMA": "#8c564b",              # brown
                                "SARIMA": "#e377c2",             # pink
                                "Exponential Smoothing": "#7f7f7f" # gray
                            }
                            
                            # Generate predictions for each model
                            for model_name in selected_models:
                                model_preds = prediction_funcs[model_name](df, prediction_days)
                                all_predictions[model_name] = model_preds
                            
                            # Calculate average prediction
                            all_predictions['Consensus'] = all_predictions.mean(axis=1)
                            
                            with tab1:
                                st.markdown("<h3>Prediction Comparison</h3>", unsafe_allow_html=True)
                                
                                # Create comparison chart
                                comp_fig = go.Figure()
                                
                                # Add historical data (last 30 days for better visibility)
                                comp_fig.add_trace(go.Scatter(
                                    x=df.index[-30:],
                                    y=df['Close'][-30:],
                                    name="Historical",
                                    line=dict(color='black', width=2)
                                ))
                                
                                # Add predictions from each model
                                for model_name in selected_models:
                                    comp_fig.add_trace(go.Scatter(
                                        x=pred_dates,
                                        y=all_predictions[model_name],
                                        name=model_name,
                                        line=dict(
                                            color=model_colors.get(model_name, "#636EFA"),
                                            dash='dash'
                                        )
                                    ))
                                
                                # Add consensus line
                                comp_fig.add_trace(go.Scatter(
                                    x=pred_dates,
                                    y=all_predictions['Consensus'],
                                    name="Consensus (Average)",
                                    line=dict(color='#17BECF', width=3)
                                ))
                                
                                # Improve layout
                                comp_fig.update_layout(
                                    title=f"{stock_symbol} - Multi-Model Prediction Comparison",
                                    xaxis_title="Date",
                                    yaxis_title="Price (USD)",
                                    template="plotly_white",
                                    height=500,
                                    legend=dict(
                                        orientation="h",
                                        y=1.02,
                                        x=0.5,
                                        xanchor="center"
                                    )
                                )
                                
                                st.plotly_chart(comp_fig, use_container_width=True)
                                
                                # Show the range of predictions
                                final_day = all_predictions.index[-1]
                                min_pred = all_predictions.loc[final_day].min()
                                max_pred = all_predictions.loc[final_day].max()
                                consensus = all_predictions['Consensus'].loc[final_day]
                                
                                st.markdown(f"""
                                <div class='card'>
                                    <h4>Prediction Range ({final_day.strftime('%Y-%m-%d')})</h4>
                                    <p>Min: <strong>${min_pred:.2f}</strong> | Max: <strong>${max_pred:.2f}</strong> | Consensus: <strong>${consensus:.2f}</strong></p>
                                    <p>Spread: <strong>${max_pred - min_pred:.2f}</strong> ({((max_pred - min_pred) / consensus * 100):.1f}% of consensus)</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with tab2:
                                st.markdown("<h3>Prediction Data</h3>", unsafe_allow_html=True)
                                
                                # Format the table with styling
                                st.dataframe(
                                    all_predictions.style.format("${:.2f}"),
                                    use_container_width=True
                                )
                                
                                # Download as CSV option
                                csv_data = all_predictions.reset_index().to_csv(index=False)
                                b64_csv = base64.b64encode(csv_data.encode()).decode()
                                href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="{stock_symbol}_predictions.csv"><button style="padding:5px 15px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">Download Predictions CSV</button></a>'
                                st.markdown(href_csv, unsafe_allow_html=True)
                            
                            with tab3:
                                st.markdown("<h3>Final Predictions Summary</h3>", unsafe_allow_html=True)
                                
                                # Show statistics for the final day prediction
                                model_stats = []
                                for model in selected_models + ["Consensus"]:
                                    final_pred = all_predictions[model].iloc[-1]
                                    change = ((final_pred / info['currentPrice']) - 1) * 100
                                    model_stats.append({
                                        "Model": model,
                                        "Final Price": final_pred,
                                        "Change %": change
                                    })
                                
                                stats_df = pd.DataFrame(model_stats)
                                
                                # Create chart to visualize final predictions
                                stats_fig = go.Figure()
                                for i, row in stats_df.iterrows():
                                    model = row["Model"]
                                    color = "#17BECF" if model == "Consensus" else model_colors.get(model, "#636EFA")
                                    width = 3 if model == "Consensus" else 1.5
                                    
                                    stats_fig.add_trace(go.Bar(
                                        x=[model],
                                        y=[row["Final Price"]],
                                        name=model,
                                        marker_color=color,
                                        width=0.6,
                                        text=f"${row['Final Price']:.2f}<br>({row['Change %']:.2f}%)",
                                        textposition="outside"
                                    ))
                                
                                stats_fig.update_layout(
                                    title=f"Final Day Price Prediction ({pred_dates[-1].strftime('%Y-%m-%d')})",
                                    xaxis_title="Model",
                                    yaxis_title="Price (USD)",
                                    template="plotly_white",
                                    height=400,
                                    showlegend=False
                                )
                                
                                # Add current price reference line
                                stats_fig.add_hline(
                                    y=info['currentPrice'],
                                    line_dash="dot",
                                    line_color="red",
                                    annotation_text=f"Current: ${info['currentPrice']:.2f}"
                                )
                                
                                st.plotly_chart(stats_fig, use_container_width=True)
                                
                                # Display model comparison table
                                st.subheader("Model Comparison Table")
                                
                                # Format the table with styling
                                st.dataframe(
                                    stats_df.style.format({
                                        "Final Price": "${:.2f}",
                                        "Change %": "{:.2f}%"
                                    }).background_gradient(subset=["Change %"], cmap="RdYlGn"),
                                    use_container_width=True
                                )
                            
                            # Generate PDF Report for comparison
                            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                            st.subheader("Export Comparison")
                            
                            if st.button("📄 Generate Comparison Report", use_container_width=True):
                                with st.spinner("Creating comparison report..."):
                                    # For now, we'll use a simple version
                                    pdf_buffer = create_pdf_report(
                                        stock_symbol, 
                                        company_name, 
                                        info['currentPrice'], 
                                        all_predictions.reset_index(), 
                                        "Multiple Models",
                                        period
                                    )
                                    
                                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                                    href = f'<a href="data:application/pdf;base64,{b64}" download="{stock_symbol}_comparison.pdf" style="text-decoration:none;"><button style="padding:10px 20px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; width:100%;">Download Comparison PDF</button></a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                    st.success("PDF comparison report generated successfully!")
                    
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.info("Please check your stock symbol and try again")

elif app_mode == "Watchlist":
    # Display description and instructional tab
    with st.expander("ℹ️ About Watchlist Mode", expanded=False):
        st.markdown("""
        **In this mode, you can:**
        * Keep track of multiple stocks in one view
        * See quick price and performance metrics
        * Quickly navigate to detailed analysis of any stock
        
        Use the sidebar to add stocks to your watchlist.
        """)
    
    # Main watchlist display
    st.markdown("<h2 class='sub-header'>Your Stock Watchlist</h2>", unsafe_allow_html=True)
    
    if not st.session_state.watchlist:
        st.info("👈 Your watchlist is empty. Add stocks using the sidebar.")
    else:
        # Add sorting options
        sort_col, _ = st.columns([1, 3])
        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                options=["Symbol", "Price", "Daily Change"],
                index=0
            )
        
        # Display header statistics
        total_stocks = len(st.session_state.watchlist)
        st.markdown(f"<div class='card'><p>Tracking {total_stocks} stocks in your watchlist</p></div>", unsafe_allow_html=True)
        
        # Prepare watchlist data
        watchlist_data = []
        with st.spinner("Loading watchlist data..."):
            for symbol in st.session_state.watchlist:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    price = info.get('currentPrice', 0)
                    change = info.get('regularMarketChangePercent', 0)
                    
                    watchlist_data.append({
                        "symbol": symbol,
                        "name": info.get('longName', symbol),
                        "price": price,
                        "change": change,
                        "sector": info.get('sector', 'N/A'),
                        "volume": info.get('volume', 0)
                    })
                except Exception as e:
                    watchlist_data.append({
                        "symbol": symbol,
                        "name": symbol,
                        "price": 0,
                        "change": 0,
                        "sector": "Error",
                        "volume": 0,
                        "error": str(e)
                    })
        
        # Sort the watchlist
        if sort_by == "Symbol":
            watchlist_data.sort(key=lambda x: x["symbol"])
        elif sort_by == "Price":
            watchlist_data.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "Daily Change":
            watchlist_data.sort(key=lambda x: x["change"], reverse=True)
        
        # Display stocks in a grid
        num_cols = 3
        rows = [watchlist_data[i:i+num_cols] for i in range(0, len(watchlist_data), num_cols)]
        
        for row in rows:
            cols = st.columns(num_cols)
            for i, stock_data in enumerate(row):
                with cols[i]:
                    # Determine card border color based on price change
                    if stock_data["change"] > 0:
                        border_color = "#4CAF50"  # Green for positive
                    elif stock_data["change"] < 0:
                        border_color = "#F44336"  # Red for negative
                    else:
                        border_color = "#9E9E9E"  # Grey for neutral
                    
                    # Stock card with dynamic styling
                    st.markdown(f"""
                    <div style="border-left: 4px solid {border_color}; background-color: #f8f9fa; border-radius: 5px; padding: 1rem; margin-bottom: 1rem;">
                        <h3 style="margin-top: 0;">{stock_data['symbol']}</h3>
                        <p style="color: #555; margin-bottom: 10px;">{stock_data['name']}</p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.2rem; font-weight: bold;">${stock_data['price']:.2f}</span>
                            <span style="color: {'#4CAF50' if stock_data['change'] > 0 else '#F44336' if stock_data['change'] < 0 else '#9E9E9E'}; font-weight: bold;">
                                {stock_data['change']:.2f}%
                            </span>
                        </div>
                        <p style="color: #555; margin-top: 10px; font-size: 0.9rem;">Sector: {stock_data['sector']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📊 Analyze", key=f"analyze_{stock_data['symbol']}"):
                            # Set the stock symbol and redirect to analysis page
                            st.session_state.stock_for_analysis = stock_data['symbol']
                            st.session_state.mode_for_redirect = "Stock Analysis"
                            st.rerun()
                    with col2:
                        if st.button(f"🗑 Remove", key=f"remove_{stock_data['symbol']}"):
                            st.session_state.watchlist.remove(stock_data['symbol'])
                            st.rerun()
        
        # Add button to analyze all stocks in batch
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Watchlist Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Compare All Models for Current Stock", use_container_width=True):
                st.session_state.stock_for_analysis = stock_symbol
                st.session_state.mode_for_redirect = "Model Comparison"
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh All Stocks", use_container_width=True):
                st.rerun()
        
        # Handle any redirects from watchlist
        if 'stock_for_analysis' in st.session_state and 'mode_for_redirect' in st.session_state:
            # This would be handled better with proper state management in a real app
            st.session_state.stock_symbol = st.session_state.stock_for_analysis
            st.session_state.app_mode = st.session_state.mode_for_redirect
            # Clear the redirect flags
            del st.session_state.stock_for_analysis
            del st.session_state.mode_for_redirect
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
    <h4 style='text-align: center; color: #0D47A1;'>💡 IntelliTrade Tips</h4>
    <p style='font-size: 0.9em;'>
    • <b>Stock Analysis:</b> Deep dive into a single stock with technical indicators<br>
    • <b>Model Comparison:</b> Compare predictions from multiple models<br>
    • <b>Watchlist:</b> Track multiple stocks in one view<br>
    • <b>Customize:</b> Adjust time periods, indicators and models in the sidebar<br>
    • <b>Export:</b> Download your analysis as PDF reports
    </p>
</div>
""", unsafe_allow_html=True)

# Footer with credits
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>IntelliTrade © 2025<br>Powered by yfinance & Streamlit</p>
</div>
""", unsafe_allow_html=True)