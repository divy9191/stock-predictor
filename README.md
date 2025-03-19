# Stock Market Prediction App

A Streamlit-based application for stock market prediction using multiple ML algorithms and real-time data analysis. The app provides technical analysis indicators and news integration for comprehensive stock analysis.

## Features

- Real-time stock data fetching using yfinance
- Multiple prediction models:
  - Linear Regression
  - Random Forest
  - Extra Trees
  - KNN
  - XGBoost
- Technical Analysis Indicators:
  - Simple Moving Average (SMA)
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
- Real-time news integration using News API
- Interactive charts using Plotly

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/divy993/stock-predictor.git
cd stock-predictor
```

2. Create required directories:
```bash
mkdir -p .streamlit models utils
```

3. Install required packages:
```bash
pip install streamlit yfinance pandas numpy scikit-learn xgboost plotly requests
```

4. Set up your News API key:
   - Get your API key from [News API](https://newsapi.org)
   - Set it as an environment variable:
     ```bash
     # On Windows
     set NEWS_API_KEY=your_api_key

     # On Mac/Linux
     export NEWS_API_KEY=your_api_key
     ```

5. Run the app:
```bash
# For local development
streamlit run app.py --server.port 5000
```

The app will open in your default web browser at `http://localhost:5000`

## Usage

1. Enter a stock symbol (e.g., AAPL for Apple)
2. Select the historical data period
3. Choose technical indicators to display
4. Select prediction model and days
5. Click 'Analyze' to see analysis and predictions

## Technical Indicators

- **SMA (Simple Moving Average)**: Helps identify trend direction
- **Bollinger Bands**: Shows volatility and potential price levels
- **MACD**: Identifies momentum and trend changes
- **RSI**: Indicates overbought or oversold conditions

## License

[MIT](https://choosealicense.com/licenses/mit/)