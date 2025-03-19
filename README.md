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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/divy993/stock-predictor.git
cd stock-predictor
```

2. Install required packages:
```bash
pip install streamlit yfinance pandas numpy scikit-learn xgboost plotly
```

3. Set up your News API key:
   - Get your API key from [News API](https://newsapi.org)
   - Set it as an environment variable:
     ```bash
     # On Windows
     set NEWS_API_KEY=your_api_key
     # On Mac/Linux
     export NEWS_API_KEY=your_api_key
     ```

4. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock symbol (e.g., AAPL for Apple)
2. Select the historical data period
3. Choose technical indicators to display
4. Select a prediction model and forecast period
5. Click 'Analyze' to see predictions and analysis

## Project Structure

```
project/
├── models/
│   ├── evaluation.py
│   └── predictions.py
├── utils/
│   ├── data_fetcher.py
│   ├── news_fetcher.py
│   └── technical_analysis.py
├── .streamlit/
│   └── config.toml
└── app.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
