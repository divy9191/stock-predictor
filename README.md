# IntelliTrade - Advanced Stock Market Prediction Platform

A comprehensive Streamlit-based stock market analysis and prediction platform that combines advanced machine learning techniques with user-friendly, interactive data visualization.

## ✨ Features

- **Multiple Prediction Models**: Linear Regression, Random Forest, Extra Trees, KNN, and XGBoost
- **Technical Analysis**: SMA, Bollinger Bands, MACD, RSI indicators
- **Interactive UI**: Clean, tabbed interface with interactive charts
- **PDF Reports**: Generate and download detailed stock analysis reports
- **News Integration**: Real-time news about your selected stocks
- **Watchlist Management**: Track multiple stocks in a personalized watchlist
- **Model Comparison**: Compare the performance of different prediction models

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install via pip)

### Installation

1. Clone the repository or unzip the downloaded file:
```bash
git clone https://github.com/yourusername/intellitrade.git
cd intellitrade
```

2. Install the required packages:
```bash
# Using the provided package list
pip install -r package_list.txt

# Or install individually
pip install streamlit yfinance pandas numpy scikit-learn xgboost plotly statsmodels fpdf reportlab tensorflow
```

3. Configure the News API key:
   - Get an API key from [News API](https://newsapi.org/)
   - Create a file `.streamlit/secrets.toml` with:
   ```toml
   NEWS_API_KEY = "your_api_key_here"
   ```
   - Or set it as an environment variable:
   ```bash
   # On Windows
   set NEWS_API_KEY=your_api_key

   # On Mac/Linux
   export NEWS_API_KEY=your_api_key
   ```

4. Run the application:
```bash
streamlit run app.py
```

## 📊 Using IntelliTrade

### Stock Analysis Mode
- Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
- Select technical indicators to display
- Choose a prediction model and forecast horizon
- Click "Analyze Stock" to see a comprehensive analysis
- Download PDF reports of your analysis

### Model Comparison Mode
- Enter a stock symbol in the sidebar
- Select multiple models to compare
- View performance metrics and prediction charts
- Compare accuracy between different algorithms

### Watchlist Mode
- Add stocks to your personalized watchlist
- View summary cards for all watched stocks
- Sort by various metrics
- Click on any stock to analyze it in detail

## 🧰 Technical Details

### Data Sources
- Real-time stock data from Yahoo Finance
- News articles from News API

### Prediction Methods
- Linear Regression: Simple trend forecasting
- Random Forest: Ensemble learning for complex patterns
- Extra Trees: Enhanced randomization for robustness
- KNN: Pattern matching based on historical similarities
- XGBoost: Gradient boosting for high accuracy

### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

## 🔧 Project Structure

```
intellitrade/
├── app.py                  # Main application
├── models/
│   ├── evaluation.py       # Model evaluation metrics
│   └── predictions.py      # Prediction algorithms
├── utils/
│   ├── data_fetcher.py     # Stock data fetching
│   ├── news_fetcher.py     # News API integration
│   └── technical_analysis.py  # Technical indicators
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the awesome web framework
- [Yahoo Finance](https://finance.yahoo.com/) for providing financial data
- [News API](https://newsapi.org/) for news integration
- [Plotly](https://plotly.com/) for interactive visualizations