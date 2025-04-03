# IntelliTrade Installation Guide

This guide provides detailed instructions for setting up and deploying the IntelliTrade stock market prediction platform.

## Local Installation

### Prerequisites

1. Python 3.7+ installed on your system
2. Git (optional, for cloning the repository)
3. Package manager (pip)

### Step-by-Step Installation

1. **Get the code**:
   - Option 1: Clone the repository:
     ```bash
     git clone https://github.com/yourusername/intellitrade.git
     cd intellitrade
     ```
   - Option 2: Unzip the downloaded zip file:
     ```bash
     unzip intellitrade.zip
     cd intellitrade
     ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r package_list.txt
   ```

4. **Set up News API**:
   - Sign up at [News API](https://newsapi.org) to get a free API key
   - Create a file `.streamlit/secrets.toml` with your key:
     ```toml
     NEWS_API_KEY = "your_api_key_here"
     ```
   - Or set as an environment variable:
     ```bash
     # On Windows
     set NEWS_API_KEY=your_api_key

     # On Mac/Linux
     export NEWS_API_KEY=your_api_key
     ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Deployment Options

### Deploying on Streamlit Cloud

1. Create a GitHub repository with your IntelliTrade code
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app by selecting your repository
5. Add your News API key in the Streamlit Cloud secrets management

### Deploying on Heroku

1. Install Heroku CLI and create an account
2. Create a `Procfile` with:
   ```
   web: streamlit run app.py --server.port $PORT
   ```
3. Create a `runtime.txt` with:
   ```
   python-3.10.12
   ```
4. Deploy with:
   ```bash
   heroku login
   heroku create intellitrade-app
   git push heroku main
   ```
5. Set your API key:
   ```bash
   heroku config:set NEWS_API_KEY=your_api_key
   ```

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   - Ensure all packages are installed: `pip install -r package_list.txt`

2. **News API not working**:
   - Verify your API key is correctly set up
   - Check your News API usage limits (free tier has limitations)

3. **No data for certain stocks**:
   - Verify the stock symbol is correct
   - Some very recent or obscure stocks may not have data in Yahoo Finance

4. **Performance issues**:
   - For slow machines, reduce prediction days or use lighter models

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r package_list.txt --upgrade
```

## Support

For questions or issues, please open an issue on the GitHub repository or contact the maintainer at your-email@example.com.