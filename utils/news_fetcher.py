import requests
import streamlit as st
from datetime import datetime, timedelta
import os

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_stock_news(symbol: str, company_name: str = None) -> list:
    """
    Fetch news related to a stock from News API
    
    Args:
        symbol (str): Stock symbol
        company_name (str): Company name for better news search
    
    Returns:
        list: List of news articles
    """
    try:
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            st.error("News API key not found")
            return []

        # Create search query using both symbol and company name
        query = f"{symbol}"
        if company_name:
            query += f" OR {company_name}"

        # Get news from the last 7 days
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"Error fetching news: {response.status_code}")
            return []
            
        news_data = response.json()
        
        # Process and filter relevant articles
        articles = news_data.get('articles', [])
        processed_articles = []
        
        for article in articles[:10]:  # Get top 10 articles
            processed_articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '')
            })
            
        return processed_articles
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []
