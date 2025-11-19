"""
News sentiment analysis from Sarmaaya
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class NewsSentimentAnalyzer:
    """Scrape and analyze news sentiment"""
    
    def __init__(self):
        self.base_url = "https://sarmaaya.pk"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_latest_news(self, max_articles=50):
        """Scrape latest news from Sarmaaya"""
        
        news_url = f"{self.base_url}/news"
        
        try:
            response = self.session.get(news_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            
            # Find news articles (adjust selectors based on actual HTML structure)
            news_items = soup.find_all('article', class_='news-item', limit=max_articles)
            
            for item in news_items:
                try:
                    title_elem = item.find('h2') or item.find('h3')
                    title = title_elem.text.strip() if title_elem else ''
                    
                    link_elem = item.find('a')
                    link = link_elem['href'] if link_elem else ''
                    if link and not link.startswith('http'):
                        link = self.base_url + link
                    
                    date_elem = item.find('time') or item.find(class_='date')
                    date = date_elem.text.strip() if date_elem else datetime.now().strftime('%Y-%m-%d')
                    
                    summary_elem = item.find('p')
                    summary = summary_elem.text.strip() if summary_elem else ''
                    
                    if title:
                        articles.append({
                            'title': title,
                            'summary': summary,
                            'link': link,
                            'date': date,
                            'scraped_at': datetime.now()
                        })
                
                except Exception as e:
                    continue
            
            return pd.DataFrame(articles)
        
        except Exception as e:
            print(f"Error scraping news: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        
        if not text:
            return 0, 'neutral'
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return polarity, sentiment
        
        except:
            return 0, 'neutral'
    
    def extract_stock_mentions(self, text):
        """Extract stock symbols mentioned in text"""
        
        # Common PSX stock patterns
        pattern = r'\b([A-Z]{2,5})\b'
        
        matches = re.findall(pattern, text)
        
        # Filter common words that aren't stocks
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        stocks = [m for m in matches if m not in common_words]
        
        return list(set(stocks))
    
    def get_stock_sentiment_score(self, symbol, days=7):
        """Get sentiment score for a specific stock"""
        
        news_df = self.scrape_latest_news()
        
        if len(news_df) == 0:
            return None
        
        # Analyze sentiment for each article
        news_df['text'] = news_df['title'] + ' ' + news_df['summary']
        news_df['polarity'], news_df['sentiment'] = zip(*news_df['text'].apply(self.analyze_sentiment))
        news_df['mentioned_stocks'] = news_df['text'].apply(self.extract_stock_mentions)
        
        # Filter articles mentioning this stock
        stock_news = news_df[news_df['mentioned_stocks'].apply(lambda x: symbol.upper() in x)]
        
        if len(stock_news) == 0:
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'article_count': 0,
                'latest_news': []
            }
        
        # Calculate average sentiment
        avg_sentiment = stock_news['polarity'].mean()
        
        # Classify overall sentiment
        if avg_sentiment > 0.1:
            overall_sentiment = 'positive'
        elif avg_sentiment < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'sentiment_score': avg_sentiment,
            'sentiment': overall_sentiment,
            'article_count': len(stock_news),
            'positive_count': len(stock_news[stock_news['sentiment'] == 'positive']),
            'negative_count': len(stock_news[stock_news['sentiment'] == 'negative']),
            'latest_news': stock_news[['title', 'sentiment', 'polarity', 'date']].head(5).to_dict('records')
        }
    
    def get_market_sentiment(self):
        """Get overall market sentiment from news"""
        
        news_df = self.scrape_latest_news()
        
        if len(news_df) == 0:
            return None
        
        # Analyze all articles
        news_df['text'] = news_df['title'] + ' ' + news_df['summary']
        news_df['polarity'], news_df['sentiment'] = zip(*news_df['text'].apply(self.analyze_sentiment))
        
        # Calculate metrics
        avg_sentiment = news_df['polarity'].mean()
        positive_ratio = len(news_df[news_df['sentiment'] == 'positive']) / len(news_df)
        
        return {
            'overall_sentiment_score': avg_sentiment,
            'positive_ratio': positive_ratio,
            'total_articles': len(news_df),
            'positive_count': len(news_df[news_df['sentiment'] == 'positive']),
            'negative_count': len(news_df[news_df['sentiment'] == 'negative']),
            'neutral_count': len(news_df[news_df['sentiment'] == 'neutral'])
        }


def generate_sentiment_report():
    """Generate news sentiment report"""
    
    analyzer = NewsSentimentAnalyzer()
    
    print("\n" + "="*70)
    print("NEWS SENTIMENT ANALYSIS")
    print("="*70)
    
    # Overall market sentiment
    market_sentiment = analyzer.get_market_sentiment()
    
    if market_sentiment:
        print("\n📰 OVERALL MARKET SENTIMENT")
        print("-"*70)
        print(f"Sentiment Score:  {market_sentiment['overall_sentiment_score']:+.3f}")
        print(f"Positive Ratio:   {market_sentiment['positive_ratio']:.1%}")
        print(f"Total Articles:   {market_sentiment['total_articles']}")
        print(f"  Positive:       {market_sentiment['positive_count']}")
        print(f"  Negative:       {market_sentiment['negative_count']}")
        print(f"  Neutral:        {market_sentiment['neutral_count']}")
    
    # Get sentiment for top stocks
    print("\n📊 TOP STOCKS SENTIMENT")
    print("-"*70)
    
    top_stocks = ['HBL', 'OGDC', 'PPL', 'ENGRO', 'LUCK', 'MCB', 'UBL', 'PSO', 'HUBC', 'FFC']
    
    sentiment_data = []
    
    for symbol in top_stocks:
        sentiment = analyzer.get_stock_sentiment_score(symbol)
        if sentiment:
            sentiment_data.append(sentiment)
            
            emoji = '🟢' if sentiment['sentiment'] == 'positive' else '🔴' if sentiment['sentiment'] == 'negative' else '🟡'
            print(f"{emoji} {symbol:<6} Score: {sentiment['sentiment_score']:>+6.3f}  Articles: {sentiment['article_count']:>3}  Sentiment: {sentiment['sentiment'].upper()}")
    
    print("\n" + "="*70)
    
    # Save to CSV
    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        df.to_csv('reports/sentiment_analysis.csv', index=False)
        print("✓ Sentiment report saved to: reports/sentiment_analysis.csv")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Install textblob if needed
    try:
        import textblob
    except ImportError:
        print("Installing textblob...")
        import subprocess
        subprocess.run(['pip', 'install', 'textblob', '--break-system-packages'])
    
    generate_sentiment_report()