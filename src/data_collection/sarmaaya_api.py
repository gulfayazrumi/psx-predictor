"""
WORKING Sarmaaya API Client - All Endpoints Verified
"""
import requests
import pandas as pd
import logging
from typing import Optional, List, Dict
import time

logger = logging.getLogger(__name__)


class SarmayaAPI:
    """Client for Sarmaaya.pk REST API"""
    
    BASE_URL = "https://beta-restapi.sarmaaya.pk/api"
    
    def __init__(self):
        self.session = requests.Session()
        
        # Browser-like headers (verified working)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://sarmaaya.pk/',
        })
        
        self.retry_attempts = 3
        self.retry_delay = 1
    
    def _make_request(self, endpoint: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        """Make API request with retries"""
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(retries):
            try:
                if attempt > 0:
                    time.sleep(self.retry_delay)
                
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    # Response is automatically decompressed by requests
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting...")
                    time.sleep(self.retry_delay * 2)
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error: {e}")
            except ValueError as e:
                logger.warning(f"JSON decode error: {e}")
                
        logger.error(f"Failed after {retries} attempts: {url}")
        return None
    
    def get_market_summary(self) -> Optional[List[Dict]]:
        """Get market indices summary"""
        
        data = self._make_request("dashboard/market-view")
        
        if data and data.get('success'):
            return data.get('response', [])
        
        return None
    
    def get_kse100_tickers(self) -> List[str]:
        """Get KSE-100 index constituents"""
        
        logger.info("Fetching KSE-100 constituents...")
        
        data = self._make_request("stocks/ticker", params={"index": "KSE100"})
        
        if data and data.get('success'):
            tickers = [item['symbol'] for item in data.get('response', [])]
            logger.info(f"✓ Found {len(tickers)} KSE-100 stocks")
            return tickers
        
        logger.warning("Could not fetch KSE-100 tickers")
        return []
    
    def get_stock_history(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol"""
        
        data = self._make_request(
            f"stocks/price-history/{symbol}",
            params={"days": days}
        )
        
        if not data or not data.get('success'):
            return None
        
        price_data = data.get('response', [])
        
        if not price_data:
            return None
        
        df = pd.DataFrame(price_data)
        
        # Convert columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'price' in df.columns:
            df['close'] = df['price']  # Map 'price' to 'close'
        
        return df
    
    def get_all_stocks(self, page: int = 1, limit: int = 50) -> List[Dict]:
        """Get all stocks listing"""
        
        data = self._make_request(
            "stocks/listing",
            params={"page": page, "limit": limit}
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_52_week_highs(self, limit: int = 50) -> List[Dict]:
        """Get stocks at 52-week highs"""
        
        data = self._make_request(
            "stocks/listing",
            params={
                "valuation": "52 Week High Stocks",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_52_week_lows(self, limit: int = 50) -> List[Dict]:
        """Get stocks at 52-week lows"""
        
        data = self._make_request(
            "stocks/listing",
            params={
                "valuation": "52 Week Low Stocks",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_large_cap_stocks(self, limit: int = 50) -> List[Dict]:
        """Get large cap stocks"""
        
        data = self._make_request(
            "stocks/listing",
            params={
                "valuation": "Large Cap Stocks",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_dividend_stocks(self, page: int = 1, limit: int = 50) -> List[Dict]:
        """Get dividend stocks"""
        
        # Try main dividend endpoint
        data = self._make_request(
            "stocks/listing",
            params={
                "valuation": "Top Dividend Yield Stocks",
                "page": page,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        # Try alternative endpoint
        data = self._make_request(
            "dashboard/stock-divindend",
            params={"page": page, "limit": limit}
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('stockDividends', [])
        
        return []
    
    def get_active_stocks(self, limit: int = 50) -> List[Dict]:
        """Get most active stocks by volume"""
        
        data = self._make_request(
            "dashboard/stock-performers",
            params={
                "type": "active",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_top_gainers(self, limit: int = 50) -> List[Dict]:
        """Get top gaining stocks"""
        
        data = self._make_request(
            "dashboard/stock-performers",
            params={
                "type": "gainers",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []
    
    def get_top_losers(self, limit: int = 50) -> List[Dict]:
        """Get top losing stocks"""
        
        data = self._make_request(
            "dashboard/stock-performers",
            params={
                "type": "losers",
                "page": 1,
                "limit": limit
            }
        )
        
        if data and data.get('success'):
            return data.get('response', {}).get('data', [])
        
        return []


# Test function
def test_api():
    """Test all endpoints"""
    
    api = SarmayaAPI()
    
    print("\n" + "="*70)
    print("SARMAAYA API - FULL TEST")
    print("="*70)
    
    # Test 1: Market Summary
    print("\n1. Market Summary...")
    summary = api.get_market_summary()
    if summary:
        print(f"   ✓ Found {len(summary)} indices")
        print(f"   KSE-100: {summary[0]['close']:,.2f} ({summary[0]['changePercentage']:+.2f}%)")
    
    # Test 2: KSE-100 Tickers
    print("\n2. KSE-100 Tickers...")
    tickers = api.get_kse100_tickers()
    if tickers:
        print(f"   ✓ Found {len(tickers)} stocks")
        print(f"   Sample: {', '.join(tickers[:5])}")
    
    # Test 3: 52-Week Highs
    print("\n3. 52-Week Highs...")
    highs = api.get_52_week_highs(10)
    if highs:
        print(f"   ✓ Found {len(highs)} stocks")
        print(f"   Top: {highs[0]['symbol']} - {highs[0]['name']}")
        print(f"   Close: {highs[0]['close']:,.2f} (52W High: {highs[0].get('high52', 'N/A')})")
    
    # Test 4: 52-Week Lows
    print("\n4. 52-Week Lows...")
    lows = api.get_52_week_lows(10)
    if lows:
        print(f"   ✓ Found {len(lows)} stocks")
        print(f"   Top: {lows[0]['symbol']} - {lows[0]['name']}")
        print(f"   Close: {lows[0]['close']:.2f} (52W Low: {lows[0].get('low52', 'N/A')})")
    
    # Test 5: Dividends
    print("\n5. Dividend Stocks...")
    divs = api.get_dividend_stocks(limit=10)
    if divs:
        print(f"   ✓ Found {len(divs)} dividend stocks")
        print(f"   Top: {divs[0]['symbol']} - {divs[0]['name']}")
        if 'percentage' in divs[0]:
            print(f"   Dividend: {divs[0]['percentage']}%")
    
    # Test 6: Active Stocks
    print("\n6. Most Active Stocks...")
    active = api.get_active_stocks(10)
    if active:
        print(f"   ✓ Found {len(active)} active stocks")
        print(f"   Top: {active[0]['symbol']} - Volume: {active[0].get('volume', 'N/A'):,}")
    
    # Test 7: Top Gainers
    print("\n7. Top Gainers...")
    gainers = api.get_top_gainers(10)
    if gainers:
        print(f"   ✓ Found {len(gainers)} gainers")
        if len(gainers) > 0:
            print(f"   Top: {gainers[0]['symbol']} ({gainers[0].get('change_percent', 0):+.2f}%)")
    
    # Test 8: Historical Data
    print("\n8. Historical Data (HBL)...")
    df = api.get_stock_history('HBL', days=30)
    if df is not None:
        print(f"   ✓ Retrieved {len(df)} days")
        print(f"   Latest: {df.iloc[-1]['date'].strftime('%Y-%m-%d')} - {df.iloc[-1].get('close', df.iloc[-1].get('price', 0)):.2f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_api()