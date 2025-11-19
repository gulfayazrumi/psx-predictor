"""
Debug Sarmaaya API - See exactly what's being returned
"""
import requests
import json

def test_endpoint(url, params=None, description=""):
    """Test a single endpoint and show response"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    print('-'*70)
    
    try:
        # Try with browser headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://sarmaaya.pk/',
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content Length: {len(response.content)} bytes")
        print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
        
        # Show first 500 chars of response
        content = response.text[:500]
        print(f"\nFirst 500 chars of response:")
        print(content)
        
        # Try to parse as JSON
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"\n✓ Valid JSON received")
                print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
                if isinstance(data, dict):
                    if 'success' in data:
                        print(f"Success: {data['success']}")
                    if 'response' in data:
                        resp = data['response']
                        if isinstance(resp, dict):
                            print(f"Response keys: {list(resp.keys())}")
                        elif isinstance(resp, list):
                            print(f"Response items: {len(resp)}")
                
                return data
            except Exception as e:
                print(f"\n✗ JSON parse error: {e}")
        else:
            print(f"\n✗ HTTP Error: {response.status_code}")
        
    except Exception as e:
        print(f"\n✗ Request failed: {e}")
    
    return None


def main():
    """Test all Sarmaaya endpoints"""
    
    print("\n" + "="*70)
    print("SARMAAYA API DIAGNOSTIC TEST")
    print("="*70)
    
    # Test 1: Market View (should work)
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/dashboard/market-view",
        description="Market Summary"
    )
    
    # Test 2: Dividends (should work)
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/dashboard/stock-divindend",
        params={"page": 1, "limit": 10},
        description="Dividend Stocks"
    )
    
    # Test 3: Stock Listing (should work)
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/stocks/listing",
        params={"page": 1, "limit": 10},
        description="All Stocks Listing"
    )
    
    # Test 4: 52-Week Highs
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/stocks/listing",
        params={"valuation": "52 Week High Stocks", "page": 1, "limit": 10},
        description="52-Week Highs"
    )
    
    # Test 5: Active Stocks
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/dashboard/stock-performers",
        params={"type": "active", "page": 1, "limit": 10},
        description="Active Stocks"
    )
    
    # Test 6: KSE-100 Tickers
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/stocks/ticker",
        params={"index": "KSE100"},
        description="KSE-100 Tickers"
    )
    
    # Test 7: Historical Data (HBL)
    test_endpoint(
        "https://beta-restapi.sarmaaya.pk/api/stocks/price-history/HBL",
        params={"days": 30},
        description="HBL Historical Data"
    )
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()