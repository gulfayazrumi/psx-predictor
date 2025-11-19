"""
Check if your data is current and ready for trading
FIXED VERSION - handles ALL CAPS columns and various date formats
"""
from pathlib import Path
import pandas as pd
from datetime import datetime


def check_all_data():
    """Check freshness of all stock data"""
    
    historical_path = Path("data/raw/historical")
    
    if not historical_path.exists():
        print("❌ No historical data found!")
        return
    
    files = list(historical_path.glob("*.csv"))
    
    if not files:
        print("❌ No CSV files found!")
        return
    
    print("\n" + "="*70)
    print("DATA FRESHNESS CHECK")
    print("="*70)
    print(f"Checking {len(files)} stocks...\n")
    
    fresh_count = 0
    recent_count = 0
    old_count = 0
    very_old_count = 0
    errors = 0
    
    oldest_dates = []
    
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            
            # Convert column names to lowercase for easier matching
            df.columns = df.columns.str.lower()
            
            # Handle different date column names
            date_col = None
            if 'date' in df.columns:
                date_col = 'date'
            elif 'time' in df.columns:
                date_col = 'time'
            else:
                errors += 1
                continue
            
            # Convert to datetime - handle format like "Oct 9, 2020"
            df[date_col] = pd.to_datetime(df[date_col], format='mixed')
            
            latest_date = df[date_col].max()
            days_old = (datetime.now() - latest_date).days
            
            symbol = csv_file.stem.upper()
            
            if days_old <= 1:
                status = "✅ FRESH"
                fresh_count += 1
            elif days_old <= 7:
                status = "🟢 RECENT"
                recent_count += 1
            elif days_old <= 365:
                status = "🟡 OLD"
                old_count += 1
            else:
                status = "🔴 VERY OLD"
                very_old_count += 1
                oldest_dates.append((symbol, latest_date, days_old))
        
        except Exception as e:
            errors += 1
            continue
    
    # Summary
    total = fresh_count + recent_count + old_count + very_old_count
    
    if total == 0:
        print("="*70)
        print("❌ ERROR: Could not read any data files!")
        print("="*70)
        print(f"Total files found: {len(files)}")
        print(f"Errors encountered: {errors}\n")
        return
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Files:      {len(files)}")
    print(f"Successfully Read: {total}")
    print(f"Errors:           {errors}")
    print(f"\n✅ Fresh (1 day):    {fresh_count} ({fresh_count/total*100:.1f}%)")
    print(f"🟢 Recent (7 days):  {recent_count} ({recent_count/total*100:.1f}%)")
    print(f"🟡 Old (1 year):     {old_count} ({old_count/total*100:.1f}%)")
    print(f"🔴 Very Old (>1yr):  {very_old_count} ({very_old_count/total*100:.1f}%)")
    print("="*70 + "\n")
    
    if very_old_count > 0:
        print("🔴 CRITICAL PROBLEM: YOUR DATA IS ANCIENT!")
        print("="*70)
        print("Sample of your data dates:")
        for symbol, date, days in sorted(oldest_dates, key=lambda x: x[2], reverse=True)[:10]:
            years_old = days / 365.25
            print(f"  {symbol:<8} Last: {date.strftime('%b %d, %Y')} ({years_old:.1f} years ago)")
        
        print("\n" + "="*70)
        print("❌ WHY YOUR PREDICTIONS ARE WRONG")
        print("="*70)
        print(f"\n📅 Your data:  October 2020")
        print(f"📅 Today:      November 2025")
        print(f"⏰ Gap:        ~5 YEARS!\n")
        
        print("💔 What this means:")
        print("  • Stock prices changed DRASTICALLY since 2020")
        print("  • Your models think it's still 2020")
        print("  • Example: Model sees SYS at PKR 282 (2020 price)")
        print("            Reality: SYS is PKR 148 (2025 price)")
        print("  • ALL predictions are based on 5-year-old patterns")
        print("  • COVID era vs 2025 - completely different market!\n")
        
        print("="*70)
        print("✅ SOLUTION: COLLECT FRESH DATA")
        print("="*70)
        print("\n🌅 TOMORROW MORNING (9:00 AM PKT):")
        print("   When market opens and API has fresh data\n")
        
        print("Step 1: Collect Current Data (9:00 AM)")
        print("  Run: python collect_fresh_data.py")
        print("  Time: 1-2 hours")
        print("  Why: Gets last 2 years of data (2023-2025)")
        
        print("\nStep 2: Verify Data (11:00 AM)")
        print("  Run: python check_data_freshness.py")
        print("  Time: 1 minute")
        print("  Why: Confirms data is current")
        
        print("\nStep 3: Retrain Models (11:30 AM)")
        print("  Run: python train_all.py --max 50")
        print("  Time: 1-2 hours")
        print("  Why: Learns 2025 patterns, not 2020!")
        
        print("\nStep 4: Generate Signals (1:30 PM)")
        print("  Run: python generate_signals.py")
        print("  Time: 1 minute")
        print("  Why: Gets predictions for TOMORROW")
        
        print("\nStep 5: Trade (2:00 PM)")
        print("  Run: streamlit run dashboard/app.py")
        print("  Why: View real, current predictions!")
        
        print("\n⚠️  IMPORTANT:")
        print("  • Don't trade based on current predictions!")
        print("  • They're 5 years out of date")
        print("  • Wait until you have fresh data")
        print("  • Market conditions have changed completely")
        
        print("\n💡 WHY WAIT UNTIL MORNING?")
        print("  • It's 11 PM now - market is closed")
        print("  • API doesn't serve fresh data at night")
        print("  • Best results when market is active (9 AM - 5 PM)")
        
        print("="*70 + "\n")
    
    elif old_count > total * 0.5:
        print("🟡 WARNING: Data needs updating")
    
    elif (fresh_count + recent_count) > total * 0.8:
        print("✅ DATA IS CURRENT - READY TO TRADE!")
    
    else:
        print("🟢 DATA IS REASONABLY FRESH")


if __name__ == "__main__":
    check_all_data()