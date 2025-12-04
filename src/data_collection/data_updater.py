"""
Data Updater Module
Responsible for synchronizing local historical CSVs with latest market data.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from .sarmaaya_api import SarmayaAPI

logger = logging.getLogger(__name__)

class DataUpdater:
    def __init__(self, data_dir="data/raw/historical"):
        self.data_dir = Path(data_dir)
        self.api = SarmayaAPI()
        
    def update_history(self, symbol: str) -> bool:
        """
        Update historical data for a symbol if needed.
        Returns True if data was updated, False otherwise.
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            logger.warning(f"No historical file found for {symbol}")
            return False
            
        try:
            # Load existing data
            df = pd.read_csv(csv_path)
            
            # Clean column names and coalesce duplicates
            df.columns = df.columns.str.strip().str.lower()
            
            # Combine duplicate columns explicitly
            df_clean = pd.DataFrame()
            for col in df.columns.unique():
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):  # Multiple columns with same name
                    # Combine them using combine_first (fills NaNs from first with values from others)
                    combined = col_data.iloc[:, 0]
                    for i in range(1, col_data.shape[1]):
                        combined = combined.combine_first(col_data.iloc[:, i])
                    df_clean[col] = combined
                else:
                    df_clean[col] = col_data
            df = df_clean
            
            # Handle date column
            date_col = 'time' if 'time' in df.columns else 'date'
            if date_col not in df.columns:
                logger.error(f"No date column in {symbol}.csv")
                return False
                
            # Use errors='coerce' to handle mixed formats
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values('date')
            
            # Get last valid date (max ignores NaT)
            last_date = df['date'].max()
            today = pd.Timestamp.now().normalize()
            
            # Check if update is needed (if last date is older than yesterday)
            # We allow today's data to be missing as it might be live
            days_diff = (today - last_date).days
            
            if days_diff <= 1:
                return False
                
            logger.info(f"Updating {symbol}: Last date {last_date.date()}, missing {days_diff} days")
            
            # Fetch missing data
            # Fetch a bit more to be safe, minimum 30 days to avoid API issues
            fetch_days = max(30, days_diff + 5)
            new_data = self.api.get_stock_history(symbol, days=fetch_days)
            
            if new_data is None or new_data.empty:
                logger.warning(f"Could not fetch new data for {symbol}")
                return False
                
            # Process new data
            new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
            
            # Ensure naive datetime for comparison
            if new_data['date'].dt.tz is not None:
                new_data['date'] = new_data['date'].dt.tz_localize(None)
                
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # last_date was already calculated earlier using .max(), no need to recalculate
            
            # Filter for rows after last_date
            new_rows = new_data[new_data['date'] > last_date].copy()
            
            if new_rows.empty:
                logger.info(f"No new rows found for {symbol} after filtering")
                return False
                
            # Ensure columns match
            # Map API columns to CSV columns
            # CSV usually has: time, open, high, low, close, volume
            # API might have: date, price, close, volume (without open/high/low)
            
            # First, ensure we have a 'close' price
            if 'close' not in new_rows.columns:
                if 'price' in new_rows.columns:
                    new_rows['close'] = new_rows['price']
                else:
                    logger.warning(f"No price data in API response for {symbol}")
                    return False
            
            # For missing OHLC, use close price
            for col in ['open', 'high', 'low']:
                if col not in new_rows.columns:
                    new_rows[col] = new_rows['close']
            
            # Ensure volume exists
            if 'volume' not in new_rows.columns:
                new_rows['volume'] = 0
            
            # Prepare rows to append
            rows_to_append = []
            for _, row in new_rows.iterrows():
                record = {
                    date_col: row['date'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                rows_to_append.append(record)
                
            if not rows_to_append:
                return False
                
            # Append and save
            new_df = pd.DataFrame(rows_to_append)
            
            # Ensure we don't introduce duplicates
            # Combine, drop duplicates by date, sort
            combined_df = pd.concat([df, new_df])
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date')
            
            # Restore original column format if needed (e.g. 'time' vs 'date')
            # The df already has 'date' column added for processing, we should remove it if it wasn't there
            # But wait, we want to save it back in the original format.
            # If original had 'time', we use 'time'.
            
            # Drop the helper 'date' column if it wasn't in original (but we used it for sorting)
            # Actually, let's just save the columns that were in the original CSV + any new ones
            # But we need to make sure 'date' or 'time' is formatted correctly string-wise if needed?
            # Pandas to_csv handles datetime objects fine.
            
            # Just keep the columns that were in the original CSV
            original_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            original_cols = [c.strip().lower() for c in original_cols]
            
            # Map our standardized columns back to original names if possible?
            # Or just save standardized columns. The project seems to use 'time' or 'date'.
            # Let's standardize on saving what we have, but ensuring 'time' is populated if it was the date col.
            
            if 'time' in original_cols and 'date' not in original_cols:
                combined_df['time'] = combined_df['date']
                if 'date' in combined_df.columns:
                    combined_df = combined_df.drop(columns=['date'])
            
            # Save
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"Updated {symbol} with {len(new_rows)} new rows. New last date: {combined_df.iloc[-1][date_col]}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
            return False
