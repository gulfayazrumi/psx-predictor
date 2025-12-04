"""
Auto-update system for dashboard
Runs essential data updates when Streamlit loads
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

def should_update(file_path, max_age_minutes=30):
    """Check if file needs updating based on age"""
    if not Path(file_path).exists():
        return True
    
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    age = datetime.now() - file_time
    
    return age > timedelta(minutes=max_age_minutes)

def run_essential_updates():
    """Run essential data updates"""
    
    updates_needed = []
    
    # Check if updates are needed
    if should_update("reports/trading_signals.csv", max_age_minutes=30):
        updates_needed.append(("Live Signals", "update_live_signals.py"))
    
    if should_update("reports/sector_analysis.csv", max_age_minutes=60):
        updates_needed.append(("Market Analysis", "src/analysis/complete_analyzer.py"))

    if should_update("reports/seasonality_analysis.csv", max_age_minutes=1440):
        updates_needed.append(("Seasonality Analysis", "src/analysis/seasonality_analyzer.py"))
    
    if not updates_needed:
        return {"status": "up_to_date", "message": "All data is current"}
    
    # Run updates
    results = []
    for name, script in updates_needed:
        try:
            print(f"Updating {name}...")
            subprocess.run(
                [sys.executable, script],
                timeout=60,
                capture_output=True,
                check=False
            )
            results.append(f"✓ {name}")
        except Exception as e:
            results.append(f"✗ {name}: {str(e)}")
    
    return {
        "status": "updated",
        "message": "\n".join(results)
    }

if __name__ == "__main__":
    result = run_essential_updates()
    print(result['message'])
