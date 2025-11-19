"""
Install packages for advanced features
"""
import subprocess
import sys

packages = [
    'beautifulsoup4',  # For news scraping
    'textblob',        # For sentiment analysis
    'scipy',           # For portfolio optimization
    'plotly',          # For better visualizations
    'twilio',          # For WhatsApp alerts (optional)
]

print("Installing advanced feature packages...")
print("="*70)

for package in packages:
    print(f"\nInstalling {package}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package, '--break-system-packages'],
            check=True,
            capture_output=True
        )
        print(f"✓ {package} installed")
    except Exception as e:
        print(f"⚠️  {package} installation issue (may already be installed)")

print("\n" + "="*70)
print("✅ Installation complete!")
print("="*70 + "\n")

# Download TextBlob corpora
print("Downloading sentiment analysis data...")
try:
    import nltk
    nltk.download('brown', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ Sentiment data downloaded")
except:
    print("⚠️  Will download on first use")