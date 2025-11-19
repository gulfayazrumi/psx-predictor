"""
Install all required packages
"""
import subprocess
import sys

packages = [
    'pandas',
    'numpy',
    'scikit-learn',
    'tensorflow',
    'xgboost',
    'streamlit',
    'fastapi',
    'uvicorn',
    'requests',
    'beautifulsoup4',
    'textblob',
    'scipy',
    'python-dotenv',
    'schedule',
    'twilio',  # For WhatsApp
    'plotly',  # For better charts
]

print("Installing all required packages...")
print("="*70)

for package in packages:
    print(f"\nInstalling {package}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package, '--break-system-packages'],
            check=True
        )
        print(f"✓ {package} installed")
    except:
        print(f"✗ {package} failed (may already be installed)")

print("\n" + "="*70)
print("✅ Installation complete!")
print("="*70 + "\n")