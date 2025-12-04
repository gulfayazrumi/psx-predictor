from src.data_collection.data_updater import DataUpdater
import logging

logging.basicConfig(level=logging.INFO)

updater = DataUpdater()
print("Updating OGDC...")
result = updater.update_history("OGDC")
print(f"Update result: {result}")
