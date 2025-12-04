from src.data_collection.sarmaaya_api import SarmayaAPI
import json

api = SarmayaAPI()

print("Testing sectors/list...")
sectors = api._make_request("sectors/list")
if sectors:
    print(json.dumps(sectors, indent=2))
else:
    print("✗ sectors/list failed")

