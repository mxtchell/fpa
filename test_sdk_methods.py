"""Test SDK methods for data access"""
from dotenv import load_dotenv
load_dotenv()

from answer_rocket import AnswerRocketClient
import os

DATASET_ID = "d762aa87-6efb-47c4-b491-3bdc27147d4e"

print("=" * 80)
print("Testing SDK Data Methods")
print("=" * 80)

client = AnswerRocketClient(
    url=os.getenv('AR_BASE_URL'),
    token=os.getenv('AR_API_KEY')
)

print(f"\nDataset ID: {DATASET_ID}")

# Check what methods are available
print("\n Available data methods:")
for method in dir(client.data):
    if not method.startswith('_'):
        print(f"  - {method}")

# Try to query data using get_data_preview
print("\n\nTrying get_data_preview...")
try:
    preview = client.data.get_data_preview(dataset_id=DATASET_ID, row_count=10)
    print(f"✓ Data preview retrieved!")
    print(f"  Type: {type(preview)}")
    print(f"  Content: {preview}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Try to get metadata
print("\n\nTrying to get dataset metadata...")
try:
    metadata_result = client.data.get_metadata(dataset_id=DATASET_ID)
    print(f"✓ Metadata retrieved!")
    print(f"  Type: {type(metadata_result)}")
    if hasattr(metadata_result, '__dict__'):
        for key, value in metadata_result.__dict__.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    else:
        print(f"  {metadata_result}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 80)
