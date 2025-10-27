"""Test dataset access"""
from dotenv import load_dotenv
load_dotenv()

from answer_rocket import AnswerRocketClient

DATASET_ID = "d762aa87-6efb-47c4-b491-3bdc27147d4e"

print("=" * 80)
print("Testing Dataset Access")
print("=" * 80)

client = AnswerRocketClient()

print(f"\nTrying to get dataset info for: {DATASET_ID}")

try:
    # Try to get dataset metadata
    dataset_info = client.data.get_dataset(dataset_id=DATASET_ID)
    print(f"\n✓ Dataset found!")
    print(f"  Dataset: {dataset_info}")

except Exception as e:
    print(f"\n✗ Failed to get dataset:")
    print(f"  {type(e).__name__}: {str(e)}")

print("\n" + "=" * 80)
