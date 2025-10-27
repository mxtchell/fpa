"""Test connection to AnswerRocket API and query dataset"""
from dotenv import load_dotenv
load_dotenv()

from answer_rocket import AnswerRocketClient
import pandas as pd
import os

# Database Configuration
DATABASE_ID = "83C2268F-AF77-4D00-8A6B-7181DC06643E"
DATASET_ID = "d762aa87-6efb-47c4-b491-3bdc27147d4e"

print("=" * 80)
print("Testing AnswerRocket Connection")
print("=" * 80)

# Create client with explicit URL and token
print("\n1. Creating AnswerRocketClient...")
url = os.getenv('AR_BASE_URL')
token = os.getenv('AR_API_KEY')

print(f"   URL: {url}")
print(f"   Token: {token[:20]}..." if token else "   Token: Not set")

client = AnswerRocketClient(url=url, token=token)
print("✓ Client created successfully")

# Test basic query to see what data structure looks like
print(f"\n2. Testing query on DATABASE_ID: {DATABASE_ID}")

# First, let's see what tables are available
tables_query = """
SHOW TABLES
"""

print(f"\nGetting list of tables...")
try:
    result = client.data.execute_sql_query(
        database_id=DATABASE_ID,
        sql_query=tables_query,
        row_limit=100
    )

    if hasattr(result, 'success') and result.success and hasattr(result, 'df'):
        print(f"\n✓ Available tables:")
        print(result.df)
    else:
        print(f"Could not get tables: {result.error if hasattr(result, 'error') else 'Unknown error'}")
except Exception as e:
    print(f"Error getting tables: {e}")

# Now try a simple query
test_query = """
SELECT *
LIMIT 10
"""

print(f"\nExecuting query:\n{test_query}")

try:
    result = client.data.execute_sql_query(
        database_id=DATABASE_ID,
        sql_query=test_query,
        row_limit=10
    )

    print(f"\n✓ Query executed successfully!")

    if hasattr(result, 'success') and result.success:
        print(f"  Success: {result.success}")

        if hasattr(result, 'df'):
            df = result.df
            print(f"\n3. Result DataFrame:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\n   First few rows:")
            print(df.head())

            # Check for scenario column
            if 'scenario' in df.columns:
                print(f"\n✓ 'scenario' column found!")
                print(f"   Unique scenario values: {df['scenario'].unique()}")
            else:
                print(f"\n✗ 'scenario' column NOT found")
                print(f"   Available columns: {list(df.columns)}")
        else:
            print(f"  No DataFrame in result")
    else:
        print(f"  Query failed")
        if hasattr(result, 'error'):
            print(f"  Error: {result.error}")
        print(f"  Result: {result}")

except Exception as e:
    print(f"\n✗ Query failed with error:")
    print(f"   {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
