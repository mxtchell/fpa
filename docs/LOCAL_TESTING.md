# Local Skill Testing Guide

## 1. Activate Python Environment

**USE THIS ENVIRONMENT** - it has all dependencies installed:

```bash
source /Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate
```

## 2. Set Up Connection

Create a `.env` file in your skill repo:

```bash
# UPDATE THESE VALUES for your target environment
AR_BASE_URL=https://YOUR-ENV.answerrocket.com
AR_API_KEY=YOUR-TOKEN-HERE
```

**To get these values:**
- `AR_BASE_URL`: Your AnswerRocket environment URL (e.g., `https://maxdemo.staging.answerrocket.com`)
- `AR_API_KEY`: Generate from AnswerRocket UI → Settings → API Tokens

## 3. Connect and View Dataset

```python
from dotenv import load_dotenv
load_dotenv()

from answer_rocket import AnswerRocketClient
import os

# Connect to environment
client = AnswerRocketClient(
    url=os.environ['AR_BASE_URL'],
    token=os.environ['AR_API_KEY']
)
print(f"Connected to: {os.environ['AR_BASE_URL']}")

# List available databases
databases = client.data.get_databases()
for db in databases:
    print(f"Database: {db.name} (ID: {db.id})")

# Query a dataset (replace DATABASE_ID with actual ID)
DATABASE_ID = "your-database-id-here"

# Show tables
result = client.data.execute_sql_query(
    database_id=DATABASE_ID,
    sql_query="SHOW TABLES",
    row_limit=100
)
if result.success:
    print("Available tables:")
    print(result.df)

# Sample data
result = client.data.execute_sql_query(
    database_id=DATABASE_ID,
    sql_query="SELECT * FROM your_table LIMIT 10",
    row_limit=10
)
if result.success:
    print(f"Columns: {list(result.df.columns)}")
    print(result.df)
```

## 4. Test a Skill

```bash
cd /Users/mitchelltravis/cursor/projects/fpa
source /Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate

python -c "
from dotenv import load_dotenv
load_dotenv()

from metric_drivers import metric_drivers
from skill_framework import SkillInput

skill_input = metric_drivers.create_input(arguments={
    'metric': 'gross_revenue',
    'period': 'Jan 2025',
    'comparison_type': 'Budget',
    'breakout_dimensions': ['category'],
    'top_n': 10
})

output = metric_drivers(skill_input)
print(f'Visualizations: {len(output.visualizations)}')
print('SUCCESS!')
"
```

## Running with Preview (Browser)

```python
from dotenv import load_dotenv
load_dotenv()

from metric_drivers import metric_drivers
from skill_framework import SkillInput
from skill_framework.preview import preview_skill

skill_input = metric_drivers.create_input(arguments={
    'metric': 'gross_revenue',
    'period': 'Jan 2025',
    'comparison_type': 'Budget',
    'breakout_dimensions': ['category'],
    'top_n': 10
})

output = metric_drivers(skill_input)
preview_skill(metric_drivers, output)  # Opens in browser
```

## Running pytest

```bash
cd /Users/mitchelltravis/cursor/projects/fpa
source /Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate

pytest tests/ -v
```

## Environments Reference

| Repo | Environment Path |
|------|-----------------|
| fpa | `/Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate` |
| genpact_v2 | `/Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate` |

## Troubleshooting

### "No module named 'ar_analytics'"
You're not using the right environment. Run:
```bash
source /Users/mitchelltravis/template_skills_v1/.direnv/python-3.10/bin/activate
```

### "ImportError: cannot import name 'RunMaxSqlGenResponse'"
Wrong environment - that's a version mismatch. Use template_skills_v1 env.

### Token/Auth errors
Check your `.env` file has valid `AR_BASE_URL` and `AR_API_KEY`. Tokens can expire - regenerate if needed.
