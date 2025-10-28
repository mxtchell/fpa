from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from types import SimpleNamespace

from ar_analytics import ArUtils
from skill_framework import (
    SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

# Hardcoded constants for COGS breakdown columns
PRICE_COL_MAPPING = {
    "price": "Price",
    "marketing_spend": "Marketing Spend",
    "material": "Material",
    "labor": "Labor",
    "overheads": "Overheads",
    "logistics": "Logistics",
    "sugar": "% of Sugar",
    "cocoa": "% of Cocoa",
    "wheat": "% of Wheat",
    "other_materials": "% Others"
}


@skill(
    name="FP&A What-If Analysis",
    llm_name="What-If Scenario Analysis for COGS, Market Share, and Marketing",
    description="Analyze the impact of price changes, commodity cost changes, or marketing spend changes on business metrics. Shows forecasted vs estimated values with percentage changes.",
    capabilities="Three scenario types: 1) COGS - analyze impact of commodity price changes (cocoa, sugar, wheat, etc.) on cost of goods sold by category. 2) Market Share - analyze impact of price changes on market share by region or category. 3) Marketing Campaign - analyze impact of marketing spend changes on revenue by brand or region.",
    limitations="Requires specific breakouts and filters based on scenario type. COGS requires category breakout. Market share requires category/region breakout with corresponding filter. Marketing requires brand/region breakout with corresponding filter.",
    example_questions="What would be the impact of a 5% increase in cocoa price on COGS? How would a 10% price increase affect our market share in EMEA for Biscuits? What if we increase marketing spend by 20% for Brand A?",
    parameter_guidance="Select the metric to analyze (gross_revenue, cogs, market_share, etc.). Specify the period. For COGS scenarios, provide commodity price changes like {'cocoa': 0.05} for 5% increase. For market share, provide price changes like {'price': 0.10} for 10% increase. For marketing, provide {'marketing_spend': 0.20} for 20% increase. Add appropriate filters and breakouts based on scenario type.",
    parameters=[
        SkillParameter(
            name="metric",
            constrained_to="metrics",
            is_multi=False,
            description="Metric to analyze (e.g., cogs, gross_revenue, market_share)"
        ),
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time period for analysis (e.g., 'Q3 2024')"
        ),
        SkillParameter(
            name="breakout",
            is_multi=False,
            constrained_to="dimensions",
            description="Dimension to break out by (category for COGS, category/region for market share, brand/region for marketing)"
        ),
        SkillParameter(
            name="price_change_scenario",
            parameter_type="json",
            description="JSON object with price changes, e.g., {'cocoa': 0.05, 'sugar': 0.03} for COGS, {'price': 0.10} for market share, {'marketing_spend': 0.20} for marketing"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters (category, region, brand, etc.)"
        )
    ]
)
def whatif_analysis(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")

    # Parse parameters
    metric = parameters.arguments.metric
    periods = parameters.arguments.periods if hasattr(parameters.arguments, 'periods') else []
    breakout = parameters.arguments.breakout if hasattr(parameters.arguments, 'breakout') else None
    other_filters = parameters.arguments.other_filters if hasattr(parameters.arguments, 'other_filters') else []

    # Parse price change scenario
    price_scenario = {}
    if hasattr(parameters.arguments, 'price_change_scenario') and parameters.arguments.price_change_scenario:
        try:
            if isinstance(parameters.arguments.price_change_scenario, dict):
                price_scenario = parameters.arguments.price_change_scenario
            else:
                price_scenario = json.loads(parameters.arguments.price_change_scenario)
            # Map to display names
            price_scenario = {PRICE_COL_MAPPING.get(k, k): float(v) for k, v in price_scenario.items()}
        except Exception as e:
            logger.error(f"Error parsing price scenario: {e}")
            return SkillOutput(
                final_prompt="Error parsing price_change_scenario parameter. Must be valid JSON.",
                narrative="Error: Invalid price_change_scenario format",
                visualizations=[],
                parameter_display_descriptions=[]
            )

    # Create analysis engine
    analyzer = WhatIfAnalysisEngine(
        metric=metric,
        periods=periods,
        breakout=breakout,
        filters=other_filters,
        price_scenario=price_scenario
    )

    # Run analysis
    results_df, chart_html = analyzer.run()

    # Create parameter display descriptions
    param_info = [
        ParameterDisplayDescription(key="Metric", value=metric),
        ParameterDisplayDescription(key="Breakout", value=breakout or "None"),
        ParameterDisplayDescription(key="Period", value=", ".join(periods) if periods else "Not specified")
    ]

    for k, v in price_scenario.items():
        formatted_val = f"{v:+.1%}"
        param_info.append(ParameterDisplayDescription(key=k, value=formatted_val))

    # Create visualization
    viz = SkillVisualization(
        title="What-If Analysis",
        layout=chart_html
    )

    # Generate insights using LLM
    ar_utils = ArUtils()
    insight_prompt = f"""
Analyze the following what-if scenario results:

Scenario: {', '.join([f'{k}: {v:+.1%}' for k, v in price_scenario.items()])}
Metric: {metric}
Breakout by: {breakout}

Results summary:
{results_df.to_string()}

Provide a brief analysis covering:
1. Overall impact magnitude and direction
2. Which breakout values are most/least affected
3. Business implications of these changes
4. Recommended actions or considerations

Use a professional finance tone. Be concise.
"""

    insights = ar_utils.get_llm_response(insight_prompt)

    return SkillOutput(
        final_prompt=insight_prompt,
        narrative=insights,
        visualizations=[viz],
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[
            ExportData(name="What-If Analysis Results", data=results_df)
        ]
    )


class WhatIfAnalysisEngine:
    """Engine for running what-if scenario analysis"""

    def __init__(self, metric, periods, breakout, filters, price_scenario):
        self.metric = metric
        self.periods = periods
        self.breakout = breakout
        self.filters = filters
        self.price_scenario = price_scenario

        self.forecasted_col = "Forecasted"
        self.estimated_col = "Estimated"
        self.change_col = "Change"

    def run(self):
        """Run the what-if analysis and return results DataFrame and chart HTML"""

        # For now, create mock data based on the scenario type
        # In production, this would query the actual database

        scenario_type = self._detect_scenario_type()

        if scenario_type == "cogs":
            results_df = self._run_cogs_scenario()
        elif scenario_type == "market_share":
            results_df = self._run_market_share_scenario()
        else:  # marketing
            results_df = self._run_marketing_scenario()

        # Generate chart HTML
        chart_html = self._generate_chart_html(results_df)

        return results_df, chart_html

    def _detect_scenario_type(self):
        """Detect which scenario type based on price_scenario keys"""
        keys = set(self.price_scenario.keys())

        cogs_keys = {"Material", "Labor", "Overheads", "Logistics",
                     "% of Sugar", "% of Cocoa", "% of Wheat", "% Others"}

        if "Price" in keys:
            return "market_share"
        elif "Marketing Spend" in keys:
            return "marketing"
        elif keys.intersection(cogs_keys):
            return "cogs"
        else:
            return "cogs"  # default

    def _run_cogs_scenario(self):
        """Run COGS commodity price change scenario"""

        # Mock data for demonstration - in production, query from database
        categories = ["Snack Bars", "Biscuits", "Cakes and Pastries", "Chocolate"]

        # Base COGS values (forecasted)
        base_cogs = {
            "Snack Bars": 1640.74,
            "Biscuits": 4441.94,
            "Cakes and Pastries": 1634.33,
            "Chocolate": 3289.79
        }

        # COGS breakdown percentages
        cogs_breakdown = {
            "Material": {"Snack Bars": 0.70, "Biscuits": 0.65, "Cakes and Pastries": 0.55, "Chocolate": 0.62},
            "Labor": {"Snack Bars": 0.18, "Biscuits": 0.20, "Cakes and Pastries": 0.25, "Chocolate": 0.22},
            "Overheads": {"Snack Bars": 0.05, "Biscuits": 0.08, "Cakes and Pastries": 0.09, "Chocolate": 0.06},
            "Logistics": {"Snack Bars": 0.07, "Biscuits": 0.07, "Cakes and Pastries": 0.11, "Chocolate": 0.10}
        }

        # Commodity breakdown within materials
        commodity_breakdown = {
            "% of Sugar": {"Snack Bars": 0.10, "Biscuits": 0.15, "Cakes and Pastries": 0.30, "Chocolate": 0.25},
            "% of Cocoa": {"Snack Bars": 0.25, "Biscuits": 0.20, "Cakes and Pastries": 0.15, "Chocolate": 0.40},
            "% of Wheat": {"Snack Bars": 0.05, "Biscuits": 0.30, "Cakes and Pastries": 0.10, "Chocolate": 0.00},
            "% Others": {"Snack Bars": 0.60, "Biscuits": 0.35, "Cakes and Pastries": 0.45, "Chocolate": 0.35}
        }

        # Calculate estimated values with price changes
        data = []
        for category in categories:
            base_val = base_cogs[category]
            material_pct = cogs_breakdown["Material"][category]
            material_base = base_val * material_pct

            # Apply commodity price changes to material cost
            material_new = material_base
            for commodity, multiplier in self.price_scenario.items():
                if commodity in commodity_breakdown:
                    commodity_pct = commodity_breakdown[commodity][category]
                    material_new += material_base * commodity_pct * multiplier

            # Apply other COGS component changes
            other_components_base = base_val * (1 - material_pct)
            other_components_new = other_components_base
            for component, multiplier in self.price_scenario.items():
                if component in cogs_breakdown:
                    component_pct = cogs_breakdown[component][category]
                    other_components_new += base_val * component_pct * multiplier

            estimated_val = material_new + other_components_new
            change_pct = (estimated_val - base_val) / base_val

            data.append({
                self.breakout: category,
                "COGS_Forecasted": base_val,
                "COGS_Estimated": estimated_val,
                "COGS_Change": change_pct
            })

        return pd.DataFrame(data)

    def _run_market_share_scenario(self):
        """Run market share price change scenario"""
        # Simplified mock implementation
        return pd.DataFrame()

    def _run_marketing_scenario(self):
        """Run marketing campaign effectiveness scenario"""
        # Simplified mock implementation
        return pd.DataFrame()

    def _generate_chart_html(self, df):
        """Generate Highcharts HTML for the visualization"""

        # Get the metric columns
        metric_cols = [col for col in df.columns if col != self.breakout]
        base_metric = metric_cols[0].split('_')[0]

        # Extract data for chart
        categories = df[self.breakout].tolist()
        forecasted_data = df[f"{base_metric}_Forecasted"].tolist()
        estimated_data = df[f"{base_metric}_Estimated"].tolist()

        html = f"""
<head>
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <style>
        #whatif-chart {{
            height: 400px;
            width: 100%;
        }}

        .table-container {{
            overflow-x: auto;
            max-height: 600px;
            margin-top: 40px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 10pt;
        }}

        thead {{
            background-color: #EEE;
            position: sticky;
            top: 0;
            z-index: 1;
        }}

        th, td {{
            text-align: right;
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tbody tr:nth-child(odd) {{
            background-color: #f9f9f9;
        }}

        .index_cols {{
            position: sticky;
            left: 0;
            background-color: #EEE;
            font-weight: bold;
        }}

        tbody tr:nth-child(odd) .index_cols {{
            background-color: #f9f9f9;
        }}

        tbody tr:nth-child(even) .index_cols {{
            background-color: #ffffff;
        }}
    </style>
</head>
<body>
    <div id="whatif-chart"></div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th class="index_cols">{self.breakout}</th>
                    <th>Forecasted</th>
                    <th>Estimated</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td class="index_cols">{row[self.breakout]}</td>
                    <td>${row[f'{base_metric}_Forecasted']:.2f}M</td>
                    <td>${row[f'{base_metric}_Estimated']:.2f}M</td>
                    <td>{row[f'{base_metric}_Change']:.2%}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            Highcharts.chart('whatif-chart', {
                chart: {
                    type: 'column'
                },
                title: {
                    text: ''
                },
                xAxis: {
                    categories: """ + json.dumps(categories) + """,
                    title: {
                        text: '""" + self.breakout + """'
                    }
                },
                yAxis: {
                    title: {
                        text: '""" + base_metric + """'
                    }
                },
                series: [{
                    name: 'Forecasted',
                    data: """ + json.dumps(forecasted_data) + """
                }, {
                    name: 'Estimated',
                    data: """ + json.dumps(estimated_data) + """
                }],
                credits: {
                    enabled: false
                }
            });
        });
    </script>
</body>
"""

        return html


if __name__ == '__main__':
    skill_input: SkillInput = whatif_analysis.create_input(arguments={
        'metric': 'cogs',
        'periods': ['Q3 2024'],
        'breakout': 'category',
        'price_change_scenario': {'cocoa': 0.05}
    })
    out = whatif_analysis(skill_input)
    print(out.narrative)
