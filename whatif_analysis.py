from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np

from ar_analytics import ArUtils, DataQuery
from skill_framework import (
    SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

# Hardcoded constants for COGS breakdown columns
PRICE_COL_MAPPING = {
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
    llm_name="COGS What-If Scenario Analysis",
    description="Analyze the impact of commodity cost changes on COGS (Cost of Goods Sold). Shows how changes in material costs (cocoa, sugar, wheat, other materials) or operational costs (labor, overheads, logistics) affect total COGS by category.",
    capabilities="COGS scenario analysis: Analyze impact of commodity price changes (cocoa, sugar, wheat, other materials) or operational cost changes (labor, overheads, logistics) on cost of goods sold by category. Shows forecasted vs estimated values with detailed breakdown by material and cost components.",
    limitations="Requires category as breakout dimension. Supports only COGS metric. Requires at least one cost component change in price_change_scenario.",
    example_questions="What would be the impact of a 5% increase in cocoa price on COGS? How would a 10% increase in labor costs affect COGS by category? What if sugar and wheat prices both increase by 3%?",
    parameter_guidance="IMPORTANT: Always use 'category' as breakout dimension for COGS analysis. Provide commodity or operational cost changes in price_change_scenario as JSON like {'cocoa': 0.05} for 5% increase in cocoa, or {'labor': 0.10, 'sugar': 0.03} for multiple changes. Values should be decimal percentages (0.05 = 5%).",
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time period for analysis (e.g., 'Q3 2024', 'Jul 2024 to Sep 2024')"
        ),
        SkillParameter(
            name="breakout",
            is_multi=False,
            constrained_to="dimensions",
            description="Breakout dimension - must be 'category' for COGS analysis",
            default_value="category"
        ),
        SkillParameter(
            name="price_change_scenario",
            parameter_type="json",
            description="JSON object with cost changes as decimal percentages. Keys: 'cocoa', 'sugar', 'wheat', 'other_materials' for commodities; 'material', 'labor', 'overheads', 'logistics' for cost components. Example: {'cocoa': 0.05, 'sugar': 0.03} for 5% cocoa and 3% sugar increase."
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters (region, brand, etc.)"
        )
    ]
)
def whatif_analysis(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")

    # Parse parameters
    periods = parameters.arguments.periods if hasattr(parameters.arguments, 'periods') else []
    breakout = parameters.arguments.breakout if hasattr(parameters.arguments, 'breakout') else 'category'
    other_filters = parameters.arguments.other_filters if hasattr(parameters.arguments, 'other_filters') else []

    # Force category as breakout for COGS
    if breakout.lower() != 'category':
        breakout = 'category'

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
                narrative="Error: Invalid price_change_scenario format. Use format like {'cocoa': 0.05} for 5% increase.",
                visualizations=[],
                parameter_display_descriptions=[]
            )

    if not price_scenario:
        return SkillOutput(
            final_prompt="No price changes specified.",
            narrative="Error: You must specify at least one cost change in price_change_scenario parameter.",
            visualizations=[],
            parameter_display_descriptions=[]
        )

    # Create analysis engine
    analyzer = WhatIfAnalysisEngine(
        periods=periods,
        breakout=breakout,
        filters=other_filters,
        price_scenario=price_scenario
    )

    # Run analysis
    try:
        results_df, chart_html = analyzer.run()
    except Exception as e:
        logger.error(f"Error running what-if analysis: {e}", exc_info=True)
        return SkillOutput(
            final_prompt=f"Error running analysis: {str(e)}",
            narrative=f"Error: {str(e)}",
            visualizations=[],
            parameter_display_descriptions=[]
        )

    # Create parameter display descriptions
    param_info = [
        ParameterDisplayDescription(key="Breakout", value=breakout),
        ParameterDisplayDescription(key="Period", value=", ".join(periods) if periods else "Not specified")
    ]

    for k, v in price_scenario.items():
        formatted_val = f"{v:+.1%}"
        param_info.append(ParameterDisplayDescription(key=k, value=formatted_val))

    # Create visualization
    viz = SkillVisualization(
        title="What-If COGS Analysis",
        layout=chart_html
    )

    # Generate insights using LLM
    ar_utils = ArUtils()
    insight_prompt = f"""
Analyze the following COGS what-if scenario results:

Scenario: {', '.join([f'{k}: {v:+.1%}' for k, v in price_scenario.items()])}
Breakout by: {breakout}

Results summary:
{results_df.to_string()}

Provide a brief analysis covering:
1. Overall COGS impact magnitude and direction across categories
2. Which categories are most/least affected and why
3. Which cost components drive the largest changes
4. Business implications and recommended actions

Use a professional finance tone. Be concise (3-4 sentences).
"""

    insights = ar_utils.get_llm_response(insight_prompt)

    return SkillOutput(
        final_prompt=insight_prompt,
        narrative=insights,
        visualizations=[viz],
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[
            ExportData(name="COGS What-If Analysis", data=results_df)
        ]
    )


class WhatIfAnalysisEngine:
    """Engine for running COGS what-if scenario analysis"""

    def __init__(self, periods, breakout, filters, price_scenario):
        self.periods = periods
        self.breakout = breakout
        self.filters = filters
        self.price_scenario = price_scenario

        self.forecasted_col = "Forecasted"
        self.estimated_col = "Estimated"
        self.change_col = "Change"

        # Initialize data query utility
        self.data_query = DataQuery()

    def run(self):
        """Run the COGS what-if analysis and return results DataFrame and chart HTML"""

        # Pull base COGS data from database
        base_df = self._pull_cogs_data()

        # Calculate COGS breakdown by category
        forecasted_df = self._calculate_category_breakouts_from_cogs(base_df)

        # Recalculate COGS with price changes
        estimated_df = self._recalculate_cogs(forecasted_df, self.price_scenario)

        # Merge and calculate changes
        results_df = self._merge_and_calculate_changes(forecasted_df, estimated_df)

        # Generate chart HTML
        chart_html = self._generate_chart_html(results_df)

        return results_df, chart_html

    def _pull_cogs_data(self):
        """Pull COGS data from database"""

        # Build query filters
        query_filters = []
        for f in self.filters:
            query_filters.append({
                "column": f.get("column") or f.get("col"),
                "operator": f.get("operator") or f.get("op", "="),
                "value": f.get("value") or f.get("val")
            })

        # Query COGS by category
        df = self.data_query.query(
            metrics=["cogs"],
            dimensions=[self.breakout],
            filters=query_filters,
            date_filter=self.periods[0] if self.periods else None
        )

        return df

    def _get_cogs_breakdown(self):
        """Get COGS breakdown percentages by category"""
        return {
            "Material": {"Biscuits": 0.65, "Chocolate": 0.62, "Snack Bars": 0.70, "Cakes and Pastries": 0.55},
            "Labor": {"Biscuits": 0.20, "Chocolate": 0.22, "Snack Bars": 0.18, "Cakes and Pastries": 0.25},
            "Overheads": {"Biscuits": 0.08, "Chocolate": 0.06, "Snack Bars": 0.05, "Cakes and Pastries": 0.09},
            "Logistics": {"Biscuits": 0.07, "Chocolate": 0.10, "Snack Bars": 0.07, "Cakes and Pastries": 0.11}
        }

    def _get_cogs_commodity_breakdown(self):
        """Get commodity breakdown percentages within materials by category"""
        return {
            "% of Sugar": {"Biscuits": 0.15, "Chocolate": 0.25, "Snack Bars": 0.10, "Cakes and Pastries": 0.30},
            "% of Cocoa": {"Biscuits": 0.20, "Chocolate": 0.40, "Snack Bars": 0.25, "Cakes and Pastries": 0.15},
            "% of Wheat": {"Biscuits": 0.30, "Chocolate": 0.00, "Snack Bars": 0.05, "Cakes and Pastries": 0.10},
            "% Others": {"Biscuits": 0.35, "Chocolate": 0.35, "Snack Bars": 0.60, "Cakes and Pastries": 0.45}
        }

    def _calculate_category_breakouts_from_cogs(self, df):
        """Calculate COGS breakdown by cost components for each category"""

        breakout_df = df.copy()

        # Get breakdown percentages
        cogs_breakdown = self._get_cogs_breakdown()
        commodity_breakdown = self._get_cogs_commodity_breakdown()

        # Calculate each cost component
        for component, category_pcts in cogs_breakdown.items():
            breakout_df[component] = breakout_df.apply(
                lambda row: row['cogs'] * category_pcts.get(row[self.breakout], 0),
                axis=1
            )

        # Calculate commodity breakdown within materials
        for commodity, category_pcts in commodity_breakdown.items():
            breakout_df[commodity] = breakout_df.apply(
                lambda row: row['Material'] * category_pcts.get(row[self.breakout], 0),
                axis=1
            )

        return breakout_df

    def _recalculate_cogs(self, df, price_changes):
        """Recalculate COGS with price changes applied"""

        estimated_df = df.copy()

        # Get commodity columns
        commodity_cols = ["% of Sugar", "% of Cocoa", "% of Wheat", "% Others"]
        cogs_component_cols = ["Material", "Labor", "Overheads", "Logistics"]

        # Apply commodity price changes first
        for commodity in commodity_cols:
            if commodity in price_changes:
                estimated_df[commodity] = estimated_df[commodity] * (1 + price_changes[commodity])

        # Recalculate Material as sum of commodities
        estimated_df["Material"] = estimated_df[commodity_cols].sum(axis=1)

        # Apply cost component price changes
        for component in cogs_component_cols:
            if component in price_changes and component != "Material":
                estimated_df[component] = estimated_df[component] * (1 + price_changes[component])

        # Recalculate total COGS
        estimated_df["cogs"] = estimated_df[cogs_component_cols].sum(axis=1)

        return estimated_df

    def _merge_and_calculate_changes(self, forecasted_df, estimated_df):
        """Merge forecasted and estimated, calculate changes"""

        # Create multi-index columns structure
        result_data = []

        for idx, row in forecasted_df.iterrows():
            category = row[self.breakout]
            est_row = estimated_df.iloc[idx]

            row_data = {self.breakout: category}

            # Add COGS columns
            row_data["COGS_Forecasted"] = row["cogs"]
            row_data["COGS_Estimated"] = est_row["cogs"]
            row_data["COGS_Change"] = (est_row["cogs"] - row["cogs"]) / row["cogs"] if row["cogs"] != 0 else 0

            # Add Material columns
            row_data["Material_Forecasted"] = row["Material"]
            row_data["Material_Estimated"] = est_row["Material"]
            row_data["Material_Change"] = (est_row["Material"] - row["Material"]) / row["Material"] if row["Material"] != 0 else 0

            # Add commodity columns
            for commodity in ["% of Sugar", "% of Cocoa", "% of Wheat", "% Others"]:
                col_name = commodity.replace("% of ", "").replace(" ", "_")
                row_data[f"{col_name}_Forecasted"] = row[commodity]
                row_data[f"{col_name}_Estimated"] = est_row[commodity]
                row_data[f"{col_name}_Change"] = (est_row[commodity] - row[commodity]) / row[commodity] if row[commodity] != 0 else 0

            result_data.append(row_data)

        return pd.DataFrame(result_data)

    def _generate_chart_html(self, df):
        """Generate Highcharts HTML for the visualization"""

        # Extract data for chart - only COGS for simplicity
        categories = df[self.breakout].tolist()
        forecasted_data = df["COGS_Forecasted"].tolist()
        estimated_data = df["COGS_Estimated"].tolist()

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
            z-index: 2;
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
            z-index: 1;
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
                    <th class="index_cols" rowspan="2">{self.breakout.title()}</th>
                    <th colspan="3" style="text-align: center;">COGS</th>
                    <th colspan="3" style="text-align: center;">Material</th>
                    <th colspan="3" style="text-align: center;">% of Cocoa</th>
                </tr>
                <tr>
                    <th>Forecasted</th>
                    <th>Estimated</th>
                    <th>Change</th>
                    <th>Forecasted</th>
                    <th>Estimated</th>
                    <th>Change</th>
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
                    <td>${row['COGS_Forecasted']/1000000:.2f}M</td>
                    <td>${row['COGS_Estimated']/1000000:.2f}M</td>
                    <td>{row['COGS_Change']:.2%}</td>
                    <td>${row['Material_Forecasted']/1000000:.2f}M</td>
                    <td>${row['Material_Estimated']/1000000:.2f}M</td>
                    <td>{row['Material_Change']:.2%}</td>
                    <td>${row['Cocoa_Forecasted']/1000000:.2f}M</td>
                    <td>${row['Cocoa_Estimated']/1000000:.2f}M</td>
                    <td>{row['Cocoa_Change']:.2%}</td>
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
                        text: '""" + self.breakout.title() + """'
                    }
                },
                yAxis: {
                    title: {
                        text: 'COGS'
                    },
                    labels: {
                        formatter: function() {
                            return '$' + (this.value / 1000000).toFixed(1) + 'M';
                        }
                    }
                },
                tooltip: {
                    formatter: function() {
                        return this.series.name + ': <b>$' + (this.y / 1000000).toFixed(2) + 'M</b>';
                    }
                },
                series: [{
                    name: 'COGS Forecasted',
                    data: """ + json.dumps(forecasted_data) + """,
                    color: '#5DADE2'
                }, {
                    name: 'COGS Estimated',
                    data: """ + json.dumps(estimated_data) + """,
                    color: '#8E44AD'
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
        'periods': ['Q3 2024'],
        'breakout': 'category',
        'price_change_scenario': {'cocoa': 0.05}
    })
    out = whatif_analysis(skill_input)
    print(out.narrative)
