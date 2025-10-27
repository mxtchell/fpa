from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
from skill_framework import (
    SkillInput,
    SkillVisualization,
    skill,
    SkillParameter,
    SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout
import jinja2
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_ID = "83C2268F-AF77-4D00-8A6B-7181DC06643E"
DATASET_ID = "d762aa87-6efb-47c4-b491-3bdc27147d4e"


# Default prompts
DEFAULT_MAX_PROMPT = """
Based on the following variance analysis facts:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide a concise executive summary (2-3 sentences) highlighting the most significant variance drivers.
"""

DEFAULT_INSIGHT_PROMPT = """
Analyze the following variance analysis data:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide detailed insights covering:
1. Key variance drivers (Price, Volume, Mix)
2. Top contributing dimensions
3. Actionable recommendations for stakeholders
4. Areas requiring immediate attention

Format the insights in clear markdown with bullet points.
"""


# Layout template for waterfall chart visualization
WATERFALL_CHART_LAYOUT = """
{
    "layoutJson": {
        "type": "Document",
        "rows": 90,
        "columns": 160,
        "rowHeight": "1.11%",
        "colWidth": "0.625%",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "20px"
        },
        "children": [
            {
                "name": "CardContainer0",
                "type": "CardContainer",
                "children": "",
                "minHeight": "80px",
                "rows": 2,
                "columns": 1,
                "style": {
                    "border-radius": "11.911px",
                    "background": "#2563EB",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "hidden": false
            },
            {
                "name": "Header0",
                "type": "Header",
                "children": "",
                "text": "Variance Analysis",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": "#ffffff",
                    "textAlign": "left",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "Paragraph0",
                "type": "Paragraph",
                "children": "",
                "text": "Price-Volume-Mix Decomposition",
                "style": {
                    "fontSize": "15px",
                    "fontWeight": "normal",
                    "textAlign": "center",
                    "verticalAlign": "start",
                    "color": "#fafafa",
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "CardContainer1",
                "type": "FlexContainer",
                "children": "",
                "direction": "column",
                "minHeight": "",
                "maxHeight": "",
                "style": {
                    "borderRadius": "11.911px",
                    "background": "var(--White, #FFF)",
                    "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "flexDirection": "row",
                "hidden": false
            },
            {
                "name": "Header1",
                "type": "Header",
                "children": "",
                "text": "Analysis Summary",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "textAlign": "left",
                    "verticalAlign": "start",
                    "color": "#000000",
                    "backgroundColor": "#ffffff",
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "borderBottom": "solid #DDD 2px"
                },
                "parentId": "CardContainer1",
                "flex": "",
                "hidden": false
            },
            {
                "name": "Markdown0",
                "type": "Markdown",
                "children": "",
                "text": "insights",
                "style": {
                    "color": "#555",
                    "backgroundColor": "#ffffff",
                    "border": "null",
                    "fontSize": "15px"
                },
                "parentId": "CardContainer1",
                "flex": "",
                "hidden": false
            },
            {
                "name": "FlexContainer5",
                "type": "FlexContainer",
                "minHeight": "450px",
                "direction": "row",
                "style": {
                    "maxWidth": "100%",
                    "width": "100%"
                }
            },
            {
                "name": "FlexContainer4",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "250px",
                "direction": "column",
                "maxHeight": "1200px"
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "450px",
                "chartOptions": {},
                "options": {
                    "chart": {
                        "type": "waterfall"
                    },
                    "title": {
                        "text": "",
                        "style": {
                            "fontSize": "18px",
                            "fontWeight": "bold"
                        }
                    },
                    "xAxis": {
                        "categories": [],
                        "title": {
                            "text": ""
                        }
                    },
                    "yAxis": {
                        "title": {
                            "text": ""
                        }
                    },
                    "series": [],
                    "credits": {
                        "enabled": false
                    },
                    "legend": {
                        "enabled": false
                    },
                    "tooltip": {
                        "pointFormat": "<b>{point.name}</b>: {point.formatted}"
                    }
                },
                "parentId": "FlexContainer5",
                "hidden": false
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [],
                "data": [],
                "parentId": "FlexContainer4",
                "caption": "",
                "styles": {
                    "td": {
                        "vertical-align": "middle"
                    }
                }
            }
        ]
    },
    "inputVariables": [
        {
            "name": "exec_summary",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Markdown0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "sub_headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Paragraph0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Header0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.xAxis.categories"
                }
            ]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.yAxis"
                }
            ]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.series"
                }
            ]
        },
        {
            "name": "chart_title",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.title.text"
                }
            ]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "data"
                }
            ]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "columns"
                }
            ]
        },
        {
            "name": "hide_chart",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "FlexContainer5",
                    "fieldName": "hidden"
                }
            ]
        }
    ]
}
"""

# Horizontal bar chart layout for dimensional breakouts
HORIZONTAL_BAR_LAYOUT = """
{
    "layoutJson": {
        "type": "Document",
        "rows": 90,
        "columns": 160,
        "rowHeight": "1.11%",
        "colWidth": "0.625%",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "20px"
        },
        "children": [
            {
                "name": "CardContainer0",
                "type": "CardContainer",
                "children": "",
                "minHeight": "80px",
                "rows": 2,
                "columns": 1,
                "style": {
                    "border-radius": "11.911px",
                    "background": "#2563EB",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "hidden": false
            },
            {
                "name": "Header0",
                "type": "Header",
                "children": "",
                "text": "Dimensional Breakout",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": "#ffffff",
                    "textAlign": "left",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "Paragraph0",
                "type": "Paragraph",
                "children": "",
                "text": "Variance by Dimension",
                "style": {
                    "fontSize": "15px",
                    "fontWeight": "normal",
                    "textAlign": "center",
                    "verticalAlign": "start",
                    "color": "#fafafa",
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "CardContainer1",
                "type": "FlexContainer",
                "children": "",
                "direction": "column",
                "minHeight": "",
                "maxHeight": "",
                "style": {
                    "borderRadius": "11.911px",
                    "background": "var(--White, #FFF)",
                    "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "flexDirection": "row",
                "hidden": false
            },
            {
                "name": "Header1",
                "type": "Header",
                "children": "",
                "text": "Analysis Summary",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "textAlign": "left",
                    "verticalAlign": "start",
                    "color": "#000000",
                    "backgroundColor": "#ffffff",
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "borderBottom": "solid #DDD 2px"
                },
                "parentId": "CardContainer1",
                "flex": "",
                "hidden": false
            },
            {
                "name": "Markdown0",
                "type": "Markdown",
                "children": "",
                "text": "insights",
                "style": {
                    "color": "#555",
                    "backgroundColor": "#ffffff",
                    "border": "null",
                    "fontSize": "15px"
                },
                "parentId": "CardContainer1",
                "flex": "",
                "hidden": false
            },
            {
                "name": "FlexContainer5",
                "type": "FlexContainer",
                "minHeight": "400px",
                "direction": "row",
                "style": {
                    "maxWidth": "100%",
                    "width": "100%"
                }
            },
            {
                "name": "FlexContainer4",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "250px",
                "direction": "column",
                "maxHeight": "1200px"
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "400px",
                "chartOptions": {},
                "options": {
                    "chart": {
                        "type": "bar"
                    },
                    "title": {
                        "text": "",
                        "style": {
                            "fontSize": "18px",
                            "fontWeight": "bold"
                        }
                    },
                    "xAxis": {
                        "categories": [],
                        "title": {
                            "text": ""
                        }
                    },
                    "yAxis": {
                        "title": {
                            "text": ""
                        }
                    },
                    "series": [],
                    "credits": {
                        "enabled": false
                    },
                    "legend": {
                        "enabled": true,
                        "align": "center",
                        "verticalAlign": "bottom",
                        "layout": "horizontal"
                    },
                    "plotOptions": {
                        "bar": {
                            "dataLabels": {
                                "enabled": false
                            }
                        }
                    },
                    "tooltip": {
                        "pointFormat": "<b>{series.name}</b>: {point.formatted}"
                    }
                },
                "parentId": "FlexContainer5",
                "hidden": false
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [],
                "data": [],
                "parentId": "FlexContainer4",
                "caption": "",
                "styles": {
                    "td": {
                        "vertical-align": "middle"
                    }
                }
            }
        ]
    },
    "inputVariables": [
        {
            "name": "exec_summary",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Markdown0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "sub_headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Paragraph0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Header0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.xAxis.categories"
                }
            ]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.yAxis"
                }
            ]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.series"
                }
            ]
        },
        {
            "name": "chart_title",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.title.text"
                }
            ]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "data"
                }
            ]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "columns"
                }
            ]
        },
        {
            "name": "hide_chart",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "FlexContainer5",
                    "fieldName": "hidden"
                }
            ]
        }
    ]
}
"""


def format_number(value, is_currency=True, decimals=1):
    """Format numbers with M/K/B abbreviations"""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return str(value)

    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{value / 1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        formatted = f"{value / 1_000:.{decimals}f}K"
    else:
        formatted = f"{value:.{decimals}f}"

    if is_currency:
        formatted = f"${formatted}"

    return formatted


class FPAVarianceAnalysis:
    """FP&A Variance Analysis with Price-Volume-Mix Decomposition"""

    def __init__(self, client, metric, period, comparison_type, breakout_dimensions=None,
                 top_n=10, other_filters=None):
        self.client = client
        self.metric = metric
        self.period = period
        self.comparison_type = comparison_type  # 'Budget', 'Forecast', 'Prior Period'
        self.breakout_dimensions = breakout_dimensions or []
        self.top_n = top_n
        self.other_filters = other_filters or []

        self.actuals_df = None
        self.comparison_df = None
        self.pvm_results = None
        self.breakout_results = {}
        self.facts = []

    def get_comparison_scenario(self):
        """Map comparison type to scenario value"""
        mapping = {
            'Budget': 'budget',
            'Forecast': 'forecast',
            'Prior Period': 'prior_period'
        }
        return mapping.get(self.comparison_type, 'budget')

    def build_filter_clause(self):
        """Build SQL WHERE clause from filters"""
        clauses = []

        if self.other_filters:
            for filter_dict in self.other_filters:
                dim = filter_dict.get('dim')
                op = filter_dict.get('op', '=')
                val = filter_dict.get('val')

                if dim and val:
                    if isinstance(val, str):
                        clauses.append(f"{dim} {op} '{val}'")
                    else:
                        clauses.append(f"{dim} {op} {val}")

        return " AND " + " AND ".join(clauses) if clauses else ""

    def query_data(self):
        """Query actuals and comparison data from database"""
        logger.info(f"Querying data for metric: {self.metric}, period: {self.period}")

        filter_clause = self.build_filter_clause()
        comparison_scenario = self.get_comparison_scenario()

        # Query actuals
        actuals_query = f"""
        SELECT *
        FROM dataset
        WHERE scenario = 'actuals'
        AND period = '{self.period}'
        {filter_clause}
        """

        logger.info(f"Actuals query: {actuals_query}")
        self.actuals_df = self.client.data.execute_sql_query(
            database_id=DATABASE_ID,
            query=actuals_query
        )

        # Query comparison data
        comparison_query = f"""
        SELECT *
        FROM dataset
        WHERE scenario = '{comparison_scenario}'
        AND period = '{self.period}'
        {filter_clause}
        """

        logger.info(f"Comparison query: {comparison_query}")
        self.comparison_df = self.client.data.execute_sql_query(
            database_id=DATABASE_ID,
            query=comparison_query
        )

        logger.info(f"Actuals shape: {self.actuals_df.shape if self.actuals_df is not None else 'None'}")
        logger.info(f"Comparison shape: {self.comparison_df.shape if self.comparison_df is not None else 'None'}")

    def calculate_price_volume_mix(self):
        """
        Calculate Price-Volume-Mix decomposition

        Formula:
        - Volume Impact = (Actual Volume - Budget Volume) * Budget Price
        - Price Impact = (Actual Price - Budget Price) * Actual Volume
        - Mix Impact = Residual
        """
        logger.info("Calculating Price-Volume-Mix decomposition")

        if self.actuals_df is None or self.comparison_df is None:
            logger.error("Data not loaded. Call query_data() first.")
            return None

        # Aggregate to get totals
        actual_revenue = self.actuals_df[self.metric].sum()
        actual_volume = self.actuals_df['volume'].sum() if 'volume' in self.actuals_df.columns else 0
        actual_price = actual_revenue / actual_volume if actual_volume > 0 else 0

        comparison_revenue = self.comparison_df[self.metric].sum()
        comparison_volume = self.comparison_df['volume'].sum() if 'volume' in self.comparison_df.columns else 0
        comparison_price = comparison_revenue / comparison_volume if comparison_volume > 0 else 0

        # Calculate impacts
        volume_impact = (actual_volume - comparison_volume) * comparison_price
        price_impact = (actual_price - comparison_price) * actual_volume

        total_variance = actual_revenue - comparison_revenue
        mix_impact = total_variance - volume_impact - price_impact

        self.pvm_results = {
            'starting_value': comparison_revenue,
            'volume_impact': volume_impact,
            'price_impact': price_impact,
            'mix_impact': mix_impact,
            'ending_value': actual_revenue,
            'total_variance': total_variance
        }

        # Create facts
        self.facts.append({
            'fact': f"Total variance: {format_number(total_variance)} ({total_variance/comparison_revenue*100:.1f}%)",
            'category': 'overall'
        })
        self.facts.append({
            'fact': f"Volume impact: {format_number(volume_impact)} ({volume_impact/abs(total_variance)*100:.1f}% of variance)",
            'category': 'pvm'
        })
        self.facts.append({
            'fact': f"Price impact: {format_number(price_impact)} ({price_impact/abs(total_variance)*100:.1f}% of variance)",
            'category': 'pvm'
        })
        self.facts.append({
            'fact': f"Mix impact: {format_number(mix_impact)} ({mix_impact/abs(total_variance)*100:.1f}% of variance)",
            'category': 'pvm'
        })

        logger.info(f"PVM Results: {self.pvm_results}")
        return self.pvm_results

    def calculate_dimensional_breakout(self, dimension):
        """Calculate variance attribution by dimension"""
        logger.info(f"Calculating breakout for dimension: {dimension}")

        if self.actuals_df is None or self.comparison_df is None:
            logger.error("Data not loaded. Call query_data() first.")
            return None

        # Merge actuals and comparison
        actuals_agg = self.actuals_df.groupby(dimension)[self.metric].sum().reset_index()
        actuals_agg.columns = [dimension, 'actual']

        comparison_agg = self.comparison_df.groupby(dimension)[self.metric].sum().reset_index()
        comparison_agg.columns = [dimension, 'comparison']

        merged = pd.merge(actuals_agg, comparison_agg, on=dimension, how='outer').fillna(0)

        # Calculate variance
        merged['variance'] = merged['actual'] - merged['comparison']
        merged['variance_pct'] = merged['variance'] / merged['comparison'] * 100

        # Rank by absolute variance
        merged['abs_variance'] = merged['variance'].abs()
        merged = merged.sort_values('abs_variance', ascending=False)

        # Take top N
        top_n_df = merged.head(self.top_n).copy()

        self.breakout_results[dimension] = top_n_df

        # Add facts for top contributors
        for idx, row in top_n_df.head(3).iterrows():
            self.facts.append({
                'fact': f"{dimension} '{row[dimension]}': {format_number(row['variance'])} variance ({row['variance_pct']:.1f}%)",
                'category': f'breakout_{dimension}'
            })

        logger.info(f"Breakout results for {dimension}: {top_n_df.shape}")
        return top_n_df

    def create_waterfall_chart_data(self):
        """Create Highcharts waterfall chart configuration"""
        if not self.pvm_results:
            return None

        categories = [
            f"{self.comparison_type}",
            "Volume Impact",
            "Price Impact",
            "Mix Impact",
            "Actuals"
        ]

        data = [
            {
                'name': f"{self.comparison_type}",
                'y': self.pvm_results['starting_value'],
                'formatted': format_number(self.pvm_results['starting_value']),
                'color': '#7CB5EC'
            },
            {
                'name': 'Volume Impact',
                'y': self.pvm_results['volume_impact'],
                'formatted': format_number(self.pvm_results['volume_impact']),
                'color': '#90ED7D' if self.pvm_results['volume_impact'] > 0 else '#F45B5B'
            },
            {
                'name': 'Price Impact',
                'y': self.pvm_results['price_impact'],
                'formatted': format_number(self.pvm_results['price_impact']),
                'color': '#90ED7D' if self.pvm_results['price_impact'] > 0 else '#F45B5B'
            },
            {
                'name': 'Mix Impact',
                'y': self.pvm_results['mix_impact'],
                'formatted': format_number(self.pvm_results['mix_impact']),
                'color': '#90ED7D' if self.pvm_results['mix_impact'] > 0 else '#F45B5B'
            },
            {
                'name': 'Actuals',
                'isSum': True,
                'formatted': format_number(self.pvm_results['ending_value']),
                'color': '#434348'
            }
        ]

        return {
            'chart_categories': categories,
            'chart_data': [{
                'name': self.metric,
                'data': data,
                'dataLabels': {
                    'enabled': True,
                    'formatter': "function() { return this.point.formatted; }"
                }
            }],
            'chart_y_axis': {
                'title': {'text': self.metric},
                'labels': {'format': '${value:,.0f}'}
            },
            'chart_title': f'Price-Volume-Mix Analysis: {self.metric}'
        }

    def create_horizontal_bar_chart_data(self, dimension):
        """Create Highcharts horizontal bar chart for dimension breakout"""
        if dimension not in self.breakout_results:
            return None

        df = self.breakout_results[dimension]

        categories = df[dimension].tolist()
        actual_data = []
        comparison_data = []

        for _, row in df.iterrows():
            actual_data.append({
                'name': row[dimension],
                'y': row['actual'],
                'formatted': format_number(row['actual'])
            })
            comparison_data.append({
                'name': row[dimension],
                'y': row['comparison'],
                'formatted': format_number(row['comparison'])
            })

        return {
            'chart_categories': categories,
            'chart_data': [
                {
                    'name': 'Actuals',
                    'data': actual_data,
                    'color': '#5DADE2'
                },
                {
                    'name': self.comparison_type,
                    'data': comparison_data,
                    'color': '#F8C471'
                }
            ],
            'chart_y_axis': {
                'title': {'text': self.metric},
                'labels': {'format': '${value:,.0f}'}
            },
            'chart_title': f'{dimension} Variance Analysis'
        }

    def get_summary_table(self):
        """Create summary table for PVM decomposition"""
        if not self.pvm_results:
            return None

        data = [
            [f"{self.comparison_type}", format_number(self.pvm_results['starting_value'])],
            ["Volume Impact", format_number(self.pvm_results['volume_impact'])],
            ["Price Impact", format_number(self.pvm_results['price_impact'])],
            ["Mix Impact", format_number(self.pvm_results['mix_impact'])],
            ["Total Variance", format_number(self.pvm_results['total_variance'])],
            ["Actuals", format_number(self.pvm_results['ending_value'])]
        ]

        columns = [
            {'name': 'Component'},
            {'name': 'Value'}
        ]

        return {'data': data, 'col_defs': columns}

    def get_breakout_table(self, dimension):
        """Create variance table for dimension breakout"""
        if dimension not in self.breakout_results:
            return None

        df = self.breakout_results[dimension]

        data = []
        for _, row in df.iterrows():
            data.append([
                row[dimension],
                format_number(row['actual']),
                format_number(row['comparison']),
                format_number(row['variance']),
                f"{row['variance_pct']:.1f}%"
            ])

        columns = [
            {'name': dimension},
            {'name': 'Actuals'},
            {'name': self.comparison_type},
            {'name': 'Variance'},
            {'name': 'Variance %'}
        ]

        return {'data': data, 'col_defs': columns}

    def run_analysis(self):
        """Run complete variance analysis"""
        logger.info("Starting FPA variance analysis")

        # Query data
        self.query_data()

        # Calculate PVM
        self.calculate_price_volume_mix()

        # Calculate dimensional breakouts
        for dim in self.breakout_dimensions:
            self.calculate_dimensional_breakout(dim)

        logger.info("Analysis complete")
        return self


@skill(
    name="FP&A Drivers",
    llm_name="Metric Drivers with Price-Volume-Mix Decomposition",
    description="Analyze variance drivers using Price-Volume-Mix decomposition with waterfall charts and dimensional breakouts. Compare actuals vs Budget, Forecast, or Prior Period.",
    capabilities=[
        "Price-Volume-Mix variance decomposition",
        "Waterfall chart visualization of variance components",
        "Dimensional breakout analysis with horizontal bar charts",
        "Top contributor identification and ranking",
        "Multi-dimensional variance attribution",
        "Comparison vs Budget, Forecast, or Prior Period"
    ],
    limitations=[
        "Requires 'scenario' column in dataset with values: actuals, budget, forecast",
        "Requires 'volume' column for accurate PVM decomposition",
        "Limited to configured metrics and dimensions",
        "Maximum 10 dimensions for breakout analysis"
    ],
    example_questions=[
        "What are the revenue drivers for Q3 2024 vs budget?",
        "Show me price-volume-mix analysis for sales vs forecast",
        "Which regions contributed most to the revenue variance?",
        "Analyze variance drivers by category and customer type",
        "What caused the variance in Q4 vs prior period?"
    ],
    parameter_guidance={
        "metric": "Select the metric to analyze (e.g., Revenue, Profit, Units)",
        "period": "Time period for analysis (e.g., Q3 2024, 2024, Jan 2024)",
        "comparison_type": "Choose Budget, Forecast, or Prior Period for comparison",
        "breakout_dimensions": "Select dimensions for detailed breakout (e.g., Region, Category, Customer)",
        "top_n": "Number of top contributors to display (default 10)",
        "other_filters": "Additional filters to apply (e.g., Region = North, Product = Electronics)"
    },
    parameters=[
        SkillParameter(
            name="metric",
            constrained_to="metrics",
            is_multi=False,
            description="Metric to analyze (e.g., Revenue, Profit)"
        ),
        SkillParameter(
            name="period",
            constrained_to="date_filter",
            is_multi=False,
            description="Time period in format 'Q3 2024', '2024', 'Jan 2024', etc."
        ),
        SkillParameter(
            name="comparison_type",
            constrained_to=None,
            constrained_values=["Budget", "Forecast", "Prior Period"],
            description="Comparison type: Budget, Forecast, or Prior Period",
            default_value="Budget"
        ),
        SkillParameter(
            name="breakout_dimensions",
            constrained_to="dimensions",
            is_multi=True,
            description="Dimensions for breakout analysis (e.g., Region, Category, Customer Type)"
        ),
        SkillParameter(
            name="top_n",
            description="Number of top contributors to display",
            default_value=10
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters to apply to the analysis"
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for executive summary",
            default_value=DEFAULT_MAX_PROMPT
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights",
            default_value=DEFAULT_INSIGHT_PROMPT
        )
    ]
)
def metric_drivers(parameters: SkillInput):
    """Execute FP&A Variance Analysis with Price-Volume-Mix decomposition"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    metric = getattr(parameters.arguments, 'metric', None)
    period = getattr(parameters.arguments, 'period', None)
    comparison_type = getattr(parameters.arguments, 'comparison_type', 'Budget')
    breakout_dimensions = getattr(parameters.arguments, 'breakout_dimensions', [])
    top_n = getattr(parameters.arguments, 'top_n', 10)
    other_filters = getattr(parameters.arguments, 'other_filters', [])
    max_prompt = getattr(parameters.arguments, 'max_prompt', DEFAULT_MAX_PROMPT)
    insight_prompt = getattr(parameters.arguments, 'insight_prompt', DEFAULT_INSIGHT_PROMPT)

    # Get AnswerRocketClient
    try:
        from ar_analytics import ArUtils
        ar_utils = ArUtils()
        client = ar_utils.sp
    except Exception as e:
        logger.error(f"Failed to initialize AnswerRocketClient: {e}")
        # Create mock client for testing
        client = SimpleNamespace(
            data=SimpleNamespace(
                execute_sql_query=lambda database_id, query: pd.DataFrame()
            )
        )

    # Run analysis
    analysis = FPAVarianceAnalysis(
        client=client,
        metric=metric,
        period=period,
        comparison_type=comparison_type,
        breakout_dimensions=breakout_dimensions,
        top_n=top_n,
        other_filters=other_filters
    )

    analysis.run_analysis()

    # Generate insights
    facts_list = [pd.DataFrame(analysis.facts)]
    insight_template = jinja2.Template(insight_prompt).render(facts=[facts_list])
    max_response_prompt = jinja2.Template(max_prompt).render(facts=[facts_list])

    try:
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "Variance analysis complete. Review the waterfall chart and dimensional breakouts for detailed insights."

    # Create visualizations
    viz_list = []
    export_data = {}

    # Tab 1: Waterfall Chart + Summary Table
    waterfall_data = analysis.create_waterfall_chart_data()
    summary_table = analysis.get_summary_table()

    if waterfall_data and summary_table:
        general_vars = {
            "headline": f"{metric} Variance Analysis",
            "sub_headline": f"{period} | Actuals vs {comparison_type}",
            "exec_summary": insights,
            "hide_chart": False
        }

        layout_vars = {**general_vars, **waterfall_data, **summary_table}
        rendered = wire_layout(json.loads(WATERFALL_CHART_LAYOUT), layout_vars)
        viz_list.append(SkillVisualization(title="Price-Volume-Mix Analysis", layout=rendered))
        export_data["PVM_Summary"] = pd.DataFrame(summary_table['data'], columns=['Component', 'Value'])

    # Tab 2+: Horizontal Bar Charts for each dimension
    for dimension in breakout_dimensions:
        bar_data = analysis.create_horizontal_bar_chart_data(dimension)
        table_data = analysis.get_breakout_table(dimension)

        if bar_data and table_data:
            general_vars = {
                "headline": f"{dimension} Breakout",
                "sub_headline": f"Top {top_n} Contributors to Variance",
                "exec_summary": "",
                "hide_chart": False
            }

            layout_vars = {**general_vars, **bar_data, **table_data}
            rendered = wire_layout(json.loads(HORIZONTAL_BAR_LAYOUT), layout_vars)
            viz_list.append(SkillVisualization(title=f"{dimension} Analysis", layout=rendered))
            export_data[f"{dimension}_Variance"] = analysis.breakout_results[dimension]

    # Create parameter display
    param_info = [
        ParameterDisplayDescription(key="Metric", value=metric),
        ParameterDisplayDescription(key="Period", value=period),
        ParameterDisplayDescription(key="Comparison", value=comparison_type),
        ParameterDisplayDescription(key="Dimensions", value=", ".join(breakout_dimensions) if breakout_dimensions else "None")
    ]

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=insights,
        visualizations=viz_list,
        parameter_display_descriptions=param_info,
        followup_questions=[
            f"Which {breakout_dimensions[0] if breakout_dimensions else 'dimensions'} had the highest variance?",
            "What drove the price impact?",
            "How does volume variance compare across regions?",
            f"Show me variance trends over time vs {comparison_type.lower()}"
        ],
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )
