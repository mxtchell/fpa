from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import jinja2
from ar_analytics import AdvanceTrend, TrendTemplateParameterSetup, ArUtils
from ar_analytics.defaults import trend_analysis_config, default_trend_chart_layout, default_table_layout, \
    get_table_layout_vars, default_ppt_trend_chart_layout, default_ppt_table_layout
from skill_framework import SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput, \
    ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData

RUNNING_LOCALLY = False

logger = logging.getLogger(__name__)


@skill(
    name="FP&A Trend",
    llm_name="Trend Analysis with Budget and Forecast Comparison",
    description="Analyze trends over time comparing actuals vs budget and forecast. Breakout by scenario dimension to show actuals, budget, and forecast on same chart. When comparing to plan/budget, speak like a finance executive: beating plan is positive, missing forecast indicates challenges.",
    capabilities="Multi-period trend analysis. Budget vs Actuals comparison. Forecast vs Actuals comparison. Show actuals, budget, and forecast by using scenario breakout. Period-over-period and year-over-year growth. Dimensional breakouts. Smart currency formatting ($M for large metrics). Finance-focused narrative when comparing to plan.",
    limitations="Requires 'scenario' column in dataset with values: actuals, budget, forecast. Limited to configured metrics and dimensions. Maximum 10 breakout values per dimension.",
    example_questions="Did we hit plan in Jan 2025? Show gross revenue trend by scenario from Sep 2024 to Jan 2025. What is the trend for net revenue compared to budget? Compare revenue actuals to forecast over the last 6 months. Show brand contribution margin by scenario. Are we on track to hit our forecast?",
    parameter_guidance="IMPORTANT: When user asks about 'hitting plan', 'vs budget', 'vs forecast', or similar comparisons, ALWAYS use 'scenario' as a breakout dimension. This shows actuals, budget, and forecast as separate lines on the same chart. Select metric(s) to analyze (e.g., gross_revenue, net_revenue). Specify time periods (e.g., 'Jan 2025', 'Q4 2024', 'Sep 2024 to Jan 2025'). Add other breakout dimensions for detailed views if needed. Apply filters as needed. Speak like a finance executive: if actuals exceed budget, that's positive (we beat plan). If actuals fall short of forecast, that indicates business challenges or missed expectations.",
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time periods in format 'q2 2023', '2021', 'jan 2023', 'mat nov 2022', 'ytd q4 2022', 'ytd 2023', etc."
        ),
        SkillParameter(
            name="metrics",
            is_multi=True,
            constrained_to="metrics",
            description="Metrics to analyze (e.g., gross_revenue, net_revenue, brand_contribution_margin)"
        ),
        SkillParameter(
            name="limit_n",
            description="Limit the number of breakout values",
            default_value=10
        ),
        SkillParameter(
            name="breakouts",
            is_multi=True,
            constrained_to="dimensions",
            description="Breakout dimension(s) for analysis. Use 'scenario' to show actuals/budget/forecast on same chart."
        ),
        SkillParameter(
            name="time_granularity",
            is_multi=False,
            constrained_to="date_dimensions",
            description="Time granularity (month, quarter, year). Only add if explicitly stated by user."
        ),
        SkillParameter(
            name="growth_type",
            constrained_to=None,
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth type: Y/Y (year-over-year), P/P (period-over-period), or None"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters to apply"
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for max response",
            default_value=trend_analysis_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights",
            default_value="""
Analyze the following FP&A trend data with a finance executive perspective:

{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide insights covering:
1. **Performance vs Plan**: Did we beat or miss budget/forecast? By how much and what percentage?
2. **Trend Direction**: Are we trending up or down? Is the trajectory improving or declining?
3. **Key Observations**: Notable inflection points, variance drivers, or patterns
4. **Business Implications**: What does this mean for the business? If we exceeded budget, that's positive. If we missed forecast, that indicates challenges or unmet expectations.
5. **Recommendations**: What actions should stakeholders consider?

Use a professional finance tone. Be direct about performance - good or bad. Format in clear markdown with headers and bullet points.
"""
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="chart_viz_layout",
            parameter_type="visualization",
            description="Chart Viz Layout",
            default_value=default_trend_chart_layout
        ),
        SkillParameter(
            name="chart_ppt_layout",
            parameter_type="visualization",
            description="Chart slide Viz Layout",
            default_value=default_ppt_trend_chart_layout
        ),
        SkillParameter(
            name="table_ppt_export_viz_layout",
            parameter_type="visualization",
            description="Table slide Viz Layout",
            default_value=default_ppt_table_layout
        )
    ]
)
def trend(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")
    param_dict = {
        "periods": [],
        "metrics": None,
        "limit_n": 10,
        "breakouts": [],
        "growth_type": None,
        "other_filters": [],
        "time_granularity": None
    }

    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    env = SimpleNamespace(**param_dict)
    TrendTemplateParameterSetup(env=env)

    # Use standard AdvanceTrend
    env.trend = AdvanceTrend.from_env(env=env)
    df = env.trend.run_from_env()

    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.trend.paramater_display_infomation.items()]
    tables = [env.trend.display_dfs.get("Metrics Table")]

    insights_dfs = [env.trend.df_notes, env.trend.facts, env.trend.top_facts, env.trend.bottom_facts]

    charts = env.trend.get_dynamic_layout_chart_vars()

    # Apply custom formatting to charts
    charts = apply_fpa_formatting(charts)

    viz, slides, insights, final_prompt = render_layout(
        charts,
        tables,
        env.trend.title,
        env.trend.subtitle,
        insights_dfs,
        env.trend.warning_message,
        parameters.arguments.max_prompt,
        parameters.arguments.insight_prompt,
        parameters.arguments.table_viz_layout,
        parameters.arguments.chart_viz_layout,
        parameters.arguments.chart_ppt_layout,
        parameters.arguments.table_ppt_export_viz_layout
    )

    display_charts = env.trend.display_charts

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        ppt_slides=slides,
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[
            ExportData(name="Metrics Table", data=tables[0]),
            *[ExportData(name=chart, data=display_charts[chart].get("df")) for chart in display_charts.keys()]
        ]
    )


def apply_fpa_formatting(charts):
    """Apply FP&A currency formatting to chart configurations"""
    # Metrics that should be formatted in millions
    large_currency_metrics = ['gross_revenue', 'net_revenue', 'brand_contribution_margin', 'gross_profit']

    for chart_name, vars_dict in charts.items():
        # Try to get metric name from multiple sources
        metric_name = vars_dict.get('absolute_metric_name', '').lower()

        # If absolute_metric_name is empty, try the chart name itself
        if not metric_name:
            metric_name = chart_name.lower()

        # Check if this is a large currency metric
        use_millions = any(large_metric in metric_name for large_metric in large_currency_metrics)

        logger.info(f"Processing chart: {chart_name}, metric from name: {metric_name}, use_millions: {use_millions}")
        logger.info(f"Chart vars keys: {vars_dict.keys()}")

        if use_millions:
            # Apply to series data - scale to millions FIRST
            if 'absolute_series' in vars_dict:
                logger.info(f"Found absolute_series: {type(vars_dict['absolute_series'])}")
                for idx, series in enumerate(vars_dict['absolute_series']):
                    logger.info(f"Series {idx}: type={type(series)}, keys={series.keys() if isinstance(series, dict) else 'not dict'}")
                    if isinstance(series, dict) and 'data' in series:
                        original_data = series['data'][:3] if len(series['data']) > 3 else series['data']
                        logger.info(f"Original data sample: {original_data}")
                        # Scale data points to millions
                        series['data'] = [val / 1_000_000 if val is not None else None for val in series['data']]
                        scaled_data = series['data'][:3] if len(series['data']) > 3 else series['data']
                        logger.info(f"Scaled data sample: {scaled_data}")

            # Apply to y-axis - simple format with M suffix
            if 'absolute_y_axis' in vars_dict:
                y_axis = vars_dict['absolute_y_axis']
                logger.info(f"Y-axis before: {y_axis}")
                if isinstance(y_axis, dict):
                    y_axis['labels'] = y_axis.get('labels', {})
                    # Highcharts format syntax
                    y_axis['labels']['format'] = '${value:.1f}M'
                    logger.info(f"Y-axis after: {y_axis}")

    return charts


def map_chart_variables(chart_vars, prefix):
    """
    Maps prefixed chart variables to generic variable names expected by the layout.

    Args:
        chart_vars: Dictionary containing all chart variables with prefixes
        prefix: The prefix to extract (e.g., 'absolute_', 'growth_', 'difference_')

    Returns:
        Dictionary with mapped variables using generic names
    """
    suffixes = ['series', 'x_axis_categories', 'y_axis', 'metric_name', 'meta_df_id']

    mapped_vars = {}

    for suffix in suffixes:
        prefixed_key = f"{prefix}{suffix}"
        if prefixed_key in chart_vars:
            mapped_vars[suffix] = chart_vars[prefixed_key]

    if 'footer' in chart_vars:
        mapped_vars['footer'] = chart_vars['footer']
    if 'hide_footer' in chart_vars:
        mapped_vars['hide_footer'] = chart_vars['hide_footer']

    return mapped_vars


def render_layout(charts, tables, title, subtitle, insights_dfs, warnings, max_prompt, insight_prompt,
                  table_viz_layout, chart_viz_layout, chart_ppt_layout, table_ppt_export_viz_layout):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # Adding insights
    ar_utils = ArUtils()
    insights = ar_utils.get_llm_response(insight_template)

    tab_vars = {
        "headline": title if title else "Total",
        "sub_headline": subtitle or "Trend Analysis",
        "hide_growth_warning": False if warnings else True,
        "exec_summary": insights if insights else "No Insight.",
        "warning": warnings
    }

    viz = []
    slides = []
    for name, chart_vars in charts.items():
        chart_vars["footer"] = f"*{chart_vars['footer']}" if chart_vars.get('footer') else "No additional info."
        rendered = wire_layout(json.loads(chart_viz_layout), {**tab_vars, **chart_vars})
        viz.append(SkillVisualization(title=name, layout=rendered))

        prefixes = ["absolute_", "growth_", "difference_"]

        for prefix in prefixes:
            if (prefix in ["growth_", "difference_"] and
                chart_vars.get("hide_growth_chart", False)):
                continue

            try:
                mapped_vars = map_chart_variables(chart_vars, prefix)
                slide = wire_layout(json.loads(chart_ppt_layout), {**tab_vars, **mapped_vars})
                slides.append(slide)
            except Exception as e:
                logger.error(f"Error rendering chart ppt slide for prefix '{prefix}' in chart '{name}': {e}")

    table_vars = get_table_layout_vars(tables[0])
    table = wire_layout(json.loads(table_viz_layout), {**tab_vars, **table_vars})
    viz.append(SkillVisualization(title="Metrics Table", layout=table))

    if table_ppt_export_viz_layout is not None:
        try:
            table_slide = wire_layout(json.loads(table_ppt_export_viz_layout), {**tab_vars, **table_vars})
            slides.append(table_slide)
        except Exception as e:
            logger.error(f"Error rendering table ppt slide: {e}")
    else:
        slides.append(table)

    return viz, slides, insights, max_response_prompt


if __name__ == '__main__':
    skill_input: SkillInput = trend.create_input(arguments={
        'metrics': ["gross_revenue"],
        'periods': ["Jan 2025"],
        'breakouts': ["scenario"],
        'growth_type': "None"
    })
    out = trend(skill_input)
    preview_skill(trend, out)
