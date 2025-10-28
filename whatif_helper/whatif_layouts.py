WHATIF_LAYOUT = """{
    "layoutJson": {
        "type": "Document",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "15px"
        },
        "children": [
            {
                "name": "FlexContainer_Header",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "80px",
                "direction": "column",
                "style": {
                    "backgroundColor": "#3b82f6",
                    "padding": "20px",
                    "borderRadius": "8px",
                    "marginBottom": "20px"
                },
                "label": "FlexContainer-Header"
            },
            {
                "name": "Header_Title",
                "type": "Header",
                "children": "",
                "text": "COGS What-If Analysis",
                "style": {
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "color": "#ffffff",
                    "textAlign": "left",
                    "margin": "0"
                },
                "parentId": "FlexContainer_Header",
                "label": "Header-Main_Title"
            },
            {
                "name": "Header_Subtitle",
                "type": "Header",
                "children": "",
                "text": "Impact of Price Changes on COGS",
                "style": {
                    "fontSize": "16px",
                    "fontWeight": "normal",
                    "color": "#e5e7eb",
                    "textAlign": "left",
                    "marginTop": "5px"
                },
                "parentId": "FlexContainer_Header",
                "label": "Header-Subtitle"
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "children": "",
                "minHeight": "400px",
                "options": {
                    "chart": {
                        "type": "column",
                        "backgroundColor": "#f8fafc"
                    },
                    "title": {
                        "text": "COGS Forecasted vs Estimated",
                        "style": {
                            "fontSize": "20px"
                        }
                    },
                    "xAxis": {
                        "categories": ["Snack Bars", "Biscuits", "Cakes and Pastries", "Chocolate"],
                        "title": {
                            "text": "Category"
                        }
                    },
                    "yAxis": {
                        "title": {
                            "text": "COGS"
                        },
                        "labels": {
                            "format": "${value:,.0f}"
                        }
                    },
                    "series": [
                        {
                            "name": "COGS Forecasted",
                            "data": [1640740, 4441940, 1634330, 3289790],
                            "color": "#5DADE2"
                        },
                        {
                            "name": "COGS Estimated",
                            "data": [1655090, 4470820, 1641080, 3330580],
                            "color": "#8E44AD"
                        }
                    ],
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
                        "column": {
                            "dataLabels": {
                                "enabled": false
                            }
                        }
                    },
                    "tooltip": {
                        "pointFormat": "<b>{series.name}</b>: ${point.y:,.0f}"
                    }
                },
                "label": "HighchartsChart-COGS",
                "extraStyles": "border-radius: 8px"
            },
            {
                "name": "Markdown0",
                "type": "Markdown",
                "children": "",
                "text": "Analysis insights will appear here...",
                "style": {
                    "fontSize": "16px",
                    "color": "#000000",
                    "border": "none"
                },
                "parentId": "FlexContainer0",
                "label": "Markdown-insights"
            },
            {
                "name": "FlexContainer0",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "250px",
                "style": {
                    "borderRadius": "11.911px",
                    "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                    "padding": "10px",
                    "fontFamily": "Arial",
                    "backgroundColor": "#edf2f7",
                    "border-left": "4px solid #3b82f6"
                },
                "direction": "column",
                "hidden": false,
                "label": "FlexContainer-Insights",
                "extraStyles": "border-radius: 8px;",
                "flex": "1 1 250px"
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [
                    {"name": "Category"},
                    {"name": "COGS Forecasted"},
                    {"name": "COGS Estimated"},
                    {"name": "Change"},
                    {"name": "Material Forecasted"},
                    {"name": "Material Estimated"},
                    {"name": "Material Change"},
                    {"name": "Cocoa Forecasted"},
                    {"name": "Cocoa Estimated"},
                    {"name": "Cocoa Change"}
                ],
                "data": [
                    ["Snack Bars", "$1.64M", "$1.66M", "0.88%", "$1.15M", "$1.16M", "1.25%", "$267.13M", "$301.49M", "5.0%"],
                    ["Biscuits", "$4.44M", "$4.47M", "0.65%", "$2.89M", "$2.92M", "1.0%", "$577.45M", "$606.33M", "5.0%"]
                ],
                "label": "DataTable-COGS"
            }
        ]
    },
    "inputVariables": [
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
            "name": "chart_data_series",
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
        }
    ]
}"""
