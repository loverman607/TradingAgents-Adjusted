"""
Crypto Backtest Dashboard

Interactive web dashboard for analyzing crypto backtest results using Dash.
This dashboard provides comprehensive visualization and analysis of backtest performance.
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Crypto Backtest Dashboard"

# Load backtest results
def load_backtest_results(file_path: str = None):
    """Load backtest results from JSON file"""
    if file_path is None:
        # Look for the most recent backtest results
        results_dir = Path(".")
        json_files = list(results_dir.glob("crypto_backtest_results_*.json"))
        if not json_files:
            return None
        file_path = max(json_files, key=os.path.getctime)
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

# Load results
results = load_backtest_results()

if results is None:
    # Create sample data for demonstration
    results = {
        "backtest_summary": {
            "initial_capital": 100000.0,
            "final_value": 125000.0,
            "total_return": 0.25,
            "total_trades": 150,
            "volatility": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.65
        },
        "portfolio_values": [
            {"timestamp": f"2024-01-{i:02d}", "value": 100000 + i * 100, "cash": 50000, "positions": 2}
            for i in range(1, 32)
        ],
        "trades": [
            {
                "timestamp": f"2024-01-{i:02d}",
                "ticker": "BTC" if i % 2 == 0 else "ETH",
                "side": "BUY" if i % 3 == 0 else "SELL",
                "quantity": 0.1 + i * 0.01,
                "price": 50000 + i * 100,
                "value": 5000 + i * 50,
                "confidence": 0.6 + (i % 4) * 0.1
            }
            for i in range(1, 31)
        ],
        "crypto_metrics": {
            "BTC": {
                "price_return": 0.20,
                "buy_hold_return": 0.18,
                "outperformance": 0.02,
                "volatility": 0.12
            },
            "ETH": {
                "price_return": 0.30,
                "buy_hold_return": 0.25,
                "outperformance": 0.05,
                "volatility": 0.18
            }
        }
    }

# Convert portfolio values to DataFrame
portfolio_df = pd.DataFrame(results["portfolio_values"])
portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
portfolio_df['returns'] = portfolio_df['value'].pct_change()

# Convert trades to DataFrame
trades_df = pd.DataFrame(results["trades"])
if not trades_df.empty:
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

# App layout
app.layout = html.Div([
    html.H1("Crypto Backtest Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Summary Cards
    html.Div([
        html.Div([
            html.H3("Total Return", style={'color': 'green' if results['backtest_summary']['total_return'] > 0 else 'red'}),
            html.H2(f"{results['backtest_summary']['total_return']:.2%}")
        ], className="four columns", style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3("Sharpe Ratio"),
            html.H2(f"{results['backtest_summary']['sharpe_ratio']:.2f}")
        ], className="four columns", style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3("Max Drawdown"),
            html.H2(f"{results['backtest_summary']['max_drawdown']:.2%}")
        ], className="four columns", style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px', 'borderRadius': '5px'}),
    ], className="row"),
    
    # Portfolio Value Chart
    html.Div([
        html.H2("Portfolio Value Over Time"),
        dcc.Graph(
            id='portfolio-chart',
            figure={
                'data': [
                    go.Scatter(
                        x=portfolio_df['timestamp'],
                        y=portfolio_df['value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ),
                    go.Scatter(
                        x=portfolio_df['timestamp'],
                        y=portfolio_df['cash'],
                        mode='lines',
                        name='Cash',
                        line=dict(color='green', width=1)
                    )
                ],
                'layout': go.Layout(
                    title='Portfolio Value Over Time',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Value ($)'},
                    hovermode='closest'
                )
            }
        )
    ], style={'margin': '20px'}),
    
    # Returns Distribution
    html.Div([
        html.H2("Returns Distribution"),
        dcc.Graph(
            id='returns-distribution',
            figure={
                'data': [
                    go.Histogram(
                        x=portfolio_df['returns'].dropna() * 100,
                        nbinsx=30,
                        name='Returns',
                        marker_color='skyblue'
                    )
                ],
                'layout': go.Layout(
                    title='Daily Returns Distribution',
                    xaxis={'title': 'Returns (%)'},
                    yaxis={'title': 'Frequency'}
                )
            }
        )
    ], style={'margin': '20px'}),
    
    # Trades Analysis
    html.Div([
        html.H2("Trades Analysis"),
        dcc.Graph(
            id='trades-chart',
            figure={
                'data': [
                    go.Scatter(
                        x=trades_df[trades_df['side'] == 'BUY']['timestamp'],
                        y=trades_df[trades_df['side'] == 'BUY']['price'],
                        mode='markers',
                        name='Buy Trades',
                        marker=dict(color='green', size=8)
                    ),
                    go.Scatter(
                        x=trades_df[trades_df['side'] == 'SELL']['timestamp'],
                        y=trades_df[trades_df['side'] == 'SELL']['price'],
                        mode='markers',
                        name='Sell Trades',
                        marker=dict(color='red', size=8)
                    )
                ],
                'layout': go.Layout(
                    title='Trades Over Time',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price ($)'}
                )
            }
        )
    ], style={'margin': '20px'}),
    
    # Crypto Performance Comparison
    html.Div([
        html.H2("Crypto Performance Comparison"),
        dcc.Graph(
            id='crypto-performance',
            figure={
                'data': [
                    go.Bar(
                        x=list(results['crypto_metrics'].keys()),
                        y=[results['crypto_metrics'][ticker]['price_return'] * 100 for ticker in results['crypto_metrics'].keys()],
                        name='Buy & Hold',
                        marker_color='lightblue'
                    ),
                    go.Bar(
                        x=list(results['crypto_metrics'].keys()),
                        y=[results['backtest_summary']['total_return'] * 100] * len(results['crypto_metrics']),
                        name='Strategy',
                        marker_color='orange'
                    )
                ],
                'layout': go.Layout(
                    title='Performance Comparison',
                    xaxis={'title': 'Cryptocurrency'},
                    yaxis={'title': 'Return (%)'},
                    barmode='group'
                )
            }
        )
    ], style={'margin': '20px'}),
    
    # Trades Table
    html.Div([
        html.H2("Recent Trades"),
        dash_table.DataTable(
            id='trades-table',
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Ticker", "id": "ticker"},
                {"name": "Side", "id": "side"},
                {"name": "Quantity", "id": "quantity", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Value", "id": "value", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            data=trades_df.tail(20).to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{side} = BUY'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                },
                {
                    'if': {'filter_query': '{side} = SELL'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                }
            ]
        )
    ], style={'margin': '20px'}),
    
    # Risk Metrics
    html.Div([
        html.H2("Risk Metrics"),
        html.Div([
            html.Div([
                html.H4("Volatility"),
                html.P(f"{results['backtest_summary']['volatility']:.2%}")
            ], className="three columns", style={'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'borderRadius': '5px'}),
            
            html.Div([
                html.H4("Win Rate"),
                html.P(f"{results['backtest_summary']['win_rate']:.2%}")
            ], className="three columns", style={'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'borderRadius': '5px'}),
            
            html.Div([
                html.H4("Total Trades"),
                html.P(f"{results['backtest_summary']['total_trades']}")
            ], className="three columns", style={'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'borderRadius': '5px'}),
            
            html.Div([
                html.H4("Final Value"),
                html.P(f"${results['backtest_summary']['final_value']:,.2f}")
            ], className="three columns", style={'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'borderRadius': '5px'}),
        ], className="row")
    ], style={'margin': '20px'}),
    
    # Footer
    html.Div([
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               style={'textAlign': 'center', 'color': 'gray', 'fontSize': '12px'})
    ], style={'marginTop': '50px'})
])

# Callbacks for interactive features
@app.callback(
    Output('portfolio-chart', 'figure'),
    [Input('portfolio-chart', 'id')]
)
def update_portfolio_chart(_):
    """Update portfolio chart with latest data"""
    return {
        'data': [
            go.Scatter(
                x=portfolio_df['timestamp'],
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            go.Scatter(
                x=portfolio_df['timestamp'],
                y=portfolio_df['cash'],
                mode='lines',
                name='Cash',
                line=dict(color='green', width=1)
            )
        ],
        'layout': go.Layout(
            title='Portfolio Value Over Time',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Value ($)'},
            hovermode='closest'
        )
    }

@app.callback(
    Output('returns-distribution', 'figure'),
    [Input('returns-distribution', 'id')]
)
def update_returns_distribution(_):
    """Update returns distribution chart"""
    returns = portfolio_df['returns'].dropna() * 100
    return {
        'data': [
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='Returns',
                marker_color='skyblue'
            )
        ],
        'layout': go.Layout(
            title='Daily Returns Distribution',
            xaxis={'title': 'Returns (%)'},
            yaxis={'title': 'Frequency'}
        )
    }

if __name__ == '__main__':
    print("Starting Crypto Backtest Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
