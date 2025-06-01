from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from threading import Thread
import time
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global DataFrame to store stock data history
stock_data = pd.DataFrame(columns=['Datetime', 'Open', 'Close'])

def fetch_stock_data():
    global stock_data
    last_fetched_time = None
    ticker = yf.Ticker("NFLX")

    while True:
        try:
            # Fetch latest 1-minute data (last 5 minutes to ensure we get the newest minute)
            data = ticker.history(period="1d", interval="1m")
            new_data = data[['Open', 'Close']].tail(5)  # Get last 5 minutes to avoid missing data
            new_data = new_data.reset_index()

            # Ensure minute-wise data by flooring to minute
            new_data['Datetime'] = new_data['Datetime'].dt.floor('min')

            # Filter out duplicates and only append new data
            if last_fetched_time is not None:
                new_data = new_data[new_data['Datetime'] > last_fetched_time]

            if not new_data.empty:
                # Append new data to global DataFrame
                stock_data = pd.concat([stock_data, new_data[['Datetime', 'Open', 'Close']]], ignore_index=True)
                # Remove duplicates based on Datetime
                stock_data = stock_data.drop_duplicates(subset='Datetime', keep='last')
                # Keep only the last 60 minutes
                stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
                cutoff_time = stock_data['Datetime'].max() - pd.Timedelta(minutes=60)
                stock_data = stock_data[stock_data['Datetime'] >= cutoff_time]
                last_fetched_time = stock_data['Datetime'].max()
                logger.debug(f"Appended new data. Latest rows:\n{stock_data.tail(5)}")
            else:
                logger.debug("No new data to append.")

        except Exception as e:
            logger.error(f"Error fetching data: {e}")

        time.sleep(30)  # Wait 30 seconds before next fetch

# Start background thread for data fetching
thread = Thread(target=fetch_stock_data)
thread.daemon = True
thread.start()

@app.route('/')
def dashboard():
    global stock_data
    if stock_data.empty:
        logger.warning("Stock data not yet available")
        return "Fetching data, please wait..."

    try:
        # Prepare data for charts
        df = stock_data.copy()
        df['Timestamp'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')

        # Create bar chart
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=df['Timestamp'],
            y=df['Open'],
            name='Open',
            marker_color='blue'
        ))
        bar_fig.add_trace(go.Bar(
            x=df['Timestamp'],
            y=df['Close'],
            name='Close',
            marker_color='red'
        ))
        bar_fig.update_layout(
            title='Netflix Stock: Open vs Close (Last 60 Minutes)',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            barmode='group',
            xaxis_tickangle=45
        )

        # Create line chart
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['Open'],
            name='Open',
            mode='lines+markers',
            line=dict(color='blue')
        ))
        line_fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['Close'],
            name='Close',
            mode='lines+markers',
            line=dict(color='red')
        ))
        line_fig.update_layout(
            title='Netflix Stock: Open vs Close Trend (Last 60 Minutes)',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            xaxis_tickangle=45
        )

        # Convert plots to JSON for rendering
        bar_chart = bar_fig.to_json()
        line_chart = line_fig.to_json()

        return render_template('dashboard.html', bar_chart=bar_chart, line_chart=line_chart)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return "Error rendering dashboard, please check logs."

if __name__ == '__main__':
    app.run(debug=True)
