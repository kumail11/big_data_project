import json
import time
from kafka import KafkaProducer
from datetime import datetime
import random
import numpy as np

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Global variables for data renewal
last_renewal = datetime.now()
baseline_price = 1200.0  # Starting price, close to recent NFLX value
headlines_pool = []
renewal_interval = random.randint(600, 900)  # 10-15 minutes in seconds

def generate_headlines_pool():
    """Generate a pool of 50 unique Netflix news headlines with sentiments"""
    base_headlines = [
        ("Netflix announces new blockbuster series", "positive"),
        ("Netflix stock surges after strong earnings", "positive"),
        ("Netflix partners with major studio", "positive"),
        ("Netflix expands global streaming market", "positive"),
        ("Netflix releases critically acclaimed film", "positive"),
        ("Netflix subscriber growth exceeds expectations", "positive"),
        ("Netflix unveils new AI-driven content", "positive"),
        ("Netflix secures exclusive streaming deal", "positive"),
        ("Netflix launches innovative ad platform", "positive"),
        ("Netflix wins multiple Emmy awards", "positive"),
        ("Netflix stock dips after mixed reviews", "negative"),
        ("Netflix faces subscriber churn concerns", "negative"),
        ("Netflix loses key content deal", "negative"),
        ("Netflix hit with regulatory scrutiny", "negative"),
        ("Netflix reports lower-than-expected profits", "negative"),
        ("Netflix criticized for content decisions", "negative"),
        ("Netflix faces streaming outage issues", "negative"),
        ("Netflix raises subscription prices", "negative"),
        ("Netflix new series gets mixed reception", "neutral"),
        ("Netflix to release new documentary", "neutral"),
        ("Netflix hosts virtual investor day", "neutral"),
        ("Netflix expands into gaming content", "neutral"),
        ("Netflix tests new user interface", "neutral"),
        ("Netflix announces quarterly results", "neutral"),
        ("Netflix explores live sports streaming", "neutral")
    ]
    # Generate 50 headlines by combining base headlines with variations
    new_pool = []
    for i in range(50):
        base, sentiment = random.choice(base_headlines)
        variation = f"{base} {chr(65 + i%26)}" if i >= len(base_headlines) else base
        new_pool.append((variation, sentiment))
    return new_pool

def generate_mock_stock_data():
    """Generate mock Netflix stock data with random walk"""
    global baseline_price
    # Simulate price movement with random walk
    volatility = 0.02  # 2% volatility
    change = baseline_price * random.uniform(-volatility, volatility)
    baseline_price += change  # Update baseline for trend
    current_price = max(100.0, baseline_price)  # Ensure price stays realistic
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'NFLX',
        'open': round(current_price + random.uniform(-5, 5), 2),
        'high': round(current_price + random.uniform(0, 10), 2),
        'low': round(current_price + random.uniform(-10, 0), 2),
        'close': round(current_price + random.uniform(-5, 5), 2),
        'volume': random.randint(1000000, 5000000)
    }

def generate_mock_news_data():
    """Generate mock Netflix news with sentiment"""
    global headlines_pool
    if not headlines_pool:
        headlines_pool = generate_headlines_pool()
    headline, sentiment = random.choice(headlines_pool)
    return {
        'timestamp': datetime.now().isoformat(),
        'headline': headline,
        'source': 'MockNews',
        'symbol': 'NFLX',
        'sentiment': sentiment
    }

def renew_data():
    """Renew baseline price and headlines pool"""
    global baseline_price, headlines_pool, last_renewal, renewal_interval
    if (datetime.now() - last_renewal).seconds >= renewal_interval:
        # Adjust baseline price with a larger trend
        baseline_price += random.uniform(-20, 20)  # Larger trend shift
        headlines_pool = generate_headlines_pool()  # Refresh headlines
        last_renewal = datetime.now()
        renewal_interval = random.randint(600, 900)  # New 10-15 min interval
        print(f"Renewed data: new baseline price = {baseline_price:.2f}, new headlines pool")

def stream_data():
    """Stream mock data to Kafka topics"""
    print("Starting data streaming...")
    global headlines_pool
    headlines_pool = generate_headlines_pool()  # Initialize headlines

    while True:
        try:
            # Renew data every 10-15 minutes
            renew_data()

            # Send stock data
            stock_data = generate_mock_stock_data()
            producer.send('netflix-stock', value=stock_data).get(timeout=10)
            print(f"Sent stock data: {stock_data['close']}")

            # Send news data with 10% probability
            if random.random() < 0.1:
                news_data = generate_mock_news_data()
                producer.send('netflix-news', value=news_data).get(timeout=10)
                print(f"Sent news: {news_data['headline']} (Sentiment: {news_data['sentiment']})")

            producer.flush()
            time.sleep(5)  # Send data every 5 seconds

        except KeyboardInterrupt:
            print("Stopping producer...")
            break
        except Exception as e:
            print(f"Error in streaming: {e}")
            time.sleep(1)

if __name__ == "__main__":
    stream_data()