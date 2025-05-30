from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from textblob import TextBlob
import pandas as pd
from datetime import datetime, date, timedelta
import random
import os

# --- Initialize Spark ---
spark = SparkSession.builder \
    .appName("NewsStockAnalysisPrediction") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()


def get_sentiment(text):
    if not text or text == 'No headline':
        return 'Unknown'
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def infer_genre(text):
    if not text or text == 'No headline':
        return 'Unknown'
    if 'thriller' in text.lower(): return 'Thriller'
    elif 'drama' in text.lower(): return 'Drama'
    elif 'comedy' in text.lower(): return 'Comedy'
    elif 'romance' in text.lower(): return 'Romance'
    else: return 'Other'

def infer_audience(text):
    if not text or text == 'No headline':
        return 'Unknown'
    if any(x in text.lower() for x in ['teen', 'gen z', 'youth']): return 'Youth'
    elif 'adult' in text.lower(): return 'Adults'
    else: return 'General'

def calculate_impact(sentiment, open_price, close_price):
    if sentiment == 'Unknown' or open_price is None or close_price is None:
        return 'Unknown'
    change = (close_price - open_price) / open_price * 100
    if sentiment == 'Positive' and change > 0:
        return 'Positive'
    elif sentiment == 'Negative' and change < 0:
        return 'Negative'
    else:
        return 'Neutral'

sentiment_udf = udf(get_sentiment, StringType())
genre_udf = udf(infer_genre, StringType())
audience_udf = udf(infer_audience, StringType())
impact_udf = udf(calculate_impact, StringType())


def is_weekend(date_obj):
    return date_obj.weekday() >= 5

def mock_news_by_date(date_str):
    """Simulate news headlines."""
    headlines = [
        "Netflix Releases New Thriller Series",
        "Netflix Announces New Comedy Special",
        "Netflix Stock Surges on New Drama Release",
        "Netflix Faces Criticism Over Content",
        "Netflix Partners for New Series",
        "Netflix Expands Youth Programming"
    ]
    published_at = f"{date_str}T{random.randint(8, 18):02d}:00:00Z"
    if random.random() < 0.3:
        # print(f"No news for {date_str}")
        return None, None
    headline = random.choice(headlines)
    # print(f"News for {date_str}: {headline}")
    return headline, published_at

def mock_stock_price(date_str, sentiment=None):
    """Simulate stock prices, tied to sentiment."""
    open_price = 600.00 + random.uniform(-10, 10)
    if sentiment == "Positive":
        close_price = open_price + random.uniform(0, 5)
    elif sentiment == "Negative":
        close_price = open_price + random.uniform(-5, 0)
    else:
        close_price = open_price + random.uniform(-5, 5)
    return round(open_price, 2), round(close_price, 2), datetime.strptime(date_str, "%Y-%m-%d").date()

def cache_data(data, filename):
    if not data:
        # print(f"No data to cache in {filename}")
        return
    try:
        pd.DataFrame(data).to_csv(filename, index=False)
        # print(f"Cached data to {filename}")
    except Exception as e:
        print(f"Error caching data to {filename}: {e}")

def load_cached_data(filename):
    if not os.path.exists(filename):
        # print(f"No cache file found: {filename}")
        return None
    try:
        df = pd.read_csv(filename)
        if df.empty or df.columns.empty:
            # print(f"Cache file {filename} is empty or invalid")
            return None
        data = df.to_dict('records')
        for row in data:
            if pd.notnull(row.get('stock_date')):
                row['stock_date'] = pd.to_datetime(row['stock_date']).date()
        # print(f"Loaded cached data from {filename}")
        return data
    except Exception as e:
        print(f"Error loading cache file {filename}: {e}")
        return None

def analyze_news_stock_range(start_date, end_date):
    cache_file = "news_stock_cache.csv"
    cached_data = load_cached_data(cache_file)
    if cached_data:
        data = cached_data
    else:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start_date_obj > end_date_obj:
            print("Error: Start date must be before end date")
            return None
        
        data = []
        current_date = start_date_obj
        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            # print(f"\nProcessing date: {date_str}")
            
            if is_weekend(current_date):
                data.append((date_str, "Non-trading day", None, None, None, None, "Weekend date"))
                current_date += timedelta(days=1)
                continue

            headline, published_at = mock_news_by_date(date_str)
            sentiment = get_sentiment(headline) if headline else "Unknown"
            if not headline:
                data.append((date_str, "No news found", None, None, None, None, "No news found"))
                current_date += timedelta(days=1)
                continue

            try:
                news_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                news_date = news_dt.date()
                if news_date != current_date:
                    # print(f"News date {news_date} does not match {current_date}")
                    data.append((date_str, headline, published_at, None, None, None, "News date mismatch"))
                    current_date += timedelta(days=1)
                    continue
            except Exception as e:
                print(f"Error parsing publishedAt: {e}")
                data.append((date_str, headline, published_at, None, None, None, "Date parsing error"))
                current_date += timedelta(days=1)
                continue

            open_price, close_price, stock_date = mock_stock_price(date_str, sentiment)
            if stock_date is None or stock_date != current_date:
                # print(f"Stock date {stock_date} does not match {current_date}")
                data.append((date_str, headline, published_at, None, None, None, "Stock date mismatch"))
                current_date += timedelta(days=1)
                continue

            data.append((date_str, headline, published_at, open_price, close_price, stock_date, None))
            current_date += timedelta(days=1)

        cache_data(data, cache_file)
    
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("headline", StringType(), True),
        StructField("publishedAt", StringType(), True),
        StructField("price_open", DoubleType(), True),
        StructField("price_close", DoubleType(), True),
        StructField("stock_date", DateType(), True),
        StructField("impact", StringType(), True)
    ])
    
    try:
        df = spark.createDataFrame(data, schema)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None
    
    df = df.withColumn("sentiment", sentiment_udf(col("headline"))) \
           .withColumn("genre", genre_udf(col("headline"))) \
           .withColumn("audience", audience_udf(col("headline"))) \
           .withColumn("impact", 
                       when(col("impact").isNull(),
                            impact_udf(col("sentiment"), col("price_open"), col("price_close"))
                           ).otherwise(col("impact")))
    
    df = df.select("date", "headline", "publishedAt", "genre", "audience", "sentiment", "price_open", "price_close", "impact")    
    return df

def fetch_historical_data(ticker='NFLX', start_date="2024-04-01", end_date="2024-04-30"):
    cache_file = "historical_cache.csv"
    cached_data = load_cached_data(cache_file)
    if cached_data:
        stock_data = cached_data
    else:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        stock_data = []
        current_date = start_date_obj
        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            if is_weekend(current_date):
                stock_data.append((date_str, None, None, None, "Non-trading day", None))
                current_date += timedelta(days=1)
                continue
            # Fetch mock news
            headline, published_at = mock_news_by_date(date_str)
            sentiment = get_sentiment(headline) if headline else "Unknown"
            # Fetch mock stock
            open_price, close_price, stock_date = mock_stock_price(date_str, sentiment)
            stock_data.append((date_str, open_price, close_price, stock_date, headline or "No headline", published_at))
            current_date += timedelta(days=1)
        
        cache_data(stock_data, cache_file)
    
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("price_open", DoubleType(), True),
        StructField("price_close", DoubleType(), True),
        StructField("stock_date", DateType(), True),
        StructField("headline", StringType(), True),
        StructField("publishedAt", StringType(), True)
    ])
    
    try:
        stock_df = spark.createDataFrame(stock_data, schema)
    except Exception as e:
        print(f"Error creating historical DataFrame: {e}")
        return None
    
    # Apply sentiment analysis
    stock_df = stock_df.withColumn("sentiment", sentiment_udf(col("headline")))
    
    return stock_df


def train_stock_prediction_model(historical_data):
    if historical_data is None:
        print("No historical data to train model")
        return None, None
    
    # Prepare features: sentiment score, previous day's close, open
    data = historical_data.withColumn("sentiment_score", 
        when(col("sentiment") == "Positive", 1.0)
        .when(col("sentiment") == "Negative", -1.0)
        .otherwise(0.0))
    
    # Create feature vector
    assembler = VectorAssembler(
        inputCols=["sentiment_score", "price_close", "price_open"],
        outputCol="features",
        handleInvalid="skip"
    )
    
    # Shift data to predict next day's close
    from pyspark.sql.window import Window
    from pyspark.sql.functions import lag
    w = Window.orderBy("date")
    data = data.withColumn("next_close", lag("price_close", -1).over(w))
    
    data = data.dropna(subset=["next_close", "price_close", "price_open", "sentiment_score"])
    
    if data.count() == 0:
        print("No valid data for training after cleaning")
        return None, None
    
    data = assembler.transform(data)
    
    # Train Linear Regression
    lr = LinearRegression(featuresCol="features", labelCol="next_close")
    model = lr.fit(data)
    return model, assembler


start_date = "2025-05-01"
end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
# end_date = "2025-05-14"  # Includes current date (2025-05-14)
analysis_df = analyze_news_stock_range(start_date, end_date)
if analysis_df is not None:
    analysis_df.show(truncate=False)
else:
    print("Failed to create analysis DataFrame")

# Train prediction model
historical_df = fetch_historical_data('NFLX', "2024-04-01", "2024-04-30")
model, assembler = train_stock_prediction_model(historical_df)

# Dynamic prediction based on current date
if model is not None and assembler is not None:
    current_date = datetime.now().date()
    previous_date = current_date - timedelta(days=1)
    previous_date_str = previous_date.strftime("%Y-%m-%d")
    tomorrow_date = current_date + timedelta(days=1)

    # Fetch previous day's data
    previous_data = analysis_df.filter(col("date") == previous_date_str) \
        .select("date", "headline", "sentiment", "price_close", "price_open") \
        .withColumn("sentiment_score", 
                    when(col("sentiment") == "Positive", 1.0)
                    .when(col("sentiment") == "Negative", -1.0)
                    .otherwise(0.0)) \
        .select("date", "headline", "sentiment", "price_close", "price_open", "sentiment_score")

    if previous_data.count() > 0:
        # Convert to pandas for easier access
        previous_row = previous_data.toPandas().iloc[0]
        
        # Print previous day's data
        print(f"\n📅 Data for {previous_date}:")
        print(f"📰 Headline: {previous_row['headline']}")
        print(f"🧠 Sentiment: {previous_row['sentiment']}")
        print(f"📈 Close Price: ${previous_row['price_close']:.2f}")

        # Prepare features for prediction
        latest_features = assembler.transform(previous_data)
        prediction = model.transform(latest_features)
        predicted_price = prediction.select("prediction").collect()[0][0]

        # Print prediction for tomorrow
        print(f"\n📅 Prediction for {tomorrow_date}:")
        print(f"📰 Based on Headline from {previous_date}: {previous_row['headline']}")
        print(f"🧠 Sentiment: {previous_row['sentiment']}")
        print(f"📈 Predicted Close Price: ${predicted_price:.2f}")
    else:
        print(f"No valid data for {previous_date_str} to make a prediction")
else:
    print("Failed to train prediction model")