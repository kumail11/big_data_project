from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lag
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# --- Initialize Spark with increased memory ---
spark = SparkSession.builder \
    .appName("NewsStockAnalysisPrediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

# Register sentiment UDFs - Using simplified approach instead of TextBlob
def get_sentiment(text):
    """Simplified sentiment analysis without TextBlob dependency"""
    if not text or text == 'No headline':
        return 'Unknown'
    
    # Simple keyword-based approach
    positive_words = ['surges', 'expands', 'partners', 'new', 'release', 'success', 'growth']
    negative_words = ['criticism', 'faces', 'drops', 'problem', 'struggles', 'issues']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
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
    
    # Handle null/None values properly
    try:
        open_val = float(open_price)
        close_val = float(close_price)
        change = (close_val - open_val) / open_val * 100
        if sentiment == 'Positive' and change > 0:
            return 'Positive'
        elif sentiment == 'Negative' and change < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except (TypeError, ValueError):
        return 'Unknown'

# Register UDFs with proper return types
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
        return None, None
    headline = random.choice(headlines)
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
    
    try:
        stock_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return round(open_price, 2), round(close_price, 2), stock_date
    except ValueError:
        print(f"Error parsing date: {date_str}")
        return round(open_price, 2), round(close_price, 2), None

def cache_data(data, filename):
    """Cache data to CSV - using Spark's native CSV handling instead of pandas"""
    if not data:
        print(f"No data to cache in {filename}")
        return
        
    try:
        # Convert to DataFrame and write directly with Spark
        df = spark.createDataFrame(data)
        df.write.mode("overwrite").option("header", "true").csv(filename)
        print(f"Cached data to {filename}")
    except Exception as e:
        print(f"Error caching data to {filename}: {e}")

def load_cached_data(filename):
    """Load cached data using Spark instead of pandas"""
    if not os.path.exists(filename):
        print(f"No cache file found: {filename}")
        return None
        
    try:
        # Read with Spark
        df = spark.read.option("header", "true").csv(filename)
        if df.count() == 0:
            print(f"Cache file {filename} is empty")
            return None
            
        # Convert to Python objects for compatibility with rest of code
        data = df.collect()
        result = []
        
        for row in data:
            row_dict = row.asDict()
            # Handle date conversion
            if 'stock_date' in row_dict and row_dict['stock_date']:
                try:
                    row_dict['stock_date'] = datetime.strptime(row_dict['stock_date'], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    row_dict['stock_date'] = None
                    
            # Convert to tuple format expected by downstream code
            result.append(tuple(row_dict.values()))
            
        print(f"Loaded cached data from {filename}")
        return result
    except Exception as e:
        print(f"Error loading cache file {filename}: {e}")
        return None

def analyze_news_stock_range(start_date, end_date):
    cache_file = "news_stock_cache.csv"
    cached_data = load_cached_data(cache_file)
    
    if cached_data:
        data = cached_data
    else:
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_date_obj > end_date_obj:
                print("Error: Start date must be before end date")
                return None
            
            data = []
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                
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
                    # Parse date without using fromisoformat (for compatibility)
                    news_dt = datetime.strptime(published_at.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    news_date = news_dt.date()
                    if news_date != current_date:
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
                    data.append((date_str, headline, published_at, None, None, None, "Stock date mismatch"))
                    current_date += timedelta(days=1)
                    continue

                data.append((date_str, headline, published_at, open_price, close_price, stock_date, None))
                current_date += timedelta(days=1)

            cache_data(data, cache_file)
        except Exception as e:
            print(f"Error generating data: {e}")
            return None
    
    # Create schema for DataFrame
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
        # Create DataFrame from data - ensure data types match schema
        processed_data = []
        for item in data:
            if len(item) != 7:
                # Handle inconsistent data
                continue
                
            # Ensure correct types for each field
            date, headline, published_at, price_open, price_close, stock_date, impact = item
            
            # Convert price to proper numeric values
            if price_open is not None:
                try:
                    price_open = float(price_open)
                except (ValueError, TypeError):
                    price_open = None
                    
            if price_close is not None:
                try:
                    price_close = float(price_close)
                except (ValueError, TypeError):
                    price_close = None
            
            processed_data.append((date, headline, published_at, price_open, price_close, stock_date, impact))
        
        df = spark.createDataFrame(processed_data, schema)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None
    
    # Apply UDFs
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
        try:
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
        except Exception as e:
            print(f"Error generating historical data: {e}")
            return None
    
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("price_open", DoubleType(), True),
        StructField("price_close", DoubleType(), True),
        StructField("stock_date", DateType(), True),
        StructField("headline", StringType(), True),
        StructField("publishedAt", StringType(), True)
    ])
    
    try:
        # Process data to ensure correct types
        processed_data = []
        for item in stock_data:
            if len(item) != 6:
                continue
                
            date, price_open, price_close, stock_date, headline, published_at = item
            
            # Convert price to proper numeric values
            if price_open is not None:
                try:
                    price_open = float(price_open)
                except (ValueError, TypeError):
                    price_open = None
                    
            if price_close is not None:
                try:
                    price_close = float(price_close)
                except (ValueError, TypeError):
                    price_close = None
            
            processed_data.append((date, price_open, price_close, stock_date, headline, published_at))
        
        stock_df = spark.createDataFrame(processed_data, schema)
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
    
    # Filter out rows with null values
    data = historical_data.filter(
        col("price_open").isNotNull() & 
        col("price_close").isNotNull() & 
        col("sentiment").isNotNull()
    )
    
    if data.count() == 0:
        print("No valid data for training after filtering")
        return None, None
    
    # Prepare features: sentiment score, previous day's close, open
    data = data.withColumn("sentiment_score", 
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
    w = Window.orderBy("date")
    data = data.withColumn("next_close", lag(col("price_close"), -1).over(w))
    
    data = data.filter(col("next_close").isNotNull())
    
    if data.count() == 0:
        print("No valid data for training after cleaning")
        return None, None
    
    data = assembler.transform(data)
    
    # Train Linear Regression
    lr = LinearRegression(featuresCol="features", labelCol="next_close")
    
    try:
        model = lr.fit(data)
        return model, assembler
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None


# Main execution
try:
    # Calculate date range for analysis
    today = datetime.now().date()
    start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")  # Last 14 days
    end_date = today.strftime("%Y-%m-%d")
    
    print(f"Analyzing news and stock data from {start_date} to {end_date}")
    
    analysis_df = analyze_news_stock_range(start_date, end_date)
    if analysis_df is not None:
        print("\n=== Recent News and Stock Analysis ===")
        analysis_df.show(truncate=False)
    else:
        print("Failed to create analysis DataFrame")
    
    # Train prediction model with historical data
    print("\n=== Training prediction model ===")
    historical_start = (today - timedelta(days=45)).strftime("%Y-%m-%d")  # 45 days of training data
    historical_end = (today - timedelta(days=15)).strftime("%Y-%m-%d")    # Up to 15 days ago
    
    print(f"Using historical data from {historical_start} to {historical_end}")
    historical_df = fetch_historical_data('NFLX', historical_start, historical_end)
    
    model, assembler = train_stock_prediction_model(historical_df)
    
    # Make prediction based on current date
    if model is not None and assembler is not None and analysis_df is not None:
        previous_date = today - timedelta(days=1)
        previous_date_str = previous_date.strftime("%Y-%m-%d")
        tomorrow_date = today + timedelta(days=1)
        tomorrow_date_str = tomorrow_date.strftime("%Y-%m-%d")
    
        # Fetch previous day's data
        previous_data = analysis_df.filter(col("date") == previous_date_str) \
            .select("date", "headline", "sentiment", "price_close", "price_open") \
            .withColumn("sentiment_score", 
                        when(col("sentiment") == "Positive", 1.0)
                        .when(col("sentiment") == "Negative", -1.0)
                        .otherwise(0.0)) \
            .select("date", "headline", "sentiment", "price_close", "price_open", "sentiment_score")
    
        if previous_data.count() > 0:
            # Convert to pandas for easier access (keeping the conversion small)
            previous_row = previous_data.limit(1).toPandas().iloc[0]
            
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
            print(f"\n📅 Prediction for {tomorrow_date_str}:")
            print(f"📰 Based on Headline from {previous_date}: {previous_row['headline']}")
            print(f"🧠 Sentiment: {previous_row['sentiment']}")
            print(f"📈 Predicted Close Price: ${predicted_price:.2f}")
        else:
            print(f"No valid data for {previous_date_str} to make a prediction")
    elif model is None:
        print("Failed to train prediction model - no valid model created")
    elif assembler is None:
        print("Failed to create feature assembler")
    else:
        print("Missing data for predictions")

except Exception as e:
    print(f"An error occurred during execution: {e}")
finally:
    # Stop the Spark session when finished
    spark.stop()
