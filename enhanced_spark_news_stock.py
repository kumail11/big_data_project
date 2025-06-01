from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, when, lit, lag, lead, avg, stddev, max as spark_max, min as spark_min,
    date_format, dayofweek, regexp_extract, length, split, size, broadcast,
    monotonically_increasing_id, row_number, rank, dense_rank
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, DateType, 
    IntegerType, TimestampType, BooleanType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from textblob import TextBlob
import pandas as pd
from datetime import datetime, date, timedelta
import random
import os

# --- Enhanced Spark Configuration ---
spark = SparkSession.builder \
    .appName("EnhancedNewsStockAnalysisPrediction") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# --- Enhanced UDFs with better error handling ---
def get_sentiment_detailed(text):
    """Enhanced sentiment analysis with confidence score"""
    if not text or text == 'No headline' or pd.isna(text):
        return ('Unknown', 0.0)
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        confidence = abs(polarity)
        
        if polarity > 0.2:
            return ('Positive', confidence)
        elif polarity < -0.2:
            return ('Negative', confidence)
        else:
            return ('Neutral', confidence)
    except Exception:
        return ('Unknown', 0.0)

def extract_keywords(text):
    """Extract key financial keywords"""
    if not text or pd.isna(text):
        return []
    
    financial_keywords = [
        'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
        'merger', 'acquisition', 'partnership', 'expansion', 'layoffs',
        'investment', 'dividend', 'stock', 'shares', 'market', 'trading'
    ]
    
    text_lower = str(text).lower()
    found_keywords = [kw for kw in financial_keywords if kw in text_lower]
    return found_keywords

def calculate_volatility(prices):
    """Calculate price volatility"""
    if not prices or len(prices) < 2:
        return 0.0
    try:
        return float(pd.Series(prices).std())
    except:
        return 0.0

# Register UDFs with return types
sentiment_detailed_udf = udf(get_sentiment_detailed, StructType([
    StructField("sentiment", StringType(), True),
    StructField("confidence", DoubleType(), True)
]))

keywords_udf = udf(extract_keywords, StringType())  # Will store as comma-separated string
volatility_udf = udf(calculate_volatility, DoubleType())

# --- Enhanced data generation functions ---
def generate_realistic_stock_data(date_str, sentiment=None, previous_close=600.0):
    """Generate more realistic stock data with trends"""
    base_volatility = 0.02  # 2% base volatility
    
    # Sentiment impact
    sentiment_multiplier = 1.0
    if sentiment == "Positive":
        sentiment_multiplier = 1.01  # 1% positive bias
    elif sentiment == "Negative":
        sentiment_multiplier = 0.99  # 1% negative bias
    
    # Random walk with sentiment bias
    daily_return = random.gauss(0, base_volatility) * sentiment_multiplier
    open_price = previous_close * (1 + random.gauss(0, 0.005))  # Small gap from previous close
    close_price = open_price * (1 + daily_return)
    
    # Generate intraday high/low
    high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
    low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
    
    # Volume (higher on volatile days)
    base_volume = 1000000
    volume_multiplier = 1 + abs(daily_return) * 10
    volume = int(base_volume * volume_multiplier * random.uniform(0.5, 2.0))
    
    return {
        'open': round(open_price, 2),
        'high': round(high_price, 2),
        'low': round(low_price, 2),
        'close': round(close_price, 2),
        'volume': volume,
        'date': datetime.strptime(date_str, "%Y-%m-%d").date()
    }

def generate_enhanced_news(date_str):
    """Generate more diverse and realistic news headlines"""
    news_templates = [
        ("Netflix announces Q{} earnings beat expectations", "Positive", "earnings"),
        ("Netflix subscriber growth slows in latest quarter", "Negative", "subscribers"),
        ("Netflix launches new original series '{}'", "Neutral", "content"),
        ("Netflix faces increased competition from rivals", "Negative", "competition"),
        ("Netflix expands into {} market", "Positive", "expansion"),
        ("Netflix stock upgraded by {} analysts", "Positive", "analyst"),
        ("Netflix content costs rise amid production surge", "Negative", "costs"),
        ("Netflix partners with {} for exclusive content", "Positive", "partnership"),
        ("Netflix implements password sharing crackdown", "Negative", "policy"),
        ("Netflix ad-tier shows strong adoption rates", "Positive", "advertising")
    ]
    
    if random.random() < 0.3:  # 30% chance of no news
        return None, None, None, []
    
    template, expected_sentiment, category = random.choice(news_templates)
    
    # Fill in template placeholders
    placeholders = {
        '{}': ['major', 'leading', 'top', 'significant'],
        'Q{}': [f'Q{random.randint(1,4)}'],
        "'{}'": ['The Crown', 'Stranger Things', 'Bridgerton', 'Wednesday'],
    }
    
    headline = template
    for placeholder, options in placeholders.items():
        if placeholder in headline:
            headline = headline.replace(placeholder, random.choice(options), 1)
    
    published_at = f"{date_str}T{random.randint(8, 18):02d}:{random.randint(0, 59):02d}:00Z"
    keywords = extract_keywords(headline)
    
    return headline, published_at, expected_sentiment, keywords

# --- Enhanced data processing functions ---
def create_enhanced_dataset(start_date, end_date, ticker='NFLX'):
    """Create a comprehensive dataset with enhanced features"""
    
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    data = []
    previous_close = 600.0  # Starting price
    
    current_date = start_date_obj
    while current_date <= end_date_obj:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Check if trading day
        is_trading_day = current_date.weekday() < 5  # Monday=0, Friday=4
        
        if not is_trading_day:
            data.append({
                'date': date_str,
                'ticker': ticker,
                'is_trading_day': False,
                'headline': None,
                'published_at': None,
                'expected_sentiment': None,
                'keywords': None,
                'open': None, 'high': None, 'low': None, 'close': None, 'volume': None,
                'previous_close': previous_close
            })
        else:
            # Generate news
            headline, published_at, expected_sentiment, keywords = generate_enhanced_news(date_str)
            
            # Get sentiment from headline if available
            if headline:
                actual_sentiment, confidence = get_sentiment_detailed(headline)
            else:
                actual_sentiment, confidence = 'Unknown', 0.0
            
            # Generate stock data
            stock_data = generate_realistic_stock_data(date_str, actual_sentiment, previous_close)
            
            data.append({
                'date': date_str,
                'ticker': ticker,
                'is_trading_day': True,
                'headline': headline,
                'published_at': published_at,
                'expected_sentiment': expected_sentiment,
                'keywords': ','.join(keywords) if keywords else None,
                'open': stock_data['open'],
                'high': stock_data['high'], 
                'low': stock_data['low'],
                'close': stock_data['close'],
                'volume': stock_data['volume'],
                'previous_close': previous_close
            })
            
            previous_close = stock_data['close']
        
        current_date += timedelta(days=1)
    
    return data

def create_spark_dataframe(data):
    """Create Spark DataFrame with comprehensive schema"""
    
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("ticker", StringType(), True),
        StructField("is_trading_day", BooleanType(), True),
        StructField("headline", StringType(), True),
        StructField("published_at", StringType(), True),
        StructField("expected_sentiment", StringType(), True),
        StructField("keywords", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", IntegerType(), True),
        StructField("previous_close", DoubleType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    
    # Add derived columns
    df = df.withColumn("date_parsed", col("date").cast(DateType())) \
           .withColumn("day_of_week", dayofweek(col("date_parsed"))) \
           .withColumn("price_change", col("close") - col("open")) \
           .withColumn("price_change_pct", 
                      when(col("open").isNotNull() & (col("open") != 0),
                           (col("close") - col("open")) / col("open") * 100)
                      .otherwise(0.0)) \
           .withColumn("daily_range", col("high") - col("low")) \
           .withColumn("daily_range_pct",
                      when(col("low").isNotNull() & (col("low") != 0),
                           (col("high") - col("low")) / col("low") * 100)
                      .otherwise(0.0))
    
    # Add sentiment analysis
    sentiment_df = df.withColumn("sentiment_analysis", 
                                when(col("headline").isNotNull(),
                                     sentiment_detailed_udf(col("headline")))
                                .otherwise(lit(None)))
    
    sentiment_df = sentiment_df.withColumn("sentiment", 
                                          col("sentiment_analysis.sentiment")) \
                              .withColumn("sentiment_confidence", 
                                         col("sentiment_analysis.confidence")) \
                              .drop("sentiment_analysis")
    
    return sentiment_df

def add_technical_indicators(df):
    """Add technical analysis indicators using Spark window functions"""
    
    # Define window for time series calculations
    window_5 = Window.partitionBy("ticker").orderBy("date_parsed").rowsBetween(-4, 0)
    window_20 = Window.partitionBy("ticker").orderBy("date_parsed").rowsBetween(-19, 0)
    
    # Add moving averages
    df = df.withColumn("ma_5", avg("close").over(window_5)) \
           .withColumn("ma_20", avg("close").over(window_20))
    
    # Add volatility (rolling standard deviation)
    df = df.withColumn("volatility_5", stddev("close").over(window_5)) \
           .withColumn("volatility_20", stddev("close").over(window_20))
    
    # Add price momentum
    window_lag = Window.partitionBy("ticker").orderBy("date_parsed")
    df = df.withColumn("close_lag_1", lag("close", 1).over(window_lag)) \
           .withColumn("close_lag_5", lag("close", 5).over(window_lag)) \
           .withColumn("momentum_1d", 
                      when(col("close_lag_1").isNotNull(),
                           (col("close") - col("close_lag_1")) / col("close_lag_1") * 100)
                      .otherwise(0.0)) \
           .withColumn("momentum_5d",
                      when(col("close_lag_5").isNotNull(), 
                           (col("close") - col("close_lag_5")) / col("close_lag_5") * 100)
                      .otherwise(0.0))
    
    # Add RSI-like indicator (simplified)
    df = df.withColumn("price_up", 
                      when(col("price_change") > 0, col("price_change")).otherwise(0.0)) \
           .withColumn("price_down",
                      when(col("price_change") < 0, -col("price_change")).otherwise(0.0))
    
    df = df.withColumn("avg_gain", avg("price_up").over(window_20)) \
           .withColumn("avg_loss", avg("price_down").over(window_20)) \
           .withColumn("rsi_like",
                      when(col("avg_loss") != 0,
                           100 - (100 / (1 + col("avg_gain") / col("avg_loss"))))
                      .otherwise(50.0))
    
    return df

def create_ml_pipeline():
    """Create a comprehensive ML pipeline for stock prediction"""
    
    # Feature engineering
    assembler = VectorAssembler(
        inputCols=[
            "sentiment_score", "sentiment_confidence", "ma_5", "ma_20",
            "volatility_5", "momentum_1d", "momentum_5d", "rsi_like",
            "daily_range_pct", "volume_scaled", "day_of_week"
        ],
        outputCol="features_raw",
        handleInvalid="skip"
    )
    
    # Feature scaling
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    # Multiple algorithms to compare
    lr = LinearRegression(featuresCol="features", labelCol="next_close")
    rf = RandomForestRegressor(featuresCol="features", labelCol="next_close", numTrees=50)
    
    # Create pipeline
    pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
    
    return pipeline_lr, pipeline_rf

def prepare_ml_features(df):
    """Prepare features for machine learning"""
    
    # Add sentiment scores
    df = df.withColumn("sentiment_score",
                      when(col("sentiment") == "Positive", 1.0)
                      .when(col("sentiment") == "Negative", -1.0)
                      .otherwise(0.0))
    
    # Scale volume
    volume_stats = df.agg(avg("volume").alias("avg_vol"), stddev("volume").alias("std_vol")).collect()[0]
    avg_vol = volume_stats["avg_vol"] or 1000000
    std_vol = volume_stats["std_vol"] or 100000
    
    df = df.withColumn("volume_scaled", 
                      (col("volume") - lit(avg_vol)) / lit(std_vol))
    
    # Create target variable (next day's close)
    window_lead = Window.partitionBy("ticker").orderBy("date_parsed")
    df = df.withColumn("next_close", lead("close", 1).over(window_lead))
    
    # Filter out non-trading days and rows without targets
    df = df.filter(col("is_trading_day") == True) \
           .filter(col("next_close").isNotNull()) \
           .filter(col("close").isNotNull())
    
    return df

def evaluate_models(df):
    """Train and evaluate multiple models"""
    
    # Prepare data
    df = prepare_ml_features(df)
    
    if df.count() < 10:
        print("Insufficient data for model training")
        return None, None, None
    
    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training samples: {train_df.count()}")
    print(f"Test samples: {test_df.count()}")
    
    # Create pipelines
    pipeline_lr, pipeline_rf = create_ml_pipeline()
    
    # Train models
    try:
        model_lr = pipeline_lr.fit(train_df)
        model_rf = pipeline_rf.fit(train_df)
        
        # Make predictions
        pred_lr = model_lr.transform(test_df)
        pred_rf = model_rf.transform(test_df)
        
        # Evaluate
        evaluator = RegressionEvaluator(labelCol="next_close", predictionCol="prediction")
        
        rmse_lr = evaluator.evaluate(pred_lr, {evaluator.metricName: "rmse"})
        rmse_rf = evaluator.evaluate(pred_rf, {evaluator.metricName: "rmse"})
        
        mae_lr = evaluator.evaluate(pred_lr, {evaluator.metricName: "mae"})
        mae_rf = evaluator.evaluate(pred_rf, {evaluator.metricName: "mae"})
        
        print(f"\nModel Performance:")
        print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
        print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")
        
        # Choose best model
        best_model = model_lr if rmse_lr < rmse_rf else model_rf
        best_name = "Linear Regression" if rmse_lr < rmse_rf else "Random Forest"
        
        print(f"Best Model: {best_name}")
        
        return best_model, train_df, test_df
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, None, None

def main():
    """Main execution function"""
    
    print("Starting Enhanced Spark News-Stock Analysis")
    
    # Generate data
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    print(f"Generating data from {start_date} to {end_date}")
    raw_data = create_enhanced_dataset(start_date, end_date)
    
    # Create Spark DataFrame
    print("Creating Spark DataFrame...")
    df = create_spark_dataframe(raw_data)
    
    # Add technical indicators
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Show sample data
    print("\nSample Data:")
    df.filter(col("is_trading_day") == True).select(
        "date", "headline", "sentiment", "sentiment_confidence", 
        "open", "close", "price_change_pct", "ma_5", "volatility_5"
    ).orderBy("date").show(10, truncate=False)
    
    # Basic statistics
    print("\nBasic Statistics:")
    df.filter(col("is_trading_day") == True).select("close", "volume", "price_change_pct").describe().show()
    
    # Sentiment distribution
    print("\nSentiment Distribution:")
    df.filter(col("headline").isNotNull()).groupBy("sentiment").count().orderBy("count", ascending=False).show()
    
    # Train and evaluate models
    print("\nTraining ML Models...")
    best_model, train_df, test_df = evaluate_models(df)
    
    if best_model is not None:
        # Make prediction for latest data
        latest_data = df.filter(col("is_trading_day") == True).orderBy(col("date_parsed").desc()).limit(1)
        
        if latest_data.count() > 0:
            latest_prepared = prepare_ml_features(latest_data)
            if latest_prepared.count() > 0:
                try:
                    prediction = best_model.transform(latest_prepared)
                    pred_row = prediction.select("date", "headline", "sentiment", "close", "prediction").collect()[0]
                    
                    print(f"\nLatest Prediction:")
                    print(f"Date: {pred_row['date']}")
                    print(f"Headline: {pred_row['headline']}")
                    print(f"Sentiment: {pred_row['sentiment']}")
                    print(f"Current Close: ${pred_row['close']:.2f}")
                    print(f"Predicted Next Close: ${pred_row['prediction']:.2f}")
                    print(f"Expected Change: {((pred_row['prediction'] - pred_row['close']) / pred_row['close'] * 100):.2f}%")
                    
                except Exception as e:
                    print(f"Error making prediction: {e}")
    
    # Clean up
    spark.stop()
    print("\nAnalysis Complete!")

# Execute main function
if __name__ == "__main__":
    main()