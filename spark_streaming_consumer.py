from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
from tabulate import tabulate

# ---------------- SPARK SESSION ----------------
spark = SparkSession.builder \
    .appName("NetflixStockStreaming") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------- SCHEMAS ----------------
stock_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", IntegerType(), True)
])

news_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("headline", StringType(), True),
    StructField("source", StringType(), True),
    StructField("symbol", StringType(), True)
])

# ---------------- SENTIMENT UDF ----------------
def get_sentiment(text):
    if not text:
        return 'Neutral'
    positive_words = ['surges', 'expands', 'partners', 'new', 'release', 'success', 'growth', 'strong', 'reports']
    negative_words = ['criticism', 'faces', 'drops', 'problem', 'struggles', 'issues', 'loses', 'deal']
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    return 'Positive' if pos_count > neg_count else 'Negative' if neg_count > pos_count else 'Neutral'

sentiment_udf = udf(get_sentiment, StringType())

# ---------------- STREAMING READ ----------------
stock_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "netflix-stock") \
    .option("startingOffsets", "latest") \
    .load()

news_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "netflix-news") \
    .option("startingOffsets", "latest") \
    .load()

# ---------------- PARSE STREAMS ----------------
stock_df = stock_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", stock_schema).alias("data")).select("data.*")

news_df = news_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", news_schema).alias("data")).select("data.*") \
    .withColumn("sentiment", sentiment_udf(col("headline")))

# ---------------- ENRICH STOCK DATA ----------------
stock_enriched = stock_df \
    .withColumn("price_change", round(col("close") - col("open"), 2)) \
    .withColumn("price_change_pct", round((col("close") - col("open")) / col("open") * 100, 2)) \
    .withColumn("trend", when(col("close") > col("open"), "UP")
                .when(col("close") < col("open"), "DOWN")
                .otherwise("FLAT"))

# ---------------- DISPLAY BATCH TO CONSOLE ----------------
# def display_stock(batch_df, batch_id):
#     if not batch_df.isEmpty():
#         pdf = batch_df.select("timestamp", "symbol", "open", "high", "low", "close", "volume",
#                               "price_change", "price_change_pct", "trend").toPandas()
#         print("\n=== Real-time Stock Data ===")
#         print(tabulate(pdf, headers='keys', tablefmt='grid', showindex=False))

# def display_news(batch_df, batch_id):
#     if not batch_df.isEmpty():
#         pdf = batch_df.select("timestamp", "symbol", "source", "headline", "sentiment").toPandas()
#         print("\n=== Real-time News Data with Sentiment ===")
#         print(tabulate(pdf, headers='keys', tablefmt='grid', showindex=False))

def display_news(df, epoch_id):
    print(f"=== News Batch {epoch_id} ===")
    df.show(truncate=False)
    
def display_stock(df, epoch_id):
    print(f"=== Stock Batch {epoch_id} ===")
    df.show(truncate=False)


# ---------------- START STREAMS ----------------

news_query = news_df.writeStream \
    .outputMode("append") \
    .foreachBatch(display_news) \
    .trigger(processingTime="10 seconds") \
    .option("checkpointLocation", "file:///C:/tmp/checkpoint/news") \
    .start()

stock_query = stock_enriched.writeStream \
    .outputMode("append") \
    .foreachBatch(display_stock) \
    .trigger(processingTime="10 seconds") \
    .option("checkpointLocation", "file:///C:/tmp/checkpoint/stock") \
    .start()



# ---------------- AWAIT TERMINATION ----------------
stock_query.awaitTermination()
news_query.awaitTermination()
