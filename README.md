# Big Data Project
## Real-Time Big Data Analysis of Netflix Stock and User Trends

PROBLEM STATEMENT
In today's digital age, streaming platforms like Netflix are influenced by various real-time
factors such as user interests, stock performance, and content trends. However, most analyses are
performed on historical data and do not consider real-time shifts in stock prices or search
interest. This project aims to address the gap by building a real-time analytics system that
integrates financial data with online search trends to monitor and visualize Netflix’s performance
dynamically.

OBJECTIVES
 To fetch and process live stock data of Netflix (NFLX) from Yahoo Finance API.
 To retrieve user interest trends from Google Trends related to Netflix.
 To apply real-time data processing using Spark Streaming.
 To visualize trends in a live Flask dashboard using Plotly.

DATASET DESCRIPTION
 Netflix Stock Data (Live): From Yahoo Finance API (e.g., yfinance library)
 Google Trends Data: Live search interest data using PyTrends API

SIZE & STRUCTURE
Due to continuous real-time data ingestion and processing, the volume and velocity of
data qualify it as a big data use case. Every minute, new records are fetched, stored, and
analyzed, emulating a real-world streaming data environment.

TOOLS & TECHNOLOGIES
 Apache Spark (PySpark Streaming)
 Flask (for dashboard)
 Yfinance (for live stock data)
 PyTrends (for Google search interest)
 Plotly (for visualizations)
 Pandas, NumPy (for data manipulation)
 Optional: Kafka/HDFS for distributed ingestion/storage (future scope)

METHODOLOGY
 Fetch real-time stock data using yfinance and user trend data using pytrends.
 Use PySpark Streaming to ingest and process the data every minute.
 Clean and aggregate data using Spark transformations.
 Store or directly visualize the results via a Flask web application with Plotly charts.
 Continuously update the dashboard with new data.

EXPECTED OUTCOMES
 We expect to build a fully functional streaming application that:
 Displays real-time trends in stock performance and user interest.
 Helps identify correlations between user engagement and financial performance.
 Success will be measured by timely data updates, accuracy of the information, and the
responsiveness of the dashboard.
BIG DATA USAGE & REAL-TIME IMPACT
Big Data Usage:
This project utilizes big data principles by processing large volumes of streaming stock
data and Google Trends data in real-time. Apache Spark Streaming is used to continuously fetch,

process, and analyze data as it arrives. The high velocity and volume of data simulate a real-
world big data environment, and by using Spark's distributed computing capabilities, we ensure

scalability and performance for heavy workloads.
Real-World Impact:
This analysis enables stakeholders such as investors, financial analysts, and media
companies to understand how public sentiment and content trends impact stock movements of
major streaming platforms like Netflix. This can lead to more informed investment decisions and
marketing strategies. Moreover, trend prediction and content success forecasting can benefit
OTT platforms in content planning and budgeting.

Monetization and Business Opportunities:
 Investment firms and hedge funds for predictive analytics on stock price movements.
 Media and entertainment companies for content performance analysis.
 Marketing firms for campaign optimization based on trend data.
 Data platforms that provide analytics-as-a-service.
 Additionally, this project can be offered as a SaaS product for live trend and financial
sentiment analysis.
