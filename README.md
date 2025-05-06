# Customer-Sentiment-Analysis-on-E-commerce-Reviews
Overview

This project performs sentiment analysis on e-commerce product reviews (e.g., Amazon product reviews) to uncover customer sentiments, identify common complaints, and provide actionable recommendations. The project uses Python for data processing, analysis, and machine learning, and Power BI for interactive visualizations. Key tasks include data cleaning, MySQL storage, exploratory data analysis (EDA), predictive modeling, and visualization of sentiment trends.

The dataset (not included due to sensitivity) contains product details and customer reviews, processed through a pipeline to generate insights like sentiment score distribution, product-wise sentiment trends, and frequent complaint words.

Project Structure





Python Scripts:





data_preparation.py: Loads and cleans the dataset, producing cleaned_data.csv.



database_setup.py: Stores cleaned data in a MySQL database and extracts negative reviews (negative_reviews.txt).



eda_analysis.py: Performs EDA and text preprocessing, generating processed_data.csv and negative_words.png.



predictive_model.py: Trains a Linear Regression model to predict sentiment scores, producing data_with_sentiment.csv and model_metrics.txt.



visualizations.py: Creates visualizations (sentiment distribution, product trends, complaints) as PNGs.



generate_report.py: Generates a markdown report (sentiment_analysis_report.md) with insights and recommendations.



Power BI Visualizations:





power_bi_visualizations.txt: Instructions for creating Power BI visuals (sentiment score distribution, product-wise trends, complaint words) using data_with_sentiment.csv.



Output Files (if included):





CSVs: cleaned_data.csv, processed_data.csv, data_with_sentiment.csv.



Text: negative_reviews.txt, model_metrics.txt, sentiment_analysis_report.md.



Images: negative_words.png, sentiment_distribution.png, product_sentiment.png, complaint_words.png.

Features





Data Cleaning: Handles missing values, standardizes text, and converts dates.



MySQL Integration: Stores product and review data in a relational database.



EDA: Analyzes review text to identify common words and trends using NLTK.



Predictive Modeling: Uses Linear Regression to predict sentiment scores based on review text.



Visualizations:





Python: Matplotlib/Seaborn plots for sentiment score distribution



Power BI: Interactive dashboard with histogram-like sentiment distribution, bar charts for product trends, and complaint word frequencies.




Usage





Python Pipeline: Run the scripts sequentially to process data, store in MySQL, perform EDA, predict sentiment, and generate visualizations/report.



Power BI Dashboard: Use Power BI to explore interactive visuals with slicers for product and date filtering.



Output Analysis: Review sentiment_analysis_report.md for insights and recommendations (e.g., improve app availability, address performance issues).
