import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 4: Predictive Analysis Using Machine Learning
def train_model(input_file, output_metrics_file):
    # Loading processed data
    data = pd.read_csv(input_file)

    # Fill missing text values
    data['reviews.text'] = data['reviews.text'].fillna('')
    data['processed_text'] = data['processed_text'].fillna('')

    # Creating a simple sentiment score
    def sentiment_score(text):
        text = text.lower()
        positive_words = ['great', 'love', 'awesome', 'excellent', 'good', 'happy']
        negative_words = ['bad', 'poor', 'not', 'disappointing', 'slow', 'problem']
        score = 0
        for word in positive_words:
            if word in text:
                score += 1
        for word in negative_words:
            if word in text:
                score -= 1
        return score

    data['sentiment_score'] = data['reviews.text'].apply(sentiment_score)

    # Preparing data for Linear Regression
    X = data['processed_text']
    y = data['sentiment_score']

    # Converting text to TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(X)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Training Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Saving metrics
    with open(output_metrics_file, 'w') as f:
        f.write(f"Mean Squared Error: {mse:.2f}\n")
        f.write(f"Mean Absolute Error: {mae:.2f}\n")
        f.write(f"R-squared: {r2:.2f}\n")

    # Saving data with sentiment scores
    data.to_csv('data_with_sentiment.csv', index=False)
    print(f"Model trained. Metrics saved to {output_metrics_file}")

    # Return y for plotting
    return data['sentiment_score']

if __name__ == "__main__":
    sentiment_scores = train_model('processed_data.csv', 'model_metrics.txt')

    # Visualize sentiment score distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(sentiment_scores, kde=True, bins=30)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
