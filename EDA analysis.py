import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(input_file, negative_reviews_file, output_file):
    # Loading cleaned data
    data = pd.read_csv(input_file)

    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
        return ' '.join(tokens)

    data['processed_text'] = data['reviews.text'].apply(preprocess_text)

    # Saving processed data
    data.to_csv(output_file, index=False)

    # Loading negative reviews
    with open(negative_reviews_file, 'r') as f:
        negative_reviews = f.read().splitlines()

    # Identifying top words in negative reviews
    negative_words = ' '.join(negative_reviews).lower()
    negative_tokens = word_tokenize(negative_words)
    negative_tokens = [lemmatizer.lemmatize(token) for token in negative_tokens if token.isalpha() and token not in stop_words]
    word_freq = pd.Series(negative_tokens).value_counts().head(10)

    # Visualizing top words in negative reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index)
    plt.title('Top 10 Words in Negative Reviews')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.savefig('negative_words.png')
    plt.close()
    print("EDA complete. Processed data saved to", output_file)

if __name__ == "__main__":
    perform_eda('cleaned_data.csv', 'negative_reviews.txt', 'processed_data.csv')
