import pandas as pd

# Step 1: Data Collection and Preparation
def prepare_data(input_file, output_file):
    # Loading the dataset
    data = pd.read_csv(input_file)

    # Cleaning the dataset
    data = data.dropna(subset=['reviews.text', 'reviews.title'])  
    data['reviews.date'] = pd.to_datetime(data['reviews.date'], errors='coerce')  
    data['reviews.text'] = data['reviews.text'].str.lower() 
    data['reviews.title'] = data['reviews.title'].str.lower()

    # Saving cleaned data
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    prepare_data('Amazon_test_data.csv', 'cleaned_data.csv')
