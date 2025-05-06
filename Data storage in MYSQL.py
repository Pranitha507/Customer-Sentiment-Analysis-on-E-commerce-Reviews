import pandas as pd
import mysql.connector
import uuid

# Step 2: Store Data in MySQL Database
def setup_database(input_file):
    # Establishing MySQL connection
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="system",
        database="ecommerce_reviews"
    )

    cursor = db.cursor()

    # Creating tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255),
        brand VARCHAR(100),
        categories TEXT,
        primaryCategories VARCHAR(255)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id VARCHAR(255) PRIMARY KEY,
        product_id VARCHAR(255),
        review_date DATETIME,
        review_text TEXT,
        review_title VARCHAR(255),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """)

    # Loading cleaned data
    data = pd.read_csv(input_file)

    # Inserting data into MySQL
    for _, row in data.iterrows():
        product_id = str(uuid.uuid4())
        review_id = str(uuid.uuid4())
        
        # Insert into products table
        cursor.execute("""
        INSERT IGNORE INTO products (product_id, name, brand, categories, primaryCategories)
        VALUES (%s, %s, %s, %s, %s)
        """, (product_id, row['name'], row['brand'], row['categories'], row['primaryCategories']))
        
        # Insert into reviews table
        cursor.execute("""
        INSERT INTO reviews (review_id, product_id, review_date, review_text, review_title)
        VALUES (%s, %s, %s, %s, %s)
        """, (review_id, product_id, row['reviews.date'], row['reviews.text'], row['reviews.title']))

    db.commit()

    # Querying negative reviews
    cursor.execute("""
    SELECT review_text
    FROM reviews
    WHERE review_text LIKE '%not%' OR review_text LIKE '%bad%' OR review_text LIKE '%poor%'
    """)
    negative_reviews = [row[0] for row in cursor.fetchall()]

    # Saving negative reviews for EDA
    with open('negative_reviews.txt', 'w') as f:
        f.write('\n'.join(negative_reviews))

    # Closing database connection
    cursor.close()
    db.close()
    print("Database setup complete. Negative reviews saved to negative_reviews.txt")

if __name__ == "__main__":
    setup_database('cleaned_data.csv')