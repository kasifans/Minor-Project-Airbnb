# 1. Load all necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

print("--- Imports successful ---")


# 2. Load the dataset
file_path = "Airbnb_Open_Data.xlsx - in.csv"
try:
    # Use low_memory=False to avoid mixed type errors
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Successfully loaded data from {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")
    df = pd.DataFrame() # Create empty df on fail

if not df.empty:
    
    # 3. Data Cleaning and Feature Engineering
    print("--- Starting data cleaning and feature engineering ---")
    
    # Make a copy to work on
    df_cleaned = df.copy()

    # Drop useless columns
    columns_to_drop = ['license', 'house_rules', 'host name', 'host id']
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    
    # Drop rows with nulls in key columns (price, location, etc.)
    critical_columns = ['price', 'neighbourhood group', 'neighbourhood', 'lat', 'long', 'room type', 'NAME']
    df_cleaned = df_cleaned.dropna(subset=critical_columns)
    
    # Drop country columns (all US)
    df_cleaned = df_cleaned.drop(columns=['country', 'country code'])
    
    # Convert 'last review' to datetime
    df_cleaned['last review'] = pd.to_datetime(df_cleaned['last review'], errors='coerce')
    
    # Fill missing categorical data
    df_cleaned['host_identity_verified'] = df_cleaned['host_identity_verified'].fillna('unconfirmed')
    df_cleaned['cancellation_policy'] = df_cleaned['cancellation_policy'].fillna(df_cleaned['cancellation_policy'].mode()[0])
    df_cleaned['instant_bookable'] = df_cleaned['instant_bookable'].fillna(df_cleaned['instant_bookable'].mode()[0])
    
    # Fill missing numerical data (Imputation)
    # Use median for construction year, service fee, etc.
    for col in ['Construction year', 'service fee', 'minimum nights', 'calculated host listings count']:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
        
    # Fill review counts with 0 (no review = 0)
    for col in ['number of reviews', 'reviews per month']:
        df_cleaned[col] = df_cleaned[col].fillna(0)
        
    # Fill review rate with median
    median_review_rate = df_cleaned['review rate number'].median()
    df_cleaned['review rate number'] = df_cleaned['review rate number'].fillna(median_review_rate)
    
    # Fill availability with 0
    df_cleaned['availability 365'] = df_cleaned['availability 365'].fillna(0)
    
    # New Feature: days_since_last_review
    # Find the most recent review date as 'today'
    max_date = df_cleaned['last review'].max()
    
    # Calculate days elapsed
    df_cleaned['days_since_last_review'] = (max_date - df_cleaned['last review']).dt.days
    
    # If a listing never had a review, fill with a large number
    # Get 99th percentile for the fill value
    high_value = df_cleaned['days_since_last_review'].quantile(0.99)
    df_cleaned['days_since_last_review'] = df_cleaned['days_since_last_review'].fillna(high_value)

    # Drop old 'last review' column, no longer needed
    df_cleaned = df_cleaned.drop(columns=['last review'])
    
    # Reset index for the recommender mapping
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    print("--- Data cleaning complete ---")

    
    # 4. Price Prediction Model (Random Forest)
    print("--- Building Price Prediction Model ---")
    
    # Define X (features) and y (target)
    price_features = ['neighbourhood group', 'neighbourhood', 'room type', 'lat', 'long',
                      'Construction year', 'minimum nights', 'number of reviews',
                      'review rate number', 'availability 365', 'days_since_last_review', 'service fee']
    target = 'price'

    X = df_cleaned[price_features]
    y = df_cleaned[target]

    # Separate feature types for preprocessing
    categorical_features = ['neighbourhood group', 'neighbourhood', 'room type']
    numerical_features = [col for col in price_features if col not in categorical_features]

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Build the model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Check the score
    model_score = model.score(X_test, y_test)
    print(f"Model R-squared score: {model_score:.2f}")

    
    # 5. Recommendation System (Content-Based)
    print("--- Building Recommendation System ---")
    
    # Features for recommender
    rec_features = ['neighbourhood group', 'neighbourhood', 'room type', 'price', 'lat', 'long']
    rec_data = df_cleaned[rec_features]
    
    # Preprocessor for recommender (scale numbers, encode categories)
    rec_cat_features = ['neighbourahood group', 'neighbourhood', 'room type']
    rec_num_features = ['price', 'lat', 'long']
    
    rec_preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), rec_num_features), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), rec_cat_features)
        ],
        remainder='drop')
        
    # Transform the data
    feature_matrix = rec_preprocessor.fit_transform(rec_data)
    
    # Get the similarity matrix
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
    
    # Map listing names to their index number
    indices = pd.Series(df_cleaned.index, index=df_cleaned['NAME']).drop_duplicates()

    # Main recommendation function
    def get_recommendations(title, cosine_sim=cosine_sim, data=df_cleaned, indices=indices):
        try:
            # Get index of the listing from its name
            idx = indices[title]
        except KeyError:
            print(f"Listing '{title}' not found.")
            return pd.Series(dtype='object')

        # Get similarity scores for all listings
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top 10 (ignore [0], it's the item itself)
        sim_scores = sim_scores[1:11]

        # Get original indices
        listing_indices = [i[0] for i in sim_scores]

        # Return the names of the top 10
        return data['NAME'].iloc[listing_indices]

        
    # 6. --- DEMO ---
    print("\n--- Running Demo ---")
    
    # Test with an example listing
    test_listing = "Skylit Midtown Castle"
    
    # Check if name is in our index
    if test_listing in indices:
        recommendations = get_recommendations(test_listing)
        print(f"\nRecommendations for '{test_listing}':")
        # Print top 3 to match the PPT slide
        print(recommendations.head(3).to_string())
    else:
        print(f"\nDemo listing '{test_listing}' not in cleaned data. Skipping demo.")

else:
    print("DataFrame is empty. Halting script.")