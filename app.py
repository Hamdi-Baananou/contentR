import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Local imports from src folder
from src.data_preprocessing import load_and_update_data
from src.feature_engineering import engineer_features
from src.model_training import train_random_forest

# Download NLTK data on first run
nltk.download('vader_lexicon')

# Set Streamlit page config
st.set_page_config(page_title="Recipe Analysis", layout="wide")

st.title("Recipe Engagement Analysis and Prediction")

# ========== 1. Data Preprocessing ==========

st.header("1. Update Data from JSON")
json_files = [
    "data/fetchedData.json",
    "data/fetchedDataFB.json"
] 
top_csv_path = "data/top_performers.csv"
low_csv_path = "data/low_performers.csv"

if st.button("Load & Update CSVs"):
    top_df, low_df = load_and_update_data(json_files, top_csv_path, low_csv_path)
    st.write("Top Performers:")
    st.dataframe(top_df.head())
    st.write("Low Performers:")
    st.dataframe(low_df.head())

# ========== 2. Feature Engineering ==========

st.header("2. Feature Engineering & Combined Data")
if st.button("Engineer Features"):
    # Reload them fresh from CSV after update
    top_df = pd.read_csv(top_csv_path)
    low_df = pd.read_csv(low_csv_path)

    # Add 'performance_category' (1=top, 0=low) before combining
    top_df['performance_category'] = 1
    low_df['performance_category'] = 0

    data = pd.concat([top_df, low_df], ignore_index=True)
    data = engineer_features(data)

    st.write("Preview after feature engineering:")
    st.dataframe(data.head())

    # Save or cache combined data
    data.to_csv("data/processed_combined_data.csv", index=False)
    st.success("Data with new features saved to data/processed_combined_data.csv")

# ========== 3. Model Training ==========

st.header("3. Train Random Forest Model")
if st.button("Train Model"):
    # Load the processed data
    data = pd.read_csv("data/processed_combined_data.csv")

    # Train
    model, X_test, y_test, y_pred = train_random_forest(data)

    st.subheader("Feature Importances")
    feature_names = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df.sort_values(by='Importance', ascending=False, inplace=True)
    st.dataframe(feature_df)

    # ========== 4. Visualization (Correlation, etc.) ==========
    st.header("4. Visualizations")
    corr_cols = ['engagement_rate', 'reactions', 'comments', 'shares',
                 'hashtags', 'emojis', 'text_length', 'sentiment', 'hour']
    corr_data = data[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.success("Model training complete!")
