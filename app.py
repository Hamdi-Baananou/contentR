import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from src.data_preprocessing import load_and_update_data
from src.feature_engineering import engineer_features
from src.model_training import train_random_forest, train_xgboost  # Import both training functions
from src.recommendations import recommend_top_recipes

nltk.download('vader_lexicon')

st.set_page_config(page_title="Recipe Analysis", layout="wide")
st.title("Recipe Engagement Analysis and Prediction")

# ==================== Session State Init =====================
if "model" not in st.session_state:
    st.session_state["model"] = None

if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "Random Forest"

# ==================== 1. Data Preprocessing ====================
st.header("1. Update Data from JSON")
json_files = ["data/fetchedData.json", "data/fetchedDataFB.json"]
top_csv_path = "data/top_performers.csv"
low_csv_path = "data/low_performers.csv"

if st.button("Load & Update CSVs"):
    top_df, low_df = load_and_update_data(json_files, top_csv_path, low_csv_path)
    st.write("Top Performers:")
    st.dataframe(top_df.head())
    st.write("Low Performers:")
    st.dataframe(low_df.head())

# ==================== 2. Feature Engineering ====================
st.header("2. Feature Engineering & Combined Data")
if st.button("Engineer Features"):
    top_df = pd.read_csv(top_csv_path)
    low_df = pd.read_csv(low_csv_path)

    top_df['performance_category'] = 1
    low_df['performance_category'] = 0

    data = pd.concat([top_df, low_df], ignore_index=True)
    data = engineer_features(data)

    st.write("Preview after feature engineering:")
    st.dataframe(data.head())

    data.to_csv("data/processed_combined_data.csv", index=False)
    st.success("Data with new features saved to data/processed_combined_data.csv")

# ==================== 3. Model Training ====================
st.header("3. Train Model")
model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])
use_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)

if st.button("Train Model"):
    # Load the processed data
    data = pd.read_csv("data/processed_combined_data.csv")

    if model_choice == "Random Forest":
        model, X_test, y_test, y_pred = train_random_forest(data, use_tuning=use_tuning)
    elif model_choice == "XGBoost":
        model, X_test, y_test, y_pred = train_xgboost(data, use_tuning=use_tuning)

    # Store the selected model and model type in session state
    st.session_state["model"] = model
    st.session_state["model_choice"] = model_choice

    # Display feature importances
    st.subheader(f"Feature Importances ({model_choice})")
    feature_names = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    feature_importances = model.feature_importances_
    feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    feature_df.sort_values(by="Importance", ascending=False, inplace=True)
    st.dataframe(feature_df)

    # Visualizations
    st.header("4. Visualizations")
    corr_cols = [
        'engagement_rate', 'reactions', 'comments', 'shares',
        'hashtags', 'emojis', 'text_length', 'sentiment', 'hour'
    ]
    corr_data = data[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.success(f"{model_choice} model training {'with tuning' if use_tuning else ''} complete!")

# ==================== 5. Generate Recommendations ====================
st.header("5. Generate Top Recommended Recipes")
if st.button("Generate Recommendations"):
    # Check if the model is available in session_state
    if st.session_state["model"] is None:
        st.error("Please train the model first!")
    else:
        data = pd.read_csv("data/processed_combined_data.csv")
        top_recs, data_with_preds = recommend_top_recipes(
            data,
            st.session_state["model"],  # Use the stored model
            top_n=10
        )

        st.subheader(f"Top Recommended Recipes ({st.session_state['model_choice']})")
        st.write(top_recs[['id', 'message', 'engagement_rate', 'predicted_performance', 'actionable_advice']])
        st.success("Recommendations with actionable advice generated!")

        # Store top_recs in session state so we can use them in step 6
        st.session_state["top_recs"] = top_recs

# ==================== 6. Blog-Style Display of Top Recipes ====================
st.header("6. Display Recommended Recipes as a Blog")
st.write("Below you can view the recommended recipes in a more visual, blog-like format.")

# Check if we have top_recs in session_state
if "top_recs" in st.session_state and st.session_state["top_recs"] is not None:
    
    top_recs_df = st.session_state["top_recs"].copy()
    
    # OPTIONAL: Provide a way to filter by ID
    all_ids = list(top_recs_df["id"].unique())
    selected_ids = st.multiselect(
        "Filter by Post IDs (leave empty to show all):",
        options=all_ids,
        default=all_ids
    )
    if selected_ids:
        top_recs_df = top_recs_df[top_recs_df["id"].isin(selected_ids)]
    
    # Display each recipe in a grid-like layout (2 columns, for example)
    num_cols = 2
    rows = [top_recs_df.iloc[i:i+num_cols] for i in range(0, len(top_recs_df), num_cols)]
    
    for row_data in rows:
        cols = st.columns(num_cols)
        for col, (_, recipe) in zip(cols, row_data.iterrows()):
            # If you have an 'image_url' column, display the image
            if "image_url" in recipe and pd.notna(recipe["image_url"]):
                col.image(recipe["image_url"], use_column_width=True)
            
            col.markdown(f"**ID**: {recipe['id']}")
            
            # Show creation date if exists
            if "created_time" in recipe:
                col.markdown(f"**Created**: {recipe['created_time']}")
                
            # Show standard info
            col.markdown(f"**Message**: {recipe['message']}")
            
            # A small JS snippet to copy text to clipboard
            copy_code = f"""
            <script>
            function copyToClipboard(text) {{
                navigator.clipboard.writeText(text);
                alert('Message copied to clipboard!');
            }}
            </script>
            <button onclick="copyToClipboard('{recipe['message'].replace("'", "\\'")}')">Copy Message</button>
            """
            col.markdown(copy_code, unsafe_allow_html=True)
            
            # Engagement metrics (if columns exist)
            if "reactions" in recipe:
                col.markdown(f"**Likes/Reactions**: {recipe['reactions']}")
            if "comments" in recipe:
                col.markdown(f"**Comments**: {recipe['comments']}")
            if "shares" in recipe:
                col.markdown(f"**Shares**: {recipe['shares']}")
            if "engagement_rate" in recipe:
                col.markdown(f"**Engagement Rate**: {recipe['engagement_rate']}")
            
            # Optionally show the actionable advice
            if "actionable_advice" in recipe:
                col.markdown(f"**Advice**: {recipe['actionable_advice']}")

            col.write("---")

else:
    st.info("Generate recommendations above to display them here.")
