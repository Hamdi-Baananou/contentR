# src/recommendations.py
import pandas as pd

def recommend_top_recipes(data, model, top_n=10):
    """
    Uses trained model to predict performance on the entire dataset,
    then returns top_n recommended recipes based on 'engagement_rate'.
    Assumes data has 'engagement_rate' column and features used by the model.
    """
    features = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    X = data[features]
    predicted_performance = model.predict(X)
    data['predicted_performance'] = predicted_performance

    top_recommendations = (
        data[data['predicted_performance'] == 1]
        .sort_values(by='engagement_rate', ascending=False)
        .head(top_n)
    )

    # Print for logging
    print("Top Recommended Recipes:")
    for idx, recipe in top_recommendations.iterrows():
        print("\nRecommended Recipe:")
        print("Message:", recipe['message'])
        print("Hashtags:", recipe['hashtags'])
        print("Emojis:", recipe['emojis'])
        print("Sentiment:", recipe['sentiment'])
        print("Text Length:", recipe['text_length'])
        print("Hour:", recipe['hour'])
        print("Day of Week:", recipe['day_of_week'])
        if 'writing_recommendations' in data.columns:
            rec = recipe['writing_recommendations']
            print("Recommendation:", rec.get('recommendation', 'No recommendation.'))

    # Save final CSV with predictions
    data.to_csv('processed_combined_data_with_predictions.csv', index=False)
    print("Model recommendations have been generated and saved.")

    return top_recommendations, data
