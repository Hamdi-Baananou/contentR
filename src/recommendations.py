import pandas as pd

def recommend_top_recipes(data, model, top_n=10):
    """
    Use the trained model to predict performance, rank posts, and provide actionable advice.

    Parameters:
        data (DataFrame): Processed dataset containing features and labels.
        model: Trained ML model for predicting performance.
        top_n (int): Number of top-performing posts to recommend.

    Returns:
        top_recommendations (DataFrame): Top recommended posts with actionable advice.
        data (DataFrame): Updated dataset with predicted performance.
    """
    features = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    X = data[features]
    predicted_performance = model.predict(X)
    data['predicted_performance'] = predicted_performance

    # Filter and sort top-performing posts
    top_recommendations = (
        data[data['predicted_performance'] == 1]
        .sort_values(by='engagement_rate', ascending=False)
        .head(top_n)
    )

    # Generate actionable advice
    def generate_advice(row):
        advice = []

        # Check hashtags
        if row['hashtags'] < 3:
            advice.append("Consider adding more relevant hashtags (3 or more).")

        # Check sentiment
        if row['sentiment'] < 0:
            advice.append("Try using more positive language to engage your audience.")

        # Check text length
        if row['text_length'] < 20:
            advice.append("Expand the post text to provide more context and detail.")

        # Check posting time
        if row['hour'] < 8 or row['hour'] > 20:
            advice.append("Consider posting during peak engagement hours (8 AM to 8 PM).")

        # Combine all advice
        return " ".join(advice)

    # Apply advice generation to each top recommendation
    top_recommendations['actionable_advice'] = top_recommendations.apply(generate_advice, axis=1)

    # Save predictions to a CSV file
    data.to_csv('processed_combined_data_with_predictions.csv', index=False)

    return top_recommendations, data
