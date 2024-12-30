import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Import SMOTE

def train_random_forest(data):
    """
    Train a Random Forest model with SMOTE for data balancing.

    data: DataFrame that must contain:
      - A target column: 'performance_category' (0=low, 1=top)
      - Feature columns: 'hashtags', 'emojis', 'sentiment', 'text_length', 'hour'

    Returns:
      - model: Fitted RandomForestClassifier
      - X_test, y_test: Testing set
      - y_pred: Predictions on X_test
    """
    features = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    X = data[features]
    y = data['performance_category']

    # Apply SMOTE for data balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split resampled dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred
