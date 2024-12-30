import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier  # Import XGBoost

def train_random_forest(data):
    """
    Train a Random Forest model with SMOTE for data balancing.
    """
    features = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    X = data[features]
    y = data['performance_category']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred


def train_xgboost(data):
    """
    Train an XGBoost model with SMOTE for data balancing.
    """
    features = ['hashtags', 'emojis', 'sentiment', 'text_length', 'hour']
    X = data[features]
    y = data['performance_category']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred
