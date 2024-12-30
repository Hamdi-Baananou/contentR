import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_random_forest(data, use_tuning=False):
    """
    Train a Random Forest model with optional hyperparameter tuning using SMOTE for data balancing.
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

    if use_tuning:
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("Best Parameters for Random Forest:", grid_search.best_params_)
    else:
        # Train without tuning
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred


def train_xgboost(data, use_tuning=False):
    """
    Train an XGBoost model with optional hyperparameter tuning using SMOTE for data balancing.
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

    if use_tuning:
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("Best Parameters for XGBoost:", grid_search.best_params_)
    else:
        # Train without tuning
        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred
