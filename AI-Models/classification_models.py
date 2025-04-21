# classification_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # SVM Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
import numpy as np
import pandas as pd # Import pandas here as well


# Define the categories for consistent labeling
# This order defines the mapping for metrics like confusion matrix rows/cols
BP_CATEGORIES = ['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2']


def train_and_predict_logistic_regression_clf(X_train, y_train, X_test):
    """Trains a Logistic Regression classifier and makes predictions."""
    print("  Training Logistic Regression classifier...")
    # Adjust parameters based on your data and tuning if needed
    # solver='liblinear' is good for small datasets, 'lbfgs' is default and good for larger.
    # max_iter increases convergence attempts.
    # multi_class='auto' selects 'ovr' or 'multinomial' based on data.
    model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter=1000, random_state=42)

    try:
        model.fit(X_train, y_train)
        print("  Logistic Regression classifier training complete.")
        print("  Predicting with Logistic Regression classifier...")
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"  Error during Logistic Regression training or prediction: {e}")
        # Return placeholder predictions if training fails - predicting the most frequent class or a placeholder
        # Ensure y_train is series/array to use value_counts or unique
        y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        if not y_train_series.empty:
             # Get the mode (most frequent value). .iloc[0] handles cases with multiple modes
             most_frequent_class = y_train_series.mode().iloc[0]
             print(f"  Returning predictions of the most frequent class from training data: {most_frequent_class}")
             # Ensure the placeholder is of the correct dtype (object for strings)
             return np.full(X_test.shape[0], most_frequent_class, dtype=object)
        else:
             print("  No training data available to determine most frequent class.")
             # Return a placeholder indicating failure, or a default category if appropriate
             return np.full(X_test.shape[0], np.nan, dtype=object)


def train_and_predict_svm_clf(X_train, y_train, X_test):
    """Trains an SVC (SVM Classifier) model and makes predictions."""
    print("  Training SVC (SVM Classifier) model...")
    # Adjust parameters based on your data and tuning if needed
    model = SVC(kernel='rbf', probability=False, random_state=42) # probability=True if you need predict_proba (slower)

    try:
        model.fit(X_train, y_train)
        print("  SVC training complete.")
        print("  Predicting with SVC model...")
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"  Error during SVC training or prediction: {e}")
        y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        if not y_train_series.empty:
             most_frequent_class = y_train_series.mode().iloc[0]
             print(f"  Returning predictions of the most frequent class from training data: {most_frequent_class}")
             return np.full(X_test.shape[0], most_frequent_class, dtype=object)
        else:
             print("  No training data available to determine most frequent class.")
             return np.full(X_test.shape[0], np.nan, dtype=object)


def train_and_predict_random_forest_clf(X_train, y_train, X_test):
    """Trains a RandomForestClassifier model and makes predictions."""
    print("  Training RandomForestClassifier model...")
    # Adjust parameters based on your data and tuning if needed
    model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees

    try:
        model.fit(X_train, y_train)
        print("  Random Forest Classifier training complete.")
        print("  Predicting with Random Forest Classifier...")
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"  Error during Random Forest Classifier training or prediction: {e}")
        y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        if not y_train_series.empty:
             most_frequent_class = y_train_series.mode().iloc[0]
             print(f"  Returning predictions of the most frequent class from training data: {most_frequent_class}")
             return np.full(X_test.shape[0], most_frequent_class, dtype=object)
        else:
             print("  No training data available to determine most frequent class.")
             return np.full(X_test.shape[0], np.nan, dtype=object)