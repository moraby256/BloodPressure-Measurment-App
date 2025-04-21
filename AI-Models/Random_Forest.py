# Random_Forest.py (Using RandomizedSearchCV)
from sklearn.ensemble import RandomForestRegressor
# Import RandomizedSearchCV instead of GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_and_predict_rf(X_train, y_train, X_test):
    """Trains a RandomForestRegressor model and makes predictions (basic version)."""
    # This function is not used in main.py anymore if tuning is enabled there
    # But kept here for completeness or alternative usage
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def tune_and_predict_rf(X_train, y_train, X_test):
    """Performs RandomizedSearchCV for RandomForestRegressor and makes predictions with the best model."""
    print("  Tuning RandomForestRegressor model...")

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define the parameter distribution (can use wider ranges than GridSearchCV)
    # Randomized search samples from these distributions/lists
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500], # Number of trees
        'max_depth': [None, 10, 20, 30, 50, 70, 100], # Maximum depth (None means full depth)
        'min_samples_split': [2, 5, 10, 20, 50], # Min samples required to split
        'min_samples_leaf': [1, 2, 4, 8, 10]     # Min samples at a leaf node
    }

    # Set up RandomizedSearchCV
    # n_iter: Number of parameter settings that are sampled. Decrease this to make it faster.
    # cv: Number of cross-validation folds.
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    # scoring='neg_root_mean_squared_error' minimizes RMSE.
    # random_state for reproducibility of the random sampling.
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=100, # Try 100 random combinations (increase/decrease as needed)
                                       scoring='neg_root_mean_squared_error', cv=5,
                                       n_jobs=-1, verbose=1, random_state=42) # verbose=1 shows progress

    # Fit the randomized search to the training data
    try:
        random_search.fit(X_train, y_train)
    except KeyboardInterrupt:
        print("\n  Randomized search interrupted by user.")
        # If interrupted, you can choose to return None, or the best model found so far if available
        if hasattr(random_search, 'best_estimator_'):
             print("  Returning best model found before interruption.")
             best_model = random_search.best_estimator_
             predictions = best_model.predict(X_test)
             return predictions
        else:
             print("  No model found before interruption. Returning NaN predictions.")
             return np.full(X_test.shape[0], np.nan) # Return NaN predictions if interrupted early


    print("  Best RandomForestRegressor parameters found: ", random_search.best_params_)
    print("  Best cross-validation RMSE: ", -random_search.best_score_) # Score is negative RMSE, so negate for actual RMSE

    # Get the best model found by the search
    best_model = random_search.best_estimator_

    print("  Predicting with best RandomForestRegressor model...")
    predictions = best_model.predict(X_test)
    return predictions