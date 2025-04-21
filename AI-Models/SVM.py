# SVM.py
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def train_and_predict_svm(X_train, y_train, X_test):
    """Trains an SVR model and makes predictions (basic version)."""
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def tune_and_predict_svm(X_train, y_train, X_test):
    """Performs GridSearchCV for SVR and makes predictions with the best model."""
    print("  Tuning SVR model...")

    model = SVR()

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1, verbose=0)

    grid_search.fit(X_train, y_train)

    print("  Best SVR parameters found: ", grid_search.best_params_)
    print("  Best cross-validation RMSE: ", -grid_search.best_score_)

    best_model = grid_search.best_estimator_

    print("  Predicting with best SVR model...")
    predictions = best_model.predict(X_test)
    return predictions