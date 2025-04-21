# Linear_Regression.py
from sklearn.linear_model import LinearRegression

def train_and_predict_lr(X_train, y_train, X_test):
    """Trains a LinearRegression model and makes predictions."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions