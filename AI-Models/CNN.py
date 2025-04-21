# CNN.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
import numpy as np

def train_and_predict_cnn(X_train, y_train, X_test, input_shape):
    """Trains a 1D CNN model for regression and makes predictions."""
    # This function uses a fixed architecture. Tuning would involve Keras Tuner.

    # Clear any previous keras session to avoid potential conflicts (Optional)
    # tf.keras.backend.clear_session()

    # Build the CNN model using Input layer (to avoid warning)
    model = Sequential([
        Input(shape=input_shape), # Input shape defined in main.py (875, 1)
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout for regularization
        Dense(1) # Output layer for regression (predicting a single value)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    print("  Training CNN model...")
    try:
        # Use a small validation split during training to monitor overfitting
        # verbose=0 hides detailed training output
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
        print("  CNN training complete.")
    except Exception as e:
        print(f"  Error during CNN training: {e}")
        # Return placeholder predictions if training fails
        return np.full(X_test.shape[0], np.nan)

    print("  Predicting with CNN model...")
    try:
        # Ensure X_test has the correct shape (samples, 875, 1) for prediction
        if X_test.ndim == 2:
             # Reshape if it's still 2D (shouldn't be based on main.py logic, but good safeguard)
             X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        else:
             X_test_reshaped = X_test

        # verbose=0 hides detailed prediction output
        predictions = model.predict(X_test_reshaped, verbose=0).flatten() # .flatten() to make the output a 1D array
    except Exception as e:
        print(f"  Error during CNN prediction: {e}")
        return np.full(X_test.shape[0], np.nan)

    return predictions