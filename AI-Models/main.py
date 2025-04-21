# main.py

import pandas as pd
import numpy as np
# Ensure all necessary scikit-learn imports are here
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer # Import SimpleImputer for handling NaNs in engineered features


# Import functions from your custom files
from feature_engineering import create_engineered_features_matrix
# Import regression model training functions (Linear Regression doesn't have a tune function)
from Logistic_Regression import train_and_predict_lr
# Import regression model tuning functions (these handle training/predicting within them)
from SVM import tune_and_predict_svm
from Random_Forest import tune_and_predict_rf
# Import CNN regression training function
from CNN import train_and_predict_cnn

# Import classification model training functions
from classification_models import (
    train_and_predict_logistic_regression_clf,
    train_and_predict_svm_clf,
    train_and_predict_random_forest_clf
)
# Also import categories defined in classification_models
from classification_models import BP_CATEGORIES


# --- Configuration ---
# !!! IMPORTANT: Replace with the actual path to your CSV file !!!
# Example: DATA_FILE = '../your_dataset.csv' if CSV is one level up
# Example: DATA_FILE = 'data/your_dataset.csv' if CSV is in a 'data' subfolder
DATA_FILE = './ppg_labels_chunk_100_samples.csv' # Make sure this path is correct!

# Corrected Target column names
TARGET_SBP = 'SBP (True)'
TARGET_DBP = 'DBP (True)'

# Corrected PPG feature column names
PPG_FEATURE_COLUMNS = [f'PPG Channel {i}' for i in range(1, 876)]

TEST_SIZE = 0.2
RANDOM_STATE = 42 # for reproducibility

# --- Helper Function for Regression Evaluation ---
def evaluate_regression_model(y_true, y_pred, model_name, target_name):
    """Calculates and prints regression metrics."""
    # Handle potential NaN predictions or cases where y_true/y_pred are empty after filtering
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"--- {model_name} Evaluation for {target_name} (Regression) ---")
        print("  No data for evaluation.")
        print("-" * (len(model_name) + len(target_name) + 37))
        print("\n")
        return

    # Ensure y_pred is numeric to check for NaNs reliably
    # Use pd.to_numeric with errors='coerce' in case the model returned non-numeric placeholders
    y_pred_numeric = pd.to_numeric(y_pred, errors='coerce')

    valid_indices = ~pd.isna(y_pred_numeric) # Use pd.isna for robustness
    if not np.any(valid_indices):
        print(f"--- {model_name} Evaluation for {target_name} (Regression) ---")
        print("  No valid predictions available.")
        print("-" * (len(model_name) + len(target_name) + 37))
        print("\n")
        return

    # Ensure y_true is also aligned after filtering invalid predictions
    y_true_valid = np.asarray(y_true)[valid_indices] # Ensure numpy array for indexing
    y_pred_valid = np.asarray(y_pred_numeric)[valid_indices] # Ensure numpy array for indexing


    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_valid, y_pred_valid)
    mean_error = np.mean(y_pred_valid - y_true_valid)
    std_dev_error = np.std(y_pred_valid - y_true_valid)

    print(f"--- {model_name} Evaluation for {target_name} (Regression) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Standard Deviation of Error: {std_dev_error:.2f}")
    print("-" * (len(model_name) + len(target_name) + 37))
    print("\n")

# --- Helper Function for Classification Evaluation ---
def evaluate_classification_model(y_true, y_pred, model_name):
    """Calculates and prints classification metrics."""
    print(f"--- {model_name} Evaluation (Classification) ---")

    # Handle potential empty y_true/y_pred
    if len(y_true) == 0 or len(y_pred) == 0:
        print("  No data for evaluation.")
        print("-" * (len(model_name) + 30))
        print("\n")
        return

    # Ensure y_true/y_pred are numpy arrays for consistent handling
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    # Handle potential NaN/None predictions from model errors (e.g., if model training failed)
    valid_indices = ~pd.isna(y_pred_np) # Use pd.isna which handles object dtypes containing None/np.nan
    if not np.any(valid_indices):
        print("  No valid predictions available.")
        print("-" * (len(model_name) + 30))
        print("\n")
        return

    y_true_valid = y_true_np[valid_indices]
    y_pred_valid = y_pred_np[valid_indices]

    # Ensure the valid y_true only contains labels that are in the expected categories
    # and ensure y_pred also contains valid labels for the report/matrix
    # This can be complex if model predicts unexpected labels.
    # For simplicity, rely on classification_report and confusion_matrix's label handling.
    # Using target_names and labels arguments helps ensure correct output order.


    try:
        # Check if there is more than one class in y_true_valid
        # Classification metrics are not well-defined if there's only one class
        if len(np.unique(y_true_valid)) < 2:
            print("  Only one class present in true labels for evaluation. Cannot compute standard classification metrics.")
            # Still try to compute accuracy if predictions match the single class
            if len(np.unique(y_pred_valid)) == 1 and np.unique(y_true_valid)[0] == np.unique(y_pred_valid)[0]:
                 accuracy = accuracy_score(y_true_valid, y_pred_valid)
                 print(f"  Accuracy (single class): {accuracy:.2f}")
            else:
                 print("  Predictions do not match the single true class.")
            print("-" * (len(model_name) + 30))
            print("\n")
            return


        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        print(f"Accuracy: {accuracy:.2f}")

        # Use classification_report for Precision, Recall, F1-score
        # target_names should match the labels found in y_true (all possible labels)
        # The labels argument in confusion_matrix ensures rows/cols are in BP_CATEGORIES order
        # Check if all BP_CATEGORIES are present in y_true_valid or y_pred_valid
        # If a class is missing in y_true_valid and not predicted, classification_report might error
        # Pass the union of unique labels in y_true_valid and y_pred_valid to classification_report's labels argument
        # This is more robust than relying on target_names alone if some classes are missing in the split
        labels_present = np.union1d(y_true_valid, y_pred_valid)
        # Filter BP_CATEGORIES to only include labels present in this split's data
        target_names_present = [cat for cat in BP_CATEGORIES if cat in labels_present]


        report = classification_report(y_true_valid, y_pred_valid, target_names=target_names_present, labels=BP_CATEGORIES, zero_division=0)
        print("\nClassification Report:\n", report)

        # Print Confusion Matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=BP_CATEGORIES)
        print("\nConfusion Matrix:")
        # Print with column headers for better readability
        cm_df = pd.DataFrame(cm, index=[f'True: {c}' for c in BP_CATEGORIES], columns=[f'Pred: {c}' for c in BP_CATEGORIES])
        print(cm_df)


        print("-" * (len(model_name) + 30))
        print("\n")

    except Exception as e:
        print(f"Error during classification evaluation: {e}")
        # print(f"y_true_valid unique values: {np.unique(y_true_valid)}") # Helps debug label issues
        # print(f"y_pred_valid unique values: {np.unique(y_pred_valid)}")
        print("-" * (len(model_name) + 30))
        print("\n")


# --- Function to Convert Continuous BP to Categories ---
def categorize_bp(row):
    """Converts SBP and DBP values to a single BP category label."""
    sbp = row[TARGET_SBP]
    dbp = row[TARGET_DBP]

    # Handle potential missing/invalid data - these rows should ideally be dropped before this function is called
    # This check is mainly for robustness
    if pd.isna(sbp) or pd.isna(dbp):
        return np.nan # Should not happen after initial data cleaning

    # Apply guidelines (ACC/AHA 2017 simplified)
    # Start with the highest categories first
    if sbp > 180 or dbp > 120:
        return 'Hypertension Stage 2' # Including Hypertensive Crisis in Stage 2 for simplicity
    elif sbp >= 140 or dbp >= 90:
        return 'Hypertension Stage 2'
    elif sbp >= 130 or dbp >= 80:
        return 'Hypertension Stage 1'
    elif sbp >= 120 and dbp < 80:
        return 'Elevated'
    elif sbp < 120 and dbp < 80:
        return 'Normal'
    else:
        # This case should ideally not be reached with valid numeric SBP/DBP
        return np.nan # Fallback for unexpected values/conditions


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Loading data...")
    try:
        # Load the CSV
        df = pd.read_csv(DATA_FILE)
        print("Data loaded successfully.")
        print(f"Initial data shape: {df.shape}")

        # --- Ensure PPG columns are numeric and handle NaNs ---
        print("Ensuring PPG columns are numeric and handling errors...")
        # Use pd.to_numeric with errors='coerce' to convert non-numeric values to NaN
        # Apply this to the selected PPG columns
        for col in PPG_FEATURE_COLUMNS:
             # Check if column exists before trying to convert
             if col in df.columns:
                 # Convert column to numeric, coercing errors to NaN
                 # This should result in a float dtype column where np.isnan works
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                 print(f"Warning: PPG feature column '{col}' not found in the dataset. Skipping coercion for this column.")
                 # Optionally, handle missing columns more strictly if they are essential


        # Identify rows with any NaN in PPG features OR NaN in SBP/DBP targets
        # Use .isnull().any(axis=1) to find rows with at least one NaN in the specified columns
        rows_with_any_nan_ppg = df[PPG_FEATURE_COLUMNS].isnull().any(axis=1)
        rows_with_nan_targets = df[TARGET_SBP].isnull() | df[TARGET_DBP].isnull()

        rows_to_drop = rows_with_any_nan_ppg | rows_with_nan_targets

        if rows_to_drop.any():
            initial_rows = len(df)
            df = df[~rows_to_drop].reset_index(drop=True)
            print(f"Dropped {initial_rows - len(df)} rows due to missing/invalid PPG features or targets.")
            print(f"New data shape: {df.shape}")

        # Ensure the DataFrame is not empty after dropping
        if df.empty:
             print("Error: No valid data remaining after preprocessing. Exiting.")
             exit()


    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_FILE}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading or initial preprocessing: {e}")
        exit()

    # --- Define Features (Original 875 PPG channels) and targets from the cleaned DataFrame ---
    try:
        # Select features and targets from the cleaned DataFrame.
        # PPG feature columns are now guaranteed numeric (or NaNs handled by dropping rows).
        X_original = df[PPG_FEATURE_COLUMNS]
        y_sbp = df[TARGET_SBP]
        y_dbp = df[TARGET_DBP]
        print("Original features and continuous targets defined from processed data.")
    except KeyError as e:
        print(f"Error: Missing target columns after preprocessing. Please check TARGET_SBP/TARGET_DBP names.")
        print(f"Details: {e}")
        print("Columns found in cleaned DataFrame:", df.columns.tolist())
        exit()


    # --- Convert Continuous BP to Categorical Target ---
    print("Categorizing continuous blood pressure...")
    # Apply the categorization function to each row to create a new target column
    # This is applied to the DataFrame after dropping rows with NaN targets, so it should work.
    df['BP_Category'] = df.apply(categorize_bp, axis=1)
    y_category = df['BP_Category'] # This should not have NaNs if SBP/DBP were not NaN and categorize_bp is deterministic

    # Check class distribution (important for classification) - use the updated y_category
    print("\nBlood Pressure Category Distribution in Dataset:")
    # Check if y_category is a pandas Series before calling value_counts
    if isinstance(y_category, pd.Series):
        print(y_category.value_counts())
    else:
        print("Could not determine category distribution (y_category is not a pandas Series).")
    print("\n")


    # --- Scaling and Splitting ORIGINAL features (for CNN Regression) ---
    print("Scaling original 875 features...")
    scaler_original = StandardScaler()
    X_scaled_original = scaler_original.fit_transform(X_original)

    # --- Split data for REGRESSION (using original scaled features and continuous targets) ---
    # This split is used by the CNN
    print(f"Splitting data for REGRESSION ({X_scaled_original.shape[1]} features) into {100*(1-TEST_SIZE)}% training and {100*TEST_SIZE}% testing...")
    # Ensure consistent split using the SAME random state
    X_train_reg, X_test_reg, y_train_sbp_reg, y_test_sbp_reg = train_test_split(X_scaled_original, y_sbp, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Split DBP targets using the same split
    _, _, y_train_dbp_reg, y_test_dbp_reg = train_test_split(X_scaled_original, y_dbp, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Ensure the y_train/test variables are numpy arrays for easier handling in model functions
    y_train_sbp_reg_np = np.asarray(y_train_sbp_reg)
    y_test_sbp_reg_np = np.asarray(y_test_sbp_reg)
    y_train_dbp_reg_np = np.asarray(y_train_dbp_reg)
    y_test_dbp_reg_np = np.asarray(y_test_dbp_reg)

    # --- Reshape the original scaled features for CNN input AND DEFINE input_shape_cnn HERE ---
    # Paste the reshaping and input_shape_cnn calculation lines here:
    X_train_cnn_input = X_train_reg.reshape(X_train_reg.shape[0], X_train_reg.shape[1], 1)
    X_test_cnn_input = X_test_reg.reshape(X_test_reg.shape[0], X_test_reg.shape[1], 1)
    input_shape_cnn = (X_train_cnn_input.shape[1], X_train_cnn_input.shape[2]) # (875, 1)
    print(f"CNN training input shape: {X_train_cnn_input.shape}")
    # The variable input_shape_cnn is now defined in the correct scope and time


    # --- Feature Engineering for flat feature models (SVM, RF, LR - both Regression and Classification) ---
    print("Engineering features...")
    # Engineer features from the cleaned DataFrame
    X_engineered = create_engineered_features_matrix(df, PPG_FEATURE_COLUMNS)
    print(f"Created engineered features matrix with shape: {X_engineered.shape}")

    # Check for NaNs introduced during feature engineering
    if X_engineered.isnull().sum().sum() > 0:
        print("\nWarning: NaN values were introduced during feature engineering.")
        print("Rows with NaNs in engineered features:", X_engineered.isnull().any(axis=1).sum())
        # Decide how to handle: drop these rows, or impute NaNs in engineered features
        # Imputation is handled below before scaling

    # --- Scaling Engineered Features ---
    print("Scaling engineered features...")
    scaler_engineered = StandardScaler()
    # Handle potential NaNs in X_engineered before scaling! StandardScaler does NOT handle NaNs.
    # Let's add imputation using the mean as a simple strategy if NaNs exist after engineering
    if X_engineered.isnull().sum().sum() > 0:
         imputer = SimpleImputer(strategy='mean')
         # Impute NaNs using the imputer fitted on the training part of the data (conceptual here,
         # in a real pipeline you'd fit on train and transform train/test separately)
         # For this script's flow, fitting on the whole engineered set before splitting is simpler,
         # but can cause data leakage if NaNs are based on test set statistics.
         # A robust pipeline would impute after the split.
         # Let's simplify and impute before splitting engineered features.
         X_engineered_imputed = imputer.fit_transform(X_engineered)
         X_scaled_engineered = scaler_engineered.fit_transform(X_engineered_imputed)
         print("  NaNs in engineered features imputed with column mean before scaling and splitting.")
    else:
         # If no NaNs, just scale directly
         X_scaled_engineered = scaler_engineered.fit_transform(X_engineered)


    # --- Split data for CLASSIFICATION and Engineered Features REGRESSION ---
    # This split is used by SVM, RF, LR (Regression and Classification)
    print(f"Splitting scaled engineered data ({X_scaled_engineered.shape[1]} features) into {100*(1-TEST_SIZE)}% training and {100*TEST_SIZE}% testing...")
    # Split scaled engineered features and corresponding targets (both continuous and categorical)
    # Ensure consistent split using the SAME random state
    # Note: These splits are from the (potentially imputed and) scaled engineered data.
    X_train_eng, X_test_eng, y_train_sbp_eng, y_test_sbp_eng = train_test_split(X_scaled_engineered, y_sbp, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, _, y_train_dbp_eng, y_test_dbp_eng = train_test_split(X_scaled_engineered, y_dbp, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Make sure y_category is aligned with X_engineered (which came from df) and split consistently
    _, _, y_train_category, y_test_category = train_test_split(X_scaled_engineered, y_category, test_size=TEST_SIZE, random_state=RANDOM_STATE)


    # Ensure the y_train/test variables are numpy arrays for easier handling in model functions
    y_train_sbp_eng_np = np.asarray(y_train_sbp_eng)
    y_test_sbp_eng_np = np.asarray(y_test_sbp_eng)
    y_train_dbp_eng_np = np.asarray(y_train_dbp_eng)
    y_test_dbp_eng_np = np.asarray(y_test_dbp_eng)
    y_train_category_np = np.asarray(y_train_category)
    y_test_category_np = np.asarray(y_test_category)


    print("\n--- Starting REGRESSION Model Training and Evaluation ---")

    # --- Train and Evaluate SVM Regression (using engineered features + Tuning) ---
    print("Training SVM Regression...")
    # Pass the engineered splits and REGRESSION targets to the tuning function
    y_pred_svm_sbp_reg = tune_and_predict_svm(X_train_eng, y_train_sbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_sbp_eng_np, y_pred_svm_sbp_reg, "SVM (Tuned, Engineered Features)", "SBP")

    y_pred_svm_dbp_reg = tune_and_predict_svm(X_train_eng, y_train_dbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_dbp_eng_np, y_pred_svm_dbp_reg, "SVM (Tuned, Engineered Features)", "DBP")


    # --- Train and Evaluate Random Forest Regression (using engineered features + Tuning) ---
    print("Training Random Forest Regression...")
    # Pass the engineered splits and REGRESSION targets to the tuning function
    y_pred_rf_sbp_reg = tune_and_predict_rf(X_train_eng, y_train_sbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_sbp_eng_np, y_pred_rf_sbp_reg, "Random Forest (Tuned, Engineered Features)", "SBP")

    y_pred_rf_dbp_reg = tune_and_predict_rf(X_train_eng, y_train_dbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_dbp_eng_np, y_pred_rf_dbp_reg, "Random Forest (Tuned, Engineered Features)", "DBP")


    # --- Train and Evaluate Linear Regression (using engineered features, no tuning here) ---
    print("Training Linear Regression...")
    # Pass the engineered splits and REGRESSION targets
    y_pred_lr_sbp_reg = train_and_predict_lr(X_train_eng, y_train_sbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_sbp_eng_np, y_pred_lr_sbp_reg, "Linear Regression (Engineered Features)", "SBP")

    y_pred_lr_dbp_reg = train_and_predict_lr(X_train_eng, y_train_dbp_eng_np, X_test_eng)
    evaluate_regression_model(y_test_dbp_eng_np, y_pred_lr_dbp_reg, "Linear Regression (Engineered Features)", "DBP")


    # --- Train and Evaluate CNN Regression (using original 875 scaled features) ---
    print("Training CNN Regression (using original 875 features)...")
    # Pass the REGRESSION splits (using original scaled features) and REGRESSION targets
    # input_shape_cnn is defined above the training call
    y_pred_cnn_sbp_reg = train_and_predict_cnn(X_train_cnn_input, y_train_sbp_reg_np, X_test_cnn_input, input_shape_cnn)
    evaluate_regression_model(y_test_sbp_reg_np, y_pred_cnn_sbp_reg, "CNN (Original Features)", "SBP")

    y_pred_cnn_dbp_reg = train_and_predict_cnn(X_train_cnn_input, y_train_dbp_reg_np, X_test_cnn_input, input_shape_cnn)
    evaluate_regression_model(y_test_dbp_reg_np, y_pred_cnn_dbp_reg, "CNN (Original Features)", "DBP")


    print("\n--- Starting CLASSIFICATION Model Training and Evaluation ---")

    # --- Train and Evaluate Logistic Regression Classifier (using engineered features) ---
    print("Training Logistic Regression Classifier...")
    # Pass the engineered splits (X_train_eng, X_test_eng) and CLASSIFICATION targets (y_train_category_np)
    y_pred_lr_clf = train_and_predict_logistic_regression_clf(X_train_eng, y_train_category_np, X_test_eng) # Corrected variables
    evaluate_classification_model(y_test_category_np, y_pred_lr_clf, "Logistic Regression")

    # --- Train and Evaluate SVM Classifier (using engineered features) ---
    print("Training SVC (SVM Classifier)...")
    # Pass the engineered splits (X_train_eng, X_test_eng) and CLASSIFICATION targets (y_train_category_np)
    y_pred_svm_clf = train_and_predict_svm_clf(X_train_eng, y_train_category_np, X_test_eng) # Corrected variables
    evaluate_classification_model(y_test_category_np, y_pred_svm_clf, "SVC (SVM Classifier)")

    # --- Train and Evaluate Random Forest Classifier (using engineered features) ---
    print("Training Random Forest Classifier...")
    # Pass the engineered splits (X_train_eng, X_test_eng) and CLASSIFICATION targets (y_train_category_np)
    y_pred_rf_clf = train_and_predict_random_forest_clf(X_train_eng, y_train_category_np, X_test_eng) # Corrected variables
    evaluate_classification_model(y_test_category_np, y_pred_rf_clf, "Random Forest Classifier")

    # Note: To add CNN Classification, you would need to modify CNN.py to be a classifier
    # and add calls here using the original scaled features split (X_train_reg, X_test_reg)
    # and the classification target (y_train_category_np, y_test_category_np)


    print("\nModel training and evaluation complete.")
    print("\n--- Reasoning News about the Dataset (from Classification) ---")
    print("The classification results show how well the models can distinguish between different blood pressure categories.")
    print("Look at the Accuracy, Classification Report (Precision, Recall, F1-score per class), and Confusion Matrix.")
    print("A high Accuracy or good F1-scores for certain classes indicates the models are better at identifying those categories.")
    print("The Confusion Matrix shows where the models are making mistakes (e.g., misclassifying Stage 1 as Normal).")
    print("This gives insight into whether the PPG features contain enough information to differentiate between BP states.")
    print("Comparing the classification accuracy to the regression error metrics (MAE, RMSE) provides a fuller picture of performance.")
    print("A high classification accuracy alongside high regression error might suggest the models can roughly group BP but struggle with precise numerical prediction.")