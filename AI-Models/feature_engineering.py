# feature_engineering.py (Attempting to fix TypeError with extra coercion)
import numpy as np
import pandas as pd # Ensure pandas is imported

def extract_ppg_features(ppg_data_row):
    """
    Extracts hand-crafted features from a single row of PPG data (875 points).

    Args:
        ppg_data_row (np.ndarray or pd.Series): A 1D array/series containing 875 PPG values.

    Returns:
        dict: A dictionary where keys are feature names and values are the calculated features.
    """
    # --- Safeguard: Ensure data is numeric, coercing errors to NaN ---
    # Convert to pandas Series first to use pd.to_numeric reliably on potentially mixed data
    # This is redundant if the main.py loading worked, but acts as a safety net
    ppg_series = pd.Series(ppg_data_row)
    ppg_numeric_series = pd.to_numeric(ppg_series, errors='coerce')

    # Convert to numpy array (should now be float dtype if conversion worked)
    ppg_values = np.asarray(ppg_numeric_series)

    # Optional: Print the dtype of the array right here to inspect what it is
    # print(f"  Debug: ppg_values dtype inside extract_ppg_features: {ppg_values.dtype}")


    # Handle cases with entirely NaN or empty rows gracefully
    # Use pd.isna which is generally more robust than np.isnan on potentially mixed dtypes
    # if len(ppg_values) == 0 or np.all(np.isnan(ppg_values)): # Old line causing error
    if len(ppg_values) == 0 or np.all(pd.isna(ppg_values)): # Check if array is empty OR all values are considered missing by pandas
        feature_names = [
            'ppg_min', 'ppg_max', 'ppg_mean', 'ppg_std_dev', 'ppg_amplitude',
            'ppg_min_idx', 'ppg_max_idx', 'ppg_duration_idx_diff', 'ppg_area_sum',
            'ppg_max_deriv1', 'ppg_min_deriv1', 'ppg_mean_deriv1',
            'ppg_max_deriv2', 'ppg_min_deriv2',
            'ppg_time_to_peak_idx', 'ppg_time_peak_to_valley_idx'
            # Add names for any new features you add
        ]
        return {name: np.nan for name in feature_names}

    # --- Simple Time-Domain Features ---
    # Use nan* functions to ignore NaNs if any exist within a row
    # These functions require the array to be of a float dtype, which should be true after coercion above
    min_val = np.nanmin(ppg_values) if np.any(~np.isnan(ppg_values)) else np.nan
    max_val = np.nanmax(ppg_values) if np.any(~np.isnan(ppg_values)) else np.nan
    mean_val = np.nanmean(ppg_values) if np.any(~np.isnan(ppg_values)) else np.nan
    std_dev_val = np.nanstd(ppg_values) if np.any(~np.isnan(ppg_values)) else np.nan
    amplitude = max_val - min_val if not pd.isna(min_val) and not pd.isna(max_val) else np.nan


    # Find indices of min and max (simple peak/valley detection)
    # argmin/argmax *can* be sensitive to NaNs. A robust way is to operate on valid indices or use signal processing libs.
    # For simplicity, let's ensure we only try if there are non-NaN values
    if np.any(~np.isnan(ppg_values)):
        # Operate on the non-NaN subset for finding min/max index
        non_nan_indices = np.where(~np.isnan(ppg_values))[0]
        if len(non_nan_indices) > 0:
             min_idx_relative = np.argmin(ppg_values[non_nan_indices]) # Index within non-NaN values
             max_idx_relative = np.argmax(ppg_values[non_nan_indices]) # Index within non-NaN values
             min_idx = non_nan_indices[min_idx_relative] # Original index
             max_idx = non_nan_indices[max_idx_relative] # Original index
        else: # Should be caught by the all-NaN check earlier, but safety
             min_idx = np.nan
             max_idx = np.nan
    else:
        min_idx = np.nan
        max_idx = np.nan

    # Simple duration surrogate (index difference)
    duration_idx_diff = abs(max_idx - min_idx) if not pd.isna(min_idx) and not pd.isna(max_idx) else np.nan

    # Area under the curve (simple sum)
    area_sum = np.nansum(ppg_values) # Use nansum to sum non-NaN values

    # --- Basic Waveform Features (using numpy) ---

    # First derivative (proxy for slope) - only calculate if there's more than one non-NaN point
    if np.sum(~np.isnan(ppg_values)) > 1:
        deriv1 = np.diff(ppg_values)
        max_deriv1 = np.nanmax(deriv1) if np.any(~np.isnan(deriv1)) else np.nan
        min_deriv1 = np.nanmin(deriv1) if np.any(~np.isnan(deriv1)) else np.nan
        mean_deriv1 = np.nanmean(deriv1) if np.any(~np.isnan(deriv1)) else np.nan
    else:
        # Handle case with 1 or 0 non-NaN points where derivative is undefined
        deriv1 = np.array([]) # Empty array
        max_deriv1 = min_deriv1 = mean_deriv1 = np.nan


    # Second derivative - only calculate if there's more than two non-NaN points
    # Need at least 2 points for 1st derivative, and at least 2 points in 1st derivative for 2nd derivative
    if len(deriv1) > 1 and np.sum(~np.isnan(deriv1)) > 1:
        deriv2 = np.diff(deriv1)
        max_deriv2 = np.nanmax(deriv2) if np.any(~np.isnan(deriv2)) else np.nan
        min_deriv2 = np.nanmin(deriv2) if np.any(~np.isnan(deriv2)) else np.nan
    else:
        max_deriv2 = min_deriv2 = np.nan

    # Timing features based on indices (relative timing if sampling rate is constant)
    time_to_peak_idx = max_idx
    time_peak_to_valley_idx = min_idx - max_idx if not pd.isna(min_idx) and not pd.isna(max_idx) and min_idx > max_idx else 0


    features = {
        'ppg_min': min_val,
        'ppg_max': max_val,
        'ppg_mean': mean_val,
        'ppg_std_dev': std_dev_val,
        'ppg_amplitude': amplitude,
        'ppg_min_idx': min_idx,
        'ppg_max_idx': max_idx,
        'ppg_duration_idx_diff': duration_idx_diff,
        'ppg_area_sum': area_sum,
        'ppg_max_deriv1': max_deriv1,
        'ppg_min_deriv1': min_deriv1,
        'ppg_mean_deriv1': mean_deriv1,
        'ppg_max_deriv2': max_deriv2,
        'ppg_min_deriv2': min_deriv2,
        'ppg_time_to_peak_idx': time_to_peak_idx,
        'ppg_time_peak_to_valley_idx': time_peak_to_valley_idx,
        # Add more features here!
    }

    # Ensure no infinities are produced (e.g., from division by zero if you add more features)
    features = {k: v if np.isfinite(v) else np.nan for k, v in features.items()}


    return features

# Keep create_engineered_features_matrix function as is in this file
def create_engineered_features_matrix(dataframe, ppg_column_names):
    """
    Applies feature extraction to each row of PPG data in a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame loaded from the CSV (should have PPG columns numeric after coercion).
        ppg_column_names (list): A list of column names containing the PPG data points.

    Returns:
        pd.DataFrame: A DataFrame where each row is a sample and columns are the engineered features.
    """
    print("  Extracting features row by row...")
    engineered_features_list = []
    # Use tqdm for a progress bar if you have many samples: pip install tqdm
    # from tqdm import tqdm
    # for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Engineering Features"):
    for index, row in dataframe.iterrows():
        # Select only the PPG columns for feature extraction for this row
        ppg_data_row = row[ppg_column_names]
        features = extract_ppg_features(ppg_data_row)
        engineered_features_list.append(features)

    # Create a new DataFrame from the list of feature dictionaries
    engineered_df = pd.DataFrame(engineered_features_list)

    # Note: NaN handling after engineering is primarily done in main.py by dropping rows
    # with any NaN PPG features during initial loading. If your extract function
    # produces NaNs for valid inputs, you might need additional NaN handling here.

    return engineered_df