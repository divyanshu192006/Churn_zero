import pandas as pd
import numpy as np
import os
import joblib

# Import local modules
from feature_engineering import engineer_features

def generate_predictions():
    print("--- Loading Model and Preprocessor ---")
    model_path = os.path.join("models", "final_catboost_model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or Preprocessor files not found. Please run train.py first!")
        
    model_data = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    final_model = model_data['model']
    optimal_threshold = model_data['optimal_threshold']
    feature_cols = model_data['feature_cols']
    
    print(f"Loaded CatBoost model. Optimal decision threshold: {optimal_threshold:.4f}")
    
    print("\n--- Loading and Processing Test Data ---")
    test_path = os.path.join("data", "ChurnZero_test_v1.csv")
    if not os.path.exists(test_path):
        test_path = r"c:\Users\rajdi\OneDrive\Desktop\IITK\data\ChurnZero_test_v1.csv"
        
    df_test = pd.read_csv(test_path)
    print(f"Loaded test dataset. Shape: {df_test.shape}")
    
    # 1. Feature Engineering
    df_test_engineered = engineer_features(df_test)
    
    # 2. Preprocessing
    df_test_preprocessed = preprocessor.transform(df_test_engineered)
    
    # Extract features
    X_test = df_test_preprocessed[feature_cols]
    
    print("\n--- Generating Predictions ---")
    # Predict probabilities
    probs = final_model.predict_proba(X_test)[:, 1]
    
    # Apply optimal threshold to make cost-sensitive predictions
    preds = (probs >= optimal_threshold).astype(int)
    
    # Create final prediction DataFrame
    df_pred = pd.DataFrame({
        'customer_id': df_test['customer_id'],
        'churn_prediction': preds,
        'churn_probability': probs
    })
    
    # --- Assertions and Validations ---
    print("\n--- Running Quality Checks & Validations ---")
    # 1. Row count check
    expected_rows = 2026
    assert len(df_pred) == expected_rows, f"ERROR: Predicted rows count is {len(df_pred)}, expected {expected_rows}."
    print(f"[PASSED] Row count is exactly {expected_rows}.")
    
    # 2. Column names check
    expected_cols = ['customer_id', 'churn_prediction', 'churn_probability']
    assert list(df_pred.columns) == expected_cols, f"ERROR: Columns are {list(df_pred.columns)}, expected {expected_cols}."
    print("[PASSED] Columns match schema exactly.")
    
    # 3. Missing values check
    null_count = df_pred.isnull().sum().sum()
    assert null_count == 0, f"ERROR: Prediction contains {null_count} null value(s)."
    print("[PASSED] No null values found in predictions.")
    
    # 4. Probability range check
    prob_min = df_pred['churn_probability'].min()
    prob_max = df_pred['churn_probability'].max()
    assert 0.0 <= prob_min <= prob_max <= 1.0, f"ERROR: Probabilities outside [0, 1] bounds: min={prob_min:.4f}, max={prob_max:.4f}"
    print(f"[PASSED] Probabilities are within valid ranges [0.0, 1.0]: min={prob_min:.4f}, max={prob_max:.4f}.")
    
    # 5. Prediction labels check
    unique_preds = df_pred['churn_prediction'].unique()
    assert all(p in [0, 1] for p in unique_preds), f"ERROR: Invalid prediction labels found: {unique_preds}"
    print(f"[PASSED] Prediction labels are strictly binary (0 or 1): {unique_preds}.")
    
    # Summary of predicted classes
    pred_counts = df_pred['churn_prediction'].value_counts()
    print(f"Predicted Class Distribution:\n{pred_counts}")
    print(f"Predicted Churn Rate: {pred_counts.get(1, 0) / len(df_pred):.2%}")
    
    # Save predictions
    output_path = os.path.join("outputs", "ChurnZero_Antigravity_Predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pred.to_csv(output_path, index=False)
    
    print(f"\nPredictions successfully written to: {output_path}")
    print("--- Done! ---")

if __name__ == '__main__':
    generate_predictions()
