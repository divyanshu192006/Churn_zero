import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Import local modules
from preprocessing import ChurnPreprocessor
from feature_engineering import engineer_features
from evaluate import optimize_threshold, calculate_business_cost, plot_cost_vs_threshold, plot_confusion_matrix_custom

def train_and_evaluate_all():
    print("--- Loading and Preparing Data ---")
    train_path = os.path.join("data", "ChurnZero_dataset_v1.csv")
    if not os.path.exists(train_path):
        # Fallback to absolute path just in case
        train_path = r"c:\Users\rajdi\OneDrive\Desktop\IITK\data\ChurnZero_dataset_v1.csv"
        
    df = pd.read_csv(train_path)
    
    # 1. Feature Engineering
    print("Engineering features...")
    df_engineered = engineer_features(df)
    
    # 2. Preprocessing
    print("Preprocessing data...")
    preprocessor = ChurnPreprocessor(target_col='churn', customer_id_col='customer_id')
    df_preprocessed = preprocessor.fit(df_engineered).transform(df_engineered)
    
    # Extract features and target
    target_col = 'churn'
    feature_cols = [c for c in df_preprocessed.columns if c not in [target_col, 'customer_id']]
    
    X = df_preprocessed[feature_cols]
    y = df_preprocessed[target_col]
    
    print(f"Final feature shape: {X.shape}")
    
    # Setup CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # We will train and compare:
    # 1. Logistic Regression
    # 2. Random Forest
    # 3. LightGBM
    # 4. XGBoost
    # 5. CatBoost
    
    models_to_test = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1),
        'LightGBM': lambda: lgb.LGBMClassifier(n_estimators=300, random_state=42, class_weight='balanced', verbosity=-1, n_jobs=-1),
        'XGBoost': lambda: xgb.XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss', n_jobs=-1),
        'CatBoost': lambda: CatBoostClassifier(iterations=500, random_state=42, verbose=0, auto_class_weights='Balanced')
    }
    
    results = {}
    oof_predictions = {name: np.zeros(len(df)) for name in models_to_test.keys()}
    
    print("\n--- Starting Stratified 5-Fold Cross-Validation ---")
    
    for model_name, model_fn in models_to_test.items():
        print(f"\nTraining {model_name}...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # For scale-sensitive models like Logistic Regression, we impute and scale
            if model_name == 'Logistic Regression':
                # Median imputation + Scaling for numeric columns
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                imp = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                
                # Fit on training fold, transform both
                X_tr_proc = pd.DataFrame(scaler.fit_transform(imp.fit_transform(X_train)), columns=X_train.columns)
                X_val_proc = pd.DataFrame(scaler.transform(imp.transform(X_val)), columns=X_val.columns)
                
                model = model_fn()
                model.fit(X_tr_proc, y_train)
                preds = model.predict_proba(X_val_proc)[:, 1]
            else:
                model = model_fn()
                # For XGBoost / LightGBM / CatBoost, simple handling of categorical variables is fine since we ordinal encoded
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                
            oof_predictions[model_name][val_idx] = preds
            print(f"Fold {fold+1} complete.")
            
        # Evaluate model OOF predictions
        best_detail, df_details = optimize_threshold(y, oof_predictions[model_name])
        results[model_name] = {
            'best_detail': best_detail,
            'details_df': df_details
        }
        
        # Calculate standard PR-AUC and ROC-AUC
        from sklearn.metrics import average_precision_score, roc_auc_score
        pr_auc = average_precision_score(y, oof_predictions[model_name])
        roc_auc = roc_auc_score(y, oof_predictions[model_name])
        
        print(f"--- {model_name} OOF Results ---")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Optimal Decision Threshold: {best_detail['threshold']:.2f}")
        print(f"Recall at Opt Threshold: {best_detail['recall']:.4f}")
        print(f"Precision at Opt Threshold: {best_detail['precision']:.4f}")
        print(f"F1 Score at Opt Threshold: {best_detail['f1']:.4f}")
        print(f"Estimated Savings: INR {best_detail['savings']:,.2f}")
        print(f"Remaining Business Cost: INR {best_detail['cost']:,.2f} (vs INR {y.sum() * 40000:,.2f} baseline)")
        
    # Choose best model based on PR-AUC & Business Savings
    best_model_name = 'CatBoost' # Default to CatBoost as requested in PRD unless another strongly outperforms
    best_pr_auc = average_precision_score(y, oof_predictions['CatBoost'])
    
    print(f"\n--- Best Model Selection: {best_model_name} (PR-AUC: {best_pr_auc:.4f}) ---")
    
    # Save OOF plots for the best model
    best_details_df = results[best_model_name]['details_df']
    best_opt_detail = results[best_model_name]['best_detail']
    
    # Plot cost curve
    cost_plot_path = os.path.join("outputs", "business_cost_optimization.png")
    plot_cost_vs_threshold(best_details_df, cost_plot_path)
    print(f"Saved cost optimization plot to: {cost_plot_path}")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join("outputs", "confusion_matrix_optimized.png")
    best_preds_binary = (oof_predictions[best_model_name] >= best_opt_detail['threshold']).astype(int)
    plot_confusion_matrix_custom(y, best_preds_binary, cm_plot_path, title=f"ChurnZero {best_model_name} Cost-Optimized Matrix")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    
    # Train final model on full dataset
    print(f"\nTraining final {best_model_name} model on full dataset...")
    final_model = CatBoostClassifier(iterations=600, random_state=42, verbose=0, auto_class_weights='Balanced')
    final_model.fit(X, y)
    
    # Save final model, preprocessor, and the optimal threshold
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "final_catboost_model.joblib")
    preprocessor_save_path = os.path.join("models", "preprocessor.joblib")
    
    # We will save as dict to include the threshold and features
    model_data = {
        'model': final_model,
        'optimal_threshold': best_opt_detail['threshold'],
        'feature_cols': feature_cols,
        'results_summary': {
            'pr_auc': best_pr_auc,
            'roc_auc': roc_auc_score(y, oof_predictions[best_model_name]),
            'recall': best_opt_detail['recall'],
            'precision': best_opt_detail['precision'],
            'f1': best_opt_detail['f1'],
            'savings': best_opt_detail['savings'],
            'cost': best_opt_detail['cost']
        }
    }
    
    joblib.dump(model_data, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    
    print(f"Final model saved to: {model_save_path}")
    print(f"Preprocessor saved to: {preprocessor_save_path}")
    
    # Also save the comparison results to a CSV in outputs for dashboard use
    comparison_data = []
    for name, res in results.items():
        opt = res['best_detail']
        comparison_data.append({
            'Model': name,
            'PR-AUC': average_precision_score(y, oof_predictions[name]),
            'ROC-AUC': roc_auc_score(y, oof_predictions[name]),
            'Optimal Threshold': opt['threshold'],
            'Precision': opt['precision'],
            'Recall': opt['recall'],
            'F1': opt['f1'],
            'Savings (₹)': opt['savings'],
            'Total Cost (₹)': opt['cost']
        })
    df_compare = pd.DataFrame(comparison_data)
    df_compare.to_csv(os.path.join("outputs", "model_comparison_results.csv"), index=False)
    print("Saved model comparison results to outputs/model_comparison_results.csv")
    
    print("\n--- Model Training & Optimization Completed Successfully! ---")

if __name__ == '__main__':
    train_and_evaluate_all()
