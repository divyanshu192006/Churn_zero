import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

class ChurnPreprocessor:
    def __init__(self, target_col='churn', customer_id_col='customer_id'):
        self.target_col = target_col
        self.customer_id_col = customer_id_col
        self.categorical_cols = []
        self.numerical_cols = []
        self.encoder = None
        self.imputer = None
        self.scaler = None
        
    def fit(self, df):
        # Identify columns
        feature_cols = [c for c in df.columns if c not in [self.target_col, self.customer_id_col]]
        
        # Identify object (categorical) and numerical columns
        self.categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = df[feature_cols].select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        print(f"Detected {len(self.categorical_cols)} categorical columns: {self.categorical_cols}")
        print(f"Detected {len(self.numerical_cols)} numerical columns: {self.numerical_cols[:10]}...")
        
        # Fit ordinal encoder on categorical columns
        if len(self.categorical_cols) > 0:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fill NAs in categorical with 'Missing' before fitting
            cat_df = df[self.categorical_cols].fillna('Missing')
            self.encoder.fit(cat_df)
            
        # Fit numerical imputer
        # Note: We will handle app_rating_given specially, but also run simple imputer for other numerical columns if any
        if len(self.numerical_cols) > 0:
            self.imputer = SimpleImputer(strategy='median')
            self.imputer.fit(df[self.numerical_cols])
            
            self.scaler = StandardScaler()
            self.scaler.fit(self.imputer.transform(df[self.numerical_cols]))
            
        return self
        
    def transform(self, df):
        df_out = df.copy()
        
        # 1. Handle app_rating_given specially
        # Add indicator for missing rating since it represents ~56% of data and is highly predictive of non-engagement
        if 'app_rating_given' in df_out.columns:
            df_out['app_rating_missing'] = df_out['app_rating_given'].isnull().astype(int)
            df_out['app_rating_given'] = df_out['app_rating_given'].fillna(-1)
            
        # 2. Encode categorical columns
        if len(self.categorical_cols) > 0:
            cat_df = df_out[self.categorical_cols].fillna('Missing')
            df_out[self.categorical_cols] = self.encoder.transform(cat_df)
            
        # 3. Impute and scale numerical columns
        if len(self.numerical_cols) > 0:
            # Avoid overwriting app_rating_given if it was processed already, but make sure other numericals are imputed
            cols_to_impute = [c for c in self.numerical_cols if c != 'app_rating_given']
            if len(cols_to_impute) > 0:
                # Temporary sub-imputer for robustness
                sub_imputer = SimpleImputer(strategy='median')
                df_out[cols_to_impute] = sub_imputer.fit_transform(df_out[cols_to_impute])
                
        return df_out
        
    def get_feature_lists(self):
        return self.categorical_cols, self.numerical_cols
