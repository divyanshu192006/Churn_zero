import pandas as pd
import numpy as np

def engineer_features(df):
    df_out = df.copy()
    
    # 1. Product Diversity Score
    product_flags = [
        'savings_account_flag', 'current_account_flag', 'credit_card_flag', 
        'personal_loan_flag', 'home_loan_flag', 'auto_loan_flag', 
        'fixed_deposit_flag', 'investment_product_flag', 'insurance_product_flag',
        'demat_account_flag'
    ]
    # Filter to actual columns present in df
    avail_product_flags = [col for col in product_flags if col in df_out.columns]
    if len(avail_product_flags) > 0:
        df_out['product_diversity_score'] = df_out[avail_product_flags].sum(axis=1)
    else:
        df_out['product_diversity_score'] = 0
        
    # 2. Engagement Score
    # Combine digital logins and ratios
    # Columns: total_digital_logins, digital_transaction_ratio, mobile_app_login_count
    logins_col = 'total_digital_logins' if 'total_digital_logins' in df_out.columns else 'mobile_app_login_count'
    ratio_col = 'digital_transaction_ratio' if 'digital_transaction_ratio' in df_out.columns else None
    
    if logins_col in df_out.columns:
        df_out['engagement_score'] = df_out[logins_col].fillna(0)
        if ratio_col and ratio_col in df_out.columns:
            df_out['engagement_score'] = df_out['engagement_score'] * (1 + df_out[ratio_col].fillna(0))
    else:
        df_out['engagement_score'] = 0
        
    # 3. Complaint Severity Index
    # Columns: total_complaints, unresolved_complaint_count, escalation_count, satisfaction_score
    tc = df_out['total_complaints'].fillna(0) if 'total_complaints' in df_out.columns else 0
    urc = df_out['unresolved_complaint_count'].fillna(0) if 'unresolved_complaint_count' in df_out.columns else 0
    esc = df_out['escalation_count'].fillna(0) if 'escalation_count' in df_out.columns else 0
    
    # Satisfaction score: usually 1 to 5. Lower is worse.
    # If missing, assume average 3.
    if 'satisfaction_score' in df_out.columns:
        sat = df_out['satisfaction_score'].fillna(3)
        sat_impact = 6 - sat # Higher impact for lower satisfaction
    else:
        sat_impact = 0
        
    df_out['complaint_severity_index'] = urc * 3.0 + esc * 2.0 + tc * 1.0 + sat_impact * 1.5
    
    # 4. Balance Decline Trend
    # Columns: balance_decline_percentage, avg_monthly_balance, current_balance
    bd_pct = df_out['balance_decline_percentage'].fillna(0) if 'balance_decline_percentage' in df_out.columns else 0
    avg_bal = df_out['avg_monthly_balance'].fillna(0) if 'avg_monthly_balance' in df_out.columns else 0
    curr_bal = df_out['current_balance'].fillna(0) if 'current_balance' in df_out.columns else 0
    
    # Absolute drop in balance
    df_out['balance_drop_abs'] = (avg_bal - curr_bal).clip(lower=0)
    df_out['balance_decline_trend'] = bd_pct * df_out['balance_drop_abs']
    
    # 5. Digital Inactivity Score
    # Columns: last_login_days, account_inactive_days
    lld = df_out['last_login_days'].fillna(30) if 'last_login_days' in df_out.columns else 30
    aid = df_out['account_inactive_days'].fillna(30) if 'account_inactive_days' in df_out.columns else 30
    df_out['digital_inactivity_score'] = lld * 0.5 + aid * 0.5
    
    # 6. Transaction Activity Trend
    # Columns: total_ct_chng_q4_q1, monthly_transaction_count
    tc_chng = df_out['total_ct_chng_q4_q1'].fillna(1.0) if 'total_ct_chng_q4_q1' in df_out.columns else 1.0
    m_tx_ct = df_out['monthly_transaction_count'].fillna(0) if 'monthly_transaction_count' in df_out.columns else 0
    # Decline in transaction count from q4 to q1
    df_out['transaction_activity_decline'] = (1.0 - tc_chng) * m_tx_ct
    
    # 7. Additional Financial Ratios
    # Outstanding Loan to Annual Income
    loan_out = df_out['loan_outstanding_amount'].fillna(0) if 'loan_outstanding_amount' in df_out.columns else 0
    income = df_out['annual_income'].fillna(1) if 'annual_income' in df_out.columns else 1
    income = income.replace(0, 1) # Prevent division by zero
    df_out['outstanding_to_income_ratio'] = loan_out / income
    
    # Credit Utilization vs limit
    credit_limit = df_out['credit_card_limit'].fillna(0) if 'credit_card_limit' in df_out.columns else 0
    credit_spend = df_out['credit_card_spend'].fillna(0) if 'credit_card_spend' in df_out.columns else 0
    df_out['cc_spend_to_limit_ratio'] = np.where(credit_limit > 0, credit_spend / credit_limit, 0)
    
    return df_out
