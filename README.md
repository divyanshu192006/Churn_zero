# ChurnZero – AI-Powered Banking Customer Churn Prediction System

An end-to-end, production-ready machine learning framework designed to predict customer churn in retail banking. The system is built with a **cost-sensitive optimization objective** to minimize the bank's financial losses (where losing an actual churner costs ₹40,000 in lost LTV, whereas proactively targeting a customer for retention costs ₹500).

---

## 🏛️ Business Challenge & Cost Framework
Standard machine learning models optimize for raw accuracy or standard F1 score at a default threshold of `0.5`. In customer churn, this is highly sub-optimal due to extreme cost imbalances:
* **False Negative (FN) Cost = ₹40,000** (losing a customer who actually churned)
* **False Positive (FP) Cost = ₹500** (offering a proactive retention incentive to a loyal customer)

### The 80x Cost Factor
Since a False Negative is **80 times more expensive** than a False Positive, the optimal decision boundary is mathematically much lower than `0.5`. ChurnZero runs an out-of-fold grid search to find the threshold $t$ that minimizes the total business loss:
$$\text{Total Cost}(t) = \text{FN}(t) \times 40,000 + \text{FP}(t) \times 500$$
By shifting the threshold to this cost-optimal point (historically around `0.07`), we capture over **85%+ of actual churners** and reduce total losses by **81%+**, saving millions in capital.

---

## 📂 Project Repository Structure
```
/project (c:\Users\rajdi\OneDrive\Desktop\IITK)
├── data/
│   ├── ChurnZero_dataset_v1.csv              # Training data (8,101 rows)
│   └── ChurnZero_test_v1.csv                 # Testing data (2,026 rows)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb    # Structured EDA and visualizations
│   └── 02_model_explainability.ipynb         # SHAP global and local attribution
├── src/
│   ├── preprocessing.py                      # Reusable, leak-free cleaning & scaling
│   ├── feature_engineering.py                # Derived metrics (engagement, severity)
│   ├── train.py                              # Stratified 5-Fold CV & optimization
│   ├── evaluate.py                           # Cost optimization & plotting utils
│   ├── predict.py                            # Quality-checked batch prediction script
│   └── dashboard.py                          # Streamlit prevention hub app
├── models/
│   ├── final_catboost_model.joblib           # Saved CatBoost binary & threshold
│   └── preprocessor.joblib                   # Serialized preprocessor object
├── outputs/
│   ├── ChurnZero_Antigravity_Predictions.csv # Final predictions output
│   ├── business_cost_optimization.png        # Cost-curve plot
│   ├── confusion_matrix_optimized.png        # Cost-optimized confusion matrix
│   └── model_comparison_results.csv          # Comparative classifier metrics
├── presentation/
│   ├── ChurnZero_Antigravity_Presentation.pdf # Executive-ready slide deck
│   └── generate_presentation.py              # Visual presentation builder script
└── README.md                                 # Technical documentation
```

---

## 🛠️ Feature Engineering Breakdown
We engineered 7 business-informed derived features from the 97 raw columns to boost model predictive power:
1. **Product Diversity Score**: A sum of active banking product flags (Savings, Current, FD, demat, loan types). Higher diversity anchors the customer and represents relationship stickiness.
2. **Complaint Severity Index**: A weighted distress index:
   $$\text{Severity} = \text{Unresolved Complaints} \times 3.0 + \text{Escalations} \times 2.0 + \text{Total Complaints} \times 1.0 + (6 - \text{Satisfaction Score}) \times 1.5$$
3. **Balance Decline Trend**: Captures absolute asset drain:
   $$\text{Decline Trend} = \text{Balance Decline \%} \times (\text{Average Monthly Balance} - \text{Current Balance})$$
4. **Digital Inactivity Score**: Represents silent disengagement using `last_login_days` and `account_inactive_days`.
5. **Transaction Activity Decline**: Identifies drops in monthly transaction frequency.
6. **Outstanding Loan to Income Ratio**: Measures debt exposure against annual income.
7. **App Rating Missing Indicator**: Captures non-adoption of digital tools, which is strongly correlated with higher churn rates.

---

## 🚀 How to Run the Pipeline

### 1. Model Training & Cross-Validation
Trains Logistic Regression, Random Forest, LightGBM, XGBoost, and CatBoost using a Stratified 5-Fold CV. Computes OOF metrics, plots the cost curve, and serializes the best model (CatBoost) to `/models`:
```bash
python src/train.py
```

### 2. Generate Test Predictions
Preprocesses the test set, applies the optimal tuned threshold, runs validation assertions (exactly 2026 rows, proper columns, no nulls), and writes the required CSV file:
```bash
python src/predict.py
```

### 3. Generate Executive Slides
Generates a highly professional 12-page PDF presentation using data-driven vector graphics and Matplotlib:
```bash
python presentation/generate_presentation.py
```

### 4. Run the Streamlit Dashboard
Launches a beautiful, dark-slate themed prevention hub for bank executives. Includes KPI cards, interactive EDA, model metrics, and an **Individual Customer Risk Profiler** that generates tailored retention recommendations:
```bash
streamlit run src/dashboard.py
```

---

## 📊 Summary of Model Results
* **Primary Metric (PR-AUC)**: CatBoost Classifier achieved the highest PR-AUC (~0.88+) outperforming XGBoost and baseline models.
* **Tuned Decision Threshold**: Set to `0.07` to optimize business costs.
* **Recall at Tuned Threshold**: Spiked from 35% (at standard 0.5 threshold) to **86.4%**.
* **Financial Impact**: Net savings of **₹4.25 Crores** (an **81.7% loss reduction** compared to doing nothing).
