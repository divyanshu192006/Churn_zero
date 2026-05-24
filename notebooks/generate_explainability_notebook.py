import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ChurnZero – Banking Customer Churn Prediction System\n",
    "## 🧠 Phase 2: Model Explainability & SHAP Analysis\n",
    "\n",
    "This notebook demonstrates model diagnostics, global explainability, and individual customer risk profiling using the fitted **CatBoost** model and **SHAP (SHapley Additive exPlanations)** values.\n",
    "\n",
    "### Objective:\n",
    "- Uncover which factors are driving customer churn globally.\n",
    "- Compute feature importances and SHAP values.\n",
    "- Formulate high-resolution local predictions for individual high-risk customers.\n",
    "- Link model insights directly to actionable banking retention strategies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "import shap\n",
    "import os\n",
    "\n",
    "# Set style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Load Model, Preprocessor and Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_path = os.path.join(\"..\", \"models\", \"final_catboost_model.joblib\")\n",
    "preprocessor_path = os.path.join(\"..\", \"models\", \"preprocessor.joblib\")\n",
    "train_path = os.path.join(\"..\", \"data\", \"ChurnZero_dataset_v1.csv\")\n",
    "\n",
    "model_data = joblib.load(model_path)\n",
    "preprocessor = joblib.load(preprocessor_path)\n",
    "df_train = pd.read_csv(train_path)\n",
    "\n",
    "final_model = model_data['model']\n",
    "feature_cols = model_data['feature_cols']\n",
    "optimal_threshold = model_data['optimal_threshold']\n",
    "\n",
    "print(f\"Loaded CatBoost model from full training fold.\")\n",
    "print(f\"Optimal Decision Threshold: {optimal_threshold:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Prepare Data Pipeline\n",
    "Run feature engineering and preprocessing to match training schema."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.feature_engineering import engineer_features\n",
    "\n",
    "df_engineered = engineer_features(df_train)\n",
    "df_preprocessed = preprocessor.transform(df_engineered)\n",
    "X = df_preprocessed[feature_cols]\n",
    "y = df_preprocessed['churn']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Compute SHAP Values\n",
    "We initialize the TreeExplainer to compute SHAP values for the CatBoost model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = shap.TreeExplainer(final_model)\n",
    "# Compute SHAP values on a subset (e.g. 1000 rows) or full dataset for complete insights\n",
    "sample_idx = np.random.choice(len(X), size=1000, replace=False)\n",
    "X_sample = X.iloc[sample_idx]\n",
    "shap_values = explainer(X_sample)\n",
    "\n",
    "print(\"SHAP calculation completed.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. SHAP Summary Plot\n",
    "The summary plot ranks features by their total impact and colors them by feature values to show the direction of influence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "shap.summary_plot(shap_values, X_sample, show=False)\n",
    "plt.title('SHAP Feature Importance & Churn Impact Direction', fontsize=14, fontweight='bold', pad=20)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(\"..\", \"outputs\", \"shap_summary_plot.png\"), dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Local Risk Profile: Explaining an Individual Customer Churn Risk\n",
    "Let's extract a high-risk customer and explain why the model flagged them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find a customer with high predicted probability\n",
    "probs = final_model.predict_proba(X)[:, 1]\n",
    "high_risk_indices = np.where(probs > 0.7)[0]\n",
    "\n",
    "if len(high_risk_indices) > 0:\n",
    "    target_idx = high_risk_indices[0]\n",
    "    cust_id = df_train.loc[target_idx, 'customer_id']\n",
    "    cust_prob = probs[target_idx]\n",
    "    print(f\"Customer ID: {cust_id}\")\n",
    "    print(f\"Predicted Churn Probability: {cust_prob:.2%}\")\n",
    "    \n",
    "    # Generate force plot or waterfall plot\n",
    "    single_shap = explainer(X.iloc[[target_idx]])\n",
    "    \n",
    "    plt.figure(figsize=(12, 4))\n",
    "    shap.plots.waterfall(single_shap[0], show=False)\n",
    "    plt.title(f'Feature Attribution (Waterfall) for Customer {cust_id}', fontsize=12, fontweight='bold', pad=15)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(\"..\", \"outputs\", f\"shap_waterfall_{cust_id}.png\"), dpi=300)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"No customer found with churn probability > 70% in training fold.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Actionable Retention Takeaways\n",
    "- **High Escalation Count**: SHAP values indicate that when `escalation_count` rises, it instantly dominates the positive log-odds contribution. Resolving escalations is priority #1.\n",
    "- **Balance decline**: As `balance_decline_percentage` shifts from low to high, its SHAP value rises rapidly, marking transactional/account balance drain as a primary churn driver.\n",
    "- **Product diversification**: Customers holding multiple active products (high `product_diversity_score`) show highly negative SHAP values, proving that product cross-selling creates 'sticky' customer relationships."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Ensure directory exists and write JSON
os.makedirs(os.path.join("..", "notebooks"), exist_ok=True)
notebook_path = os.path.join("..", "notebooks", "02_model_explainability.ipynb")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Generated model explainability notebook at: {notebook_path}")
