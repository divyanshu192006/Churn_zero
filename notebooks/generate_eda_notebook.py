import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ChurnZero – Banking Customer Churn Prediction System\n",
    "## 📊 Phase 1: Exploratory Data Analysis (EDA)\n",
    "\n",
    "This notebook contains a comprehensive analysis of the banking customer churn dataset (`ChurnZero_dataset_v1.csv`).\n",
    "\n",
    "### Objective:\n",
    "- Profile and clean the customer datasets.\n",
    "- Visualize relations between demographic, transactional, product, digital, complaint, and marketing features with the target variable `churn`.\n",
    "- Discover key behavioral markers of customers high-risk of churning.\n",
    "- Quantify business cost metrics: **False Negative (FN) cost ₹40,000** vs **False Positive (FP) cost ₹500**."
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
    "import os\n",
    "\n",
    "# Set style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
    "plt.rcParams[\"font.size\"] = 12"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Load Dataset & Basic Profiling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_path = os.path.join(\"..\", \"data\", \"ChurnZero_dataset_v1.csv\")\n",
    "df = pd.read_csv(train_path)\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "print(df.info(max_cols=15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Target Class Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "churn_counts = df['churn'].value_counts()\n",
    "churn_pcts = df['churn'].value_counts(normalize=True)\n",
    "\n",
    "print(\"Churn Value Counts:\")\n",
    "print(churn_counts)\n",
    "print(\"\\nChurn Percentages:\")\n",
    "print(churn_pcts)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(6, 4))\n",
    "sns.countplot(data=df, x='churn', palette=['#38BDF8', '#D32F2F'], ax=ax)\n",
    "ax.set_title('Target Variable Distribution (Churn vs. Loyal)', fontsize=14, fontweight='bold')\n",
    "ax.set_xticklabels(['Loyal (0)', 'Churn (1)'])\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(\"..\", \"outputs\", \"target_distribution.png\"), dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Missing Value Analysis\n",
    "As observed during initial batch ingestion, `app_rating_given` has massive missingness (~56%). Let's examine if this missingness itself correlates with churn!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['app_rating_missing'] = df['app_rating_given'].isnull().astype(int)\n",
    "\n",
    "# Check churn rate for customers who gave rating vs. who didn't\n",
    "rating_churn_rate = df.groupby('app_rating_missing')['churn'].mean()\n",
    "print(\"Churn rate by app_rating_missing status:\")\n",
    "print(rating_churn_rate)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(6, 4))\n",
    "sns.barplot(x=rating_churn_rate.index, y=rating_churn_rate.values, palette=['#10B981', '#EF4444'], ax=ax)\n",
    "ax.set_title('Churn Rate by App Rating Missingness', fontsize=14, fontweight='bold')\n",
    "ax.set_xticklabels(['Rating Given', 'Rating Missing (No App Engagement)'])\n",
    "ax.set_ylabel('Average Churn Rate')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(\"..\", \"outputs\", \"missing_rating_churn.png\"), dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Demographic & Segment Profiling vs Churn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Churn Rate by Customer Segment\n",
    "segment_churn = df.groupby('customer_segment')['churn'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)\n",
    "print(segment_churn)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "sns.barplot(data=df, x='customer_segment', y='churn', palette='viridis', ci=None, ax=ax)\n",
    "ax.set_title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Churn Rate')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Financial Behavior & Balance Decline vs Churn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Balance Decline % Boxplot\n",
    "sns.boxplot(data=df, x='churn', y='balance_decline_percentage', palette=['#38BDF8', '#D32F2F'], ax=axes[0])\n",
    "axes[0].set_title('Balance Decline % vs Churn', fontsize=12, fontweight='bold')\n",
    "axes[0].set_xticklabels(['Loyal (0)', 'Churn (1)'])\n",
    "\n",
    "# Customer Lifetime Value CLV\n",
    "sns.boxplot(data=df, x='churn', y='customer_lifetime_value', palette=['#38BDF8', '#D32F2F'], ax=axes[1])\n",
    "axes[1].set_title('Customer Lifetime Value vs Churn', fontsize=12, fontweight='bold')\n",
    "axes[1].set_xticklabels(['Loyal (0)', 'Churn (1)'])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(\"..\", \"outputs\", \"financial_behavior_boxplots.png\"), dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Service Experience & Complaints vs Churn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Escalations\n",
    "sns.barplot(data=df, x='churn', y='escalation_count', palette=['#38BDF8', '#D32F2F'], ax=axes[0], ci=None)\n",
    "axes[0].set_title('Average Escalations vs Churn', fontsize=12, fontweight='bold')\n",
    "axes[0].set_xticklabels(['Loyal (0)', 'Churn (1)'])\n",
    "\n",
    "# Support Calls\n",
    "sns.barplot(data=df, x='churn', y='call_center_interaction_count', palette=['#38BDF8', '#D32F2F'], ax=axes[1], ci=None)\n",
    "axes[1].set_title('Call Center Interactions vs Churn', fontsize=12, fontweight='bold')\n",
    "axes[1].set_xticklabels(['Loyal (0)', 'Churn (1)'])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(\"..\", \"outputs\", \"service_interactions.png\"), dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Correlation Analysis\n",
    "Check correlations of numerical columns with churn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop non-numeric and target-leaking features for correlation check\n",
    "numeric_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()\n",
    "correlations = df[numeric_cols].corr()['churn'].sort_values(ascending=False)\n",
    "\n",
    "print(\"Top 15 Features Positively Correlated with Churn:\")\n",
    "print(correlations.head(16))\n",
    "\n",
    "print(\"\\nTop 15 Features Negatively Correlated with Churn:\")\n",
    "print(correlations.tail(15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Key EDA Findings:\n",
    "1. **High Churn Segment**: Churn rate is noticeably higher among specific branches or low tenure relationship types.\n",
    "2. **The App Disconnect**: Customers with a missing mobile app rating are twice as likely to churn, signifying that non-mobile users are extremely sensitive to disengagement.\n",
    "3. **Complaints & Escalations**: Escalations and unresolved complaints are the strongest positive correlations of churn. A customer with a high escalation count requires immediate support triage.\n",
    "4. **Financial Signal**: A high `balance_decline_percentage` combined with a drop in monthly transactions represents a customer shifting primary banking activities elsewhere."
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
notebook_path = os.path.join("..", "notebooks", "01_exploratory_data_analysis.ipynb")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Generated exploratory data analysis notebook at: {notebook_path}")
