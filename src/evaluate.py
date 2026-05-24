import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_business_cost(y_true, y_pred_prob, threshold, fn_cost=40000, fp_cost=500):
    """
    Calculates the total business cost for a given probability threshold.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TN, FP, FN, TP
    # In binary classification, confusion_matrix returns:
    # [[TN, FP],
    #  [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases (e.g. all 0s or all 1s predicted)
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)
    
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    return total_cost, tn, fp, fn, tp

def optimize_threshold(y_true, y_pred_prob, fn_cost=40000, fp_cost=500):
    """
    Searches for the decision threshold that minimizes the total business cost.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    details = []
    
    # Baseline cost: assume we don't predict churn for anyone (no proactive retention)
    # So all actual churners are lost (False Negatives)
    total_churners = np.sum(y_true)
    baseline_cost = total_churners * fn_cost
    
    for t in thresholds:
        cost, tn, fp, fn, tp = calculate_business_cost(y_true, y_pred_prob, t, fn_cost, fp_cost)
        costs.append(cost)
        details.append({
            'threshold': t,
            'cost': cost,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0),
            'recall': recall_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0),
            'f1': f1_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0),
            'savings': baseline_cost - cost
        })
        
    best_idx = np.argmin(costs)
    best_detail = details[best_idx]
    
    return best_detail, pd.DataFrame(details)

def plot_cost_vs_threshold(df_details, save_path=None):
    """
    Plots the business cost and financial savings vs. the decision threshold.
    """
    plt.figure(figsize=(10, 6))
    
    # Baseline cost (cost at threshold = 1.0, meaning no one is predicted to churn)
    # The cost at threshold close to 1.0 is the cost when we do nothing
    baseline_cost = df_details.loc[df_details['threshold'].idxmax(), 'cost'] + df_details.loc[df_details['threshold'].idxmax(), 'tp'] * 40000 # Approximation
    # Exact baseline cost is when all positive instances are FNs:
    # We can fetch it from our optimization details:
    # At t=0.99, tp is almost 0, fn is almost all positive instances.
    # Let's get the exact baseline cost
    actual_churners = df_details.iloc[-1]['fn'] + df_details.iloc[-1]['tp']
    exact_baseline = actual_churners * 40000
    
    plt.plot(df_details['threshold'], df_details['cost'], color='#D32F2F', linewidth=2.5, label='Total Business Cost (₹)')
    plt.axhline(y=exact_baseline, color='#000000', linestyle='--', alpha=0.7, label=f'Do Nothing Baseline (₹{exact_baseline:,.0f})')
    
    # Highlight optimal threshold
    best_detail = df_details.loc[df_details['cost'].idxmin()]
    opt_t = best_detail['threshold']
    opt_cost = best_detail['cost']
    
    plt.axvline(x=opt_t, color='#2E7D32', linestyle=':', linewidth=2, label=f'Optimal Threshold: {opt_t:.2f}')
    plt.scatter([opt_t], [opt_cost], color='#2E7D32', s=100, zorder=5)
    
    plt.title('Business Cost Optimization Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Total Cost in ₹ (Lower is Better)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    """
    Plots a beautiful customized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    
    # Labels with financial implications
    group_names = ['True Negative\n(Loyal Customer)', 'False Positive\n(Retention Cost: ₹500)', 'False Negative\n(Lost Churner: ₹40,000)', 'True Positive\n(Saved Customer)']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n\nCount: {v2}\nRatio: {v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=False,
                xticklabels=['Predict Loyal (0)', 'Predict Churn (1)'],
                yticklabels=['Actual Loyal (0)', 'Actual Churn (1)'],
                annot_kws={"size": 11, "fontweight": "bold"})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
