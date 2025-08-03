import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve
from sklearn.utils import resample
from sklearn.metrics import brier_score_loss
import random
from scipy.stats import bootstrap

np.random.seed(1000)
random.seed(1000)

def Find_Optimal_Cutoff(TPR, FPR, threshold): # find cutoff
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]

    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
 
# define the function to compute all values
def calculate_metrics(y_true, y_prob,y_pred=None):
    
    if y_pred.all() == None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]

        y_pred = (y_prob >= best_threshold)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    pos_pred = precision_score(y_true, y_pred)
    neg_pred = tn / (tn + fn)

    f1 = f1_score(y_true, y_pred)

    score = brier_score_loss(y_true, y_prob)

    # aggregate all values
    metrics = {
        'acc': acc,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'pos_pred': pos_pred,
        'neg_pred': neg_pred,
        'f1': f1,
        'brier_score':score
    }
    
    return metrics

# compute 95% confidence range
def calculate_ci(metric, metric_distribution, alpha=0.05):
    p = (1.0 - alpha) / 2.0
    lower = max(0.0, np.percentile(metric_distribution, 100 * p))
    upper = min(1.0, np.percentile(metric_distribution, 100 * (1 - p)))
    return lower, upper

def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def bootstrap_metrics_bca(y_true, y_prob, n_bootstrap=1000,yoden_index = None): # use bootstrap

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if yoden_index == None:
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]
    else:
        best_threshold = yoden_index
    
    y_pred = (y_prob >= best_threshold)

    metrics = calculate_metrics(y_true, y_prob,y_pred)
    bootstrap_metrics = {key: [] for key in metrics.keys()}

    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True)

        while (1 not in y_true[indices]) or (0 not in y_true[indices]):
            indices = resample(np.arange(len(y_true)), replace=True)
            
        y_true_bootstrap = y_true[indices]
        y_prob_bootstrap = y_prob[indices]
        y_pred_bootstrap = y_prob_bootstrap >= best_threshold

        bootstrap_results = calculate_metrics(y_true_bootstrap, y_prob_bootstrap, y_pred_bootstrap)
        
        for key, value in bootstrap_results.items():
            bootstrap_metrics[key].append(value)

    # compute 95% confidence range
    metrics_ci = {}
    for key, values in bootstrap_metrics.items():
        metrics_ci[key] = calculate_bca_ci((values,), np.mean)
            
    return metrics, metrics_ci, best_threshold

def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=3000):
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    bootstrap_metrics = {key: [] for key in metrics.keys()}

    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True) 
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        y_prob_bootstrap = y_prob[indices]
        
        bootstrap_results = calculate_metrics(y_true_bootstrap, y_pred_bootstrap, y_prob_bootstrap)
        
        for key, value in bootstrap_results.items():
            bootstrap_metrics[key].append(value)

    metrics_ci = {key: calculate_ci(metrics[key], bootstrap_metrics[key]) for key in metrics.keys()}

    print("Metrics with 95% Confidence Intervals:")
    for key, value in metrics.items():
        if key in ['tp','tn','fp','fn']:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.3f} (95% CI: {metrics_ci[key][0]:.3f} - {metrics_ci[key][1]:.3f})",end=' ')
            print(round(value,3) <= round(metrics_ci[key][1],3) and round(value,3) >= round(metrics_ci[key][0],3))
        
    return metrics, metrics_ci


