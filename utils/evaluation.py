"""
Evaluation utilities for retail ML models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_forecast(y_true, y_pred, y_lower=None, y_upper=None):
    """Evaluate forecast performance"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    }
    
    # Coverage (if intervals provided)
    if y_lower is not None and y_upper is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        metrics['Coverage'] = coverage
    
    return metrics


def evaluate_classification(y_true, y_pred, y_proba=None):
    """Evaluate classification performance"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        except:
            pass
    
    return metrics


def plot_forecast_results(forecasts_df, actuals_df=None, title="Forecast Results"):
    """Plot forecast results with confidence intervals"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'date' in forecasts_df.columns:
        x = pd.to_datetime(forecasts_df['date'])
    else:
        x = range(len(forecasts_df))
    
    ax.plot(x, forecasts_df['forecast'], label='Forecast', linewidth=2)
    
    if 'lower_bound' in forecasts_df.columns and 'upper_bound' in forecasts_df.columns:
        ax.fill_between(x, forecasts_df['lower_bound'], forecasts_df['upper_bound'],
                       alpha=0.3, label='Confidence Interval')
    
    if actuals_df is not None:
        if 'date' in actuals_df.columns:
            x_actual = pd.to_datetime(actuals_df['date'])
        else:
            x_actual = range(len(actuals_df))
        ax.plot(x_actual, actuals_df['actual'], label='Actual', 
               marker='o', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dict, top_n=10, title="Feature Importance"):
    """Plot feature importance"""
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), importances)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

