# src/visualization.py
"""Module for visualization of model performance."""
# Visualization: Confusion Matrix and Precision-Recall Curve
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
from sklearn.metrics import auc


def plot_confusion_matrix(cm, title="Confusion Matrix", figsize=(8, 6)):
    """Create a heatmap for the confusion matrix."""
    _, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_pr_curve(y_true, y_prob, label=None, ax=None, figsize=(8, 6)):
    """Plot Precision-Recall curve."""
    if ax is None:
        # Create new figure if no axes provided
        _, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision,
            label=f'{label} (AUC = {pr_auc:.3f})' if label else f'AUC = {pr_auc:.3f}')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    if standalone:
        ax.set_title("Precision-Recall Curve")

    ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    if standalone:
        plt.show()

    return ax, pr_auc


def plot_threshold_analysis(threshold_results, metrics=None, figsize=(12, 8)):
    """
    Plot model performance across different thresholds.

    Parameters
    ----------
    threshold_results : DataFrame
        Output from evaluate_model_at_thresholds()
    metrics : list
        Metrics to plot
    """
    _, axes = plt.subplots(2, 2, figsize=figsize)

    # Avoid mutable default argument
    if metrics is None:
        metrics = ['f1', 'precision', 'recall']

    # Plot F1, Precision, Recall
    for i, metric in enumerate(metrics[:3]):
        ax = axes[i // 2, i % 2]
        ax.plot(threshold_results['threshold'], threshold_results[metric],
                marker='o', linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs Threshold')
        ax.grid(True, alpha=0.3)

        # Mark optimal point
        optimal_idx = threshold_results[metric].idxmax()
        optimal_thresh = threshold_results.loc[optimal_idx, 'threshold']
        optimal_value = threshold_results.loc[optimal_idx, metric]
        ax.plot(optimal_thresh, optimal_value, 'r*', markersize=15,
                label=f'Optimal: {optimal_thresh:.2f}')
        ax.legend()

    # Plot confusion matrix heatmap at optimal threshold
    optimal_idx = threshold_results['f1'].idxmax()
    optimal_row = threshold_results.iloc[optimal_idx]

    if all(col in optimal_row.index for col in ['true_positives', 'false_positives',
                                                'false_negatives', 'true_negatives']):
        cm_data = np.array([
            [optimal_row['true_negatives'], optimal_row['false_positives']],
            [optimal_row['false_negatives'], optimal_row['true_positives']]
        ], dtype=float)

        # Ensure integer display for annotations (round if needed)
        cm_int = np.rint(cm_data).astype(int)

        ax = axes[1, 1]
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], ax=ax)
        ax.set_title(
            f'Confusion Matrix at Threshold {optimal_row["threshold"]:.2f}')

    plt.tight_layout()
    plt.show()


def plot_metric_tradeoff(threshold_results, figsize=(10, 6)):
    """
    Plot precision-recall tradeoff across thresholds.
    """
    _, ax = plt.subplots(figsize=figsize)

    ax.plot(threshold_results['recall'], threshold_results['precision'],
            marker='o', linewidth=2)

    # Annotate some key thresholds
    key_thresholds = [0.3, 0.5, 0.65, 0.7]
    for thresh in key_thresholds:
        closest_idx = (threshold_results['threshold'] - thresh).abs().idxmin()
        row = threshold_results.loc[closest_idx]
        ax.annotate(f'Thresh={row["threshold"]:.2f}\nF1={row["f1"]:.3f}',
                    xy=(row['recall'], row['precision']),
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff Across Thresholds')
    ax.grid(True, alpha=0.3)
    plt.show()


def save_figure(path, fig=None, **savefig_kwargs):
    """
    Save a matplotlib figure ensuring the parent directory exists.

    Parameters
    ----------
    path : str or Path
        File path to save the figure to.
    fig : matplotlib.figure.Figure or None
        Figure instance to save. If None, uses `plt.gcf()`.
    **savefig_kwargs :
        Passed to `Figure.savefig` (dpi, bbox_inches, etc.).
    """

    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    fig.savefig(str(p), **savefig_kwargs)
    return str(p)


def save_figures(path, fig=None, **savefig_kwargs):
    """Backward-compatible alias for `save_figure` used by notebooks.

    Kept for compatibility with existing notebook code that imports
    `save_figures` from `src.visualization`.
    """
    return save_figure(path, fig=fig, **savefig_kwargs)
