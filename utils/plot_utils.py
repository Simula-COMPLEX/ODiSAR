import matplotlib.pyplot as plt
import numpy as np

def plot_forecast_vs_ground_truth(results_df, feature_names, sequence_index=0, forecast_horizon=60, save_path=None):
    """
    Plots the ground truth vs forecasted features for a selected forecast window.

    Args:
        results_df (pd.DataFrame): DataFrame with columns True_<feature> and Forecasted_<feature>.
        feature_names (list): List of feature names (without "True_" or "Forecasted_" prefix).
        sequence_index (int): Index of the forecast window to plot.
        forecast_horizon (int): Number of time steps per window.
        save_path (str or None): If provided, saves the plot to this path.
    """
    start_idx = sequence_index * forecast_horizon
    end_idx = start_idx + forecast_horizon

    plt.figure(figsize=(15, 6))
    for feat in feature_names:
        true_vals = results_df[f"True_{feat}"][start_idx:end_idx]
        pred_vals = results_df[f"Forecasted_{feat}"][start_idx:end_idx]
        plt.plot(true_vals, label=f"GT: {feat}", linestyle='-')
        plt.plot(pred_vals, label=f"Forecast: {feat}", linestyle='--')

    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.title(f"Forecast vs Ground Truth (Window {sequence_index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_reconstruction_error_distribution(errors, labels, threshold):
    """
    Plots the histogram of reconstruction errors for IND and OOD labels.
    """
    plt.figure()
    plt.hist(errors[labels == 0], bins=50, alpha=0.6, label='IND (Recon)')
    plt.hist(errors[labels == 1], bins=50, alpha=0.6, label='OOD (Recon)')
    plt.axvline(threshold, color='r', linestyle='--', label='Recon Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_forecast_variance_distribution(variances, labels, threshold):
    """
    Plots the histogram of forecast variance for IND and OOD labels.
    """
    plt.figure()
    plt.hist(variances[labels == 0], bins=50, alpha=0.6, label='IND (Variance)')
    plt.hist(variances[labels == 1], bins=50, alpha=0.6, label='OOD (Variance)')
    plt.axvline(threshold, color='r', linestyle='--', label='Variance Threshold')
    plt.title('Forecast Variance Distribution')
    plt.xlabel("Forecast Variance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_quadrant_scatter(recon_errors, forecast_variances, recon_thresh, var_thresh, colors, categories):
    """
    Plots a quadrant scatter plot of reconstruction error vs forecast variance, color-coded by category.

    Args:
        recon_errors (list): Reconstruction error values per sequence.
        forecast_variances (list): Forecast variance (uncertainty) values per sequence.
        recon_thresh (float): Threshold for reconstruction error.
        var_thresh (float): Threshold for variance.
        colors (list): List of color codes per sequence (e.g., "red").
        categories (list): List of category labels per sequence (e.g., "OOD & Confident").
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(recon_errors, forecast_variances, c=colors, alpha=0.7, edgecolors='k')
    plt.axvline(recon_thresh, color='red', linestyle='--', label=f'Recon Threshold: {recon_thresh:.4f}')
    plt.axhline(var_thresh, color='blue', linestyle='--', label=f'Variance Threshold: {var_thresh:.4f}')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Forecast Variance (Uncertainty)")
    plt.title("Confidence-Aware OOD Classification per Forecast Window")

    # Custom legend using (unique) category-color pairs
    legend_entries = {}
    for cat, col in zip(categories, colors):
        if cat not in legend_entries:
            legend_entries[cat] = col
    for label, color in legend_entries.items():
        plt.scatter([], [], c=color, edgecolors='k', label=label)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_forecast_vs_ground_truth_horizon(results_df, feature_names, horizon_steps=300, save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))
    for feat in feature_names:
        plt.plot(results_df[f"True_{feat}"][:horizon_steps], label=f"GT: {feat}", linestyle='-')
        plt.plot(results_df[f"Forecasted_{feat}"][:horizon_steps], label=f"Forecast: {feat}", linestyle='--')

    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.title(f"Forecast vs Ground Truth (First {horizon_steps} steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
