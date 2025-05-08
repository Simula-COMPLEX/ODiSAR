import os
import pickle
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from config.config_inference import (
    TEST_DATA_DIR,
    SCALER_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    JSON_OUTPUT_PATH,
    FORECAST_HORIZON,
    SEQ_LEN,
    BATCH_SIZE,
    INPUT_FEATURES,
    OUTPUT_FEATURES
)
from config.config_train import MODEL_PARAMS

from models.transformer_model import MergedForecastReconModel
from utils.inference_utils import (
    TimeSeriesDataset,
    load_directory_data,
    create_sequences,
    compute_forecast_recon_errors,
    compute_forecast_variance,
    evaluate_ood_detection,
    export_ood_diagnostics_json
)
from utils.plot_utils import (
    plot_reconstruction_error_distribution,
    plot_forecast_variance_distribution,
    plot_quadrant_scatter
)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and Normalize Test Data ===
print(f"Loading test data from: {TEST_DATA_DIR}")
# test_raw_df, ood_labels = load_directory_data(TEST_DATA_DIR, INPUT_FEATURES + OUTPUT_FEATURES)
combined_features = list(dict.fromkeys(INPUT_FEATURES + OUTPUT_FEATURES))  # remove duplicates
test_raw_df, ood_labels = load_directory_data(TEST_DATA_DIR, combined_features)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

normalized_df = ((test_raw_df - scaler['mean']) / scaler['std']).fillna(0)
test_inputs = normalized_df[INPUT_FEATURES]
test_outputs = normalized_df[OUTPUT_FEATURES]

# === Prepare Sequences ===
X_test, y_test = create_sequences(test_inputs, test_outputs, SEQ_LEN, FORECAST_HORIZON)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# === Load Trained Model ===
input_dim = len(INPUT_FEATURES)
output_dim = len(OUTPUT_FEATURES)

model = MergedForecastReconModel(
    input_dim=input_dim,
    output_dim=output_dim,
    d_model=MODEL_PARAMS['d_model'],
    num_heads=MODEL_PARAMS['num_heads'],
    ff_dim=MODEL_PARAMS['ff_dim'],
    input_window_size=SEQ_LEN,
    forecast_horizon_size=FORECAST_HORIZON,
    dropout=MODEL_PARAMS['dropout_rate']
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Load Thresholds ===
with open(THRESHOLD_PATH, 'rb') as f:
    thresholds = pickle.load(f)

recon_thresh = thresholds['recon_threshold']
var_thresh = thresholds['variance_threshold']

# === Inference: Errors and Variance ===
print("Computing forecast reconstruction errors...")
recon_errors, forecast_output, feature_recon_errors = compute_forecast_recon_errors(
    model, test_loader, device, OUTPUT_FEATURES
)

print("Computing forecast variance with MC dropout...")
forecast_variances = compute_forecast_variance(
    model, test_loader, device, n_passes=20
)

# === OOD Evaluation ===
evaluate_ood_detection(
    recon_errors=recon_errors,
    forecast_variances=forecast_variances,
    recon_thresh=recon_thresh,
    var_thresh=var_thresh,
    forecast_horizon=FORECAST_HORIZON,
    ood_labels=ood_labels,
    feature_names=OUTPUT_FEATURES
)

# === JSON Output ===
colors, categories = export_ood_diagnostics_json(
    recon_errors=recon_errors,
    forecast_variances=forecast_variances,
    recon_thresh=recon_thresh,
    var_thresh=var_thresh,
    forecast_horizon=FORECAST_HORIZON,
    feature_names=OUTPUT_FEATURES,
    feature_recon_errors=feature_recon_errors,
    output_path=JSON_OUTPUT_PATH
)

# === Optional Plots ===
if ood_labels is not None:
    repeated_recon_errors = np.repeat(recon_errors, FORECAST_HORIZON)
    repeated_variances = np.repeat(forecast_variances, FORECAST_HORIZON)
    gt_labels = []
    for i in range(0, len(ood_labels) - FORECAST_HORIZON, FORECAST_HORIZON * 2):
        gt_labels.extend(ood_labels[i + FORECAST_HORIZON : i + FORECAST_HORIZON * 2])
    gt_labels = np.array(gt_labels[:len(repeated_recon_errors)])

    plot_reconstruction_error_distribution(repeated_recon_errors, gt_labels, recon_thresh)
    plot_forecast_variance_distribution(repeated_variances, gt_labels, var_thresh)

    plot_quadrant_scatter(
        recon_errors,
        forecast_variances,
        recon_thresh,
        var_thresh,
        colors,
        categories
    )

# === Save Forecast Results ===
forecast_output = forecast_output.reshape(-1, FORECAST_HORIZON, len(OUTPUT_FEATURES))
results_df = pd.DataFrame()

for i, feat in enumerate(OUTPUT_FEATURES):
    results_df[f"True_{feat}"] = y_test[:forecast_output.shape[0], :, i].reshape(-1)
    results_df[f"Forecasted_{feat}"] = forecast_output[:, :, i].reshape(-1)


print("forecast_output shape before reshape:", forecast_output.shape)

results_df.to_csv(JSON_OUTPUT_PATH.replace('.json', '_forecast_results.csv'), index=False)

# === Plot First 300 Steps Instead of One Window ===
from utils.plot_utils import plot_forecast_vs_ground_truth_horizon

plot_forecast_vs_ground_truth_horizon(
    results_df=results_df,
    feature_names=OUTPUT_FEATURES,
    horizon_steps=300,
    save_path=JSON_OUTPUT_PATH.replace('.json', '_forecast_plot.png')
)

print("Inference complete.")