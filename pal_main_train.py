import os
import torch
import pickle
import pandas as pd
import numpy as np

from config.pal_config_train import (
    BASE_DIR, EXPERIMENT_NAME,
    INPUT_FEATURES, OUTPUT_FEATURES,
    DATA_PATHS, SEQUENCE_SETTINGS, MODEL_PARAMS,
    TRAINING_PARAMS, TRAINING_FLAGS,
    OUTPUT_PATHS
)

from data.dataset import load_directory_data, create_sequences
from models.transformer_model import MergedForecastReconModel
from utils.train_utils import (
    setup_logger,
    train_phase1,
    train_phase2,
    compute_forecast_recon_errors,
    compute_mc_forecast_variance
)
from utils.plot_utils import plot_forecast_vs_ground_truth

# === Device & Logger ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file_path = os.path.join(OUTPUT_PATHS['log_dir'], 'training_log.txt')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logger = setup_logger(log_file_path)

# === Load and Normalize Data ===
logger.info("Loading and normalizing training and validation data...")
combined_features = list(dict.fromkeys(INPUT_FEATURES + OUTPUT_FEATURES))  # remove duplicates
train_raw_df = load_directory_data(DATA_PATHS['train_data_dir'], combined_features)
val_raw_df   = load_directory_data(DATA_PATHS['validation_data_dir'], combined_features)

mean = train_raw_df.mean()
std  = train_raw_df.std()
train_df = ((train_raw_df - mean) / std).fillna(0)
val_df   = ((val_raw_df - mean) / std).fillna(0)

# Save normalization parameters
os.makedirs(os.path.dirname(OUTPUT_PATHS['normalization_params']), exist_ok=True)
with open(OUTPUT_PATHS['normalization_params'], 'wb') as f:
    pickle.dump({'mean': mean, 'std': std}, f)

# === Prepare Input and Output Subsets ===
train_inputs = train_df[INPUT_FEATURES]
train_outputs = train_df[OUTPUT_FEATURES]
val_inputs = val_df[INPUT_FEATURES]
val_outputs = val_df[OUTPUT_FEATURES]
print("train_inputs columns:", train_inputs.columns.tolist())
print("train_outputs columns:", train_outputs.columns.tolist())

# === Sequence Settings ===
input_window = SEQUENCE_SETTINGS['input_window_size']
forecast_horizon = SEQUENCE_SETTINGS['forecast_horizon_size']

# === Model ===
input_dim = len(INPUT_FEATURES)
output_dim = len(OUTPUT_FEATURES)
model = MergedForecastReconModel(
    input_dim=len(INPUT_FEATURES),
    output_dim=len(OUTPUT_FEATURES),
    d_model=MODEL_PARAMS['d_model'],
    num_heads=MODEL_PARAMS['num_heads'],
    ff_dim=MODEL_PARAMS['ff_dim'],
    input_window_size=input_window,
    forecast_horizon_size=forecast_horizon,
    dropout=MODEL_PARAMS['dropout_rate']
).to(device)


# === Phase 1 Training ===
if TRAINING_FLAGS['enable_phase1_training']:
    if os.path.exists(OUTPUT_PATHS['phase1_model']):
        logger.info("Loading existing Phase 1 model.")
        model.load_state_dict(torch.load(OUTPUT_PATHS['phase1_model'], map_location=device))
    else:
        X_train, y_train = create_sequences(train_inputs, train_outputs, input_window, forecast_horizon, overlap=True)
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        train_phase1(
            model, (X_train, y_train), input_window, forecast_horizon,
            TRAINING_PARAMS['batch_size'],
            TRAINING_PARAMS['learning_rate'],
            TRAINING_PARAMS['epochs_phase1'],
            device, logger,
            early_stopping=TRAINING_PARAMS.get('early_stopping', None)
        )
        torch.save(model.state_dict(), OUTPUT_PATHS['phase1_model'])

# === Phase 2 Fine-tuning ===
if TRAINING_FLAGS['enable_phase2_training']:
    if os.path.exists(OUTPUT_PATHS['phase2_model']):
        logger.info("Loading existing Phase 2 model.")
        model.load_state_dict(torch.load(OUTPUT_PATHS['phase2_model'], map_location=device))
    else:
        model.load_state_dict(torch.load(OUTPUT_PATHS['phase1_model'], map_location=device))
        X_ft, y_ft = create_sequences(train_inputs, train_outputs, input_window, forecast_horizon, overlap=False)
        train_phase2(
            model, (X_ft, y_ft),
            TRAINING_PARAMS['batch_size'],
            TRAINING_PARAMS['learning_rate'],
            TRAINING_PARAMS['epochs_phase2'],
            device, logger,
            early_stopping=TRAINING_PARAMS.get('early_stopping', None)
        )
        torch.save(model.state_dict(), OUTPUT_PATHS['phase2_model'])

# === Thresholds on Validation ===
logger.info("Computing thresholds...")
# X_val, y_val = create_sequences(val_inputs, val_outputs, input_window, forecast_horizon, overlap=False)
X_val, y_val = create_sequences(val_inputs, val_outputs, input_window, forecast_horizon, overlap=False)

val_recon_errors = compute_forecast_recon_errors(model, (X_val, y_val), TRAINING_PARAMS['batch_size'], device)
recon_threshold = val_recon_errors.mean() + 3 * val_recon_errors.std()
val_variances = compute_mc_forecast_variance(model, (X_val, y_val), TRAINING_PARAMS['batch_size'], device, n_passes=20)
variance_threshold = val_variances.mean() + 3 * val_variances.std()

with open(OUTPUT_PATHS['threshold_params'], 'wb') as f:
    pickle.dump({
        'recon_threshold': recon_threshold,
        'variance_threshold': variance_threshold,
        'recon_mean': val_recon_errors.mean(),
        'recon_std': val_recon_errors.std(),
        'var_mean': val_variances.mean(),
        'var_std': val_variances.std()
    }, f)

# === Evaluation on Training Set ===
logger.info("Evaluating on training set...")
step_size = input_window + forecast_horizon
all_forecasts, all_truth = [], []
model.eval()

with torch.no_grad():
    for i in range(0, len(train_inputs) - step_size + 1, step_size):
        x_seq = train_inputs.iloc[i:i+input_window].values
        y_seq = train_outputs.iloc[i+input_window:i+input_window+forecast_horizon].values

        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=device).unsqueeze(0)

        forecast_out, _ = model(x_tensor, y_tensor)
        all_forecasts.append(forecast_out.squeeze(0).cpu().numpy())
        all_truth.append(y_seq)

forecast_array = np.concatenate(all_forecasts, axis=0)
truth_array = np.concatenate(all_truth, axis=0)

results_df = pd.DataFrame()
for i, feature in enumerate(OUTPUT_FEATURES):
    results_df[f"True_{feature}"] = truth_array[:, i]
    results_df[f"Forecasted_{feature}"] = forecast_array[:, i]
results_df.to_csv(OUTPUT_PATHS['forecast_results_csv'], index=False)

# === Error Reporting ===
mse_vals, rmse_vals = {}, {}
for feature in OUTPUT_FEATURES:
    diff = results_df[f"True_{feature}"] - results_df[f"Forecasted_{feature}"]
    mse_vals[feature] = np.mean(diff ** 2)
    rmse_vals[feature] = np.sqrt(mse_vals[feature])

error_df = pd.DataFrame({
    "Feature": OUTPUT_FEATURES,
    "MSE": [mse_vals[f] for f in OUTPUT_FEATURES],
    "RMSE": [rmse_vals[f] for f in OUTPUT_FEATURES]
})
error_df.to_csv(OUTPUT_PATHS['forecast_errors_csv'], index=False)

# === Plot Forecast vs GT ===
plot_forecast_vs_ground_truth(
    results_df=results_df,
    feature_names=OUTPUT_FEATURES,
    sequence_index=0,
    forecast_horizon=forecast_horizon,
    save_path=OUTPUT_PATHS['forecast_plot_path']
)

logger.info("Training and evaluation completed successfully.")
