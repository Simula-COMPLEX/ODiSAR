import os

# === Base Directory (auto-detected) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_NAME = 'dt-ood-repo'

# # === Feature Selection ===
# FEATURE_NAMES = [
#     'Rudder Angle',
#     'Surge Speed',
#     'Sway Speed',
#     'Yaw Rate',
#     'Roll Angle',
#     'Roll Rate'
# ]

INPUT_FEATURES = [
    'Rudder Angle',

    'Surge Speed',
    'Sway Speed',
    'Yaw Rate',
    'Roll Angle',
    'Roll Rate'
]

OUTPUT_FEATURES = [
    'Surge Speed',
    'Sway Speed',
    'Yaw Rate',
    'Roll Angle',
    'Roll Rate'
]

# === Data Paths ===
DATA_PATHS = {
    'train_data_dir': os.path.join(BASE_DIR, 'NTNU', 'ind', 'processed-v2', 'train'),
    'validation_data_dir': os.path.join(BASE_DIR, 'NTNU', 'ind', 'processed-v2', 'validation')
}

# === Sequence Configuration ===
SEQUENCE_SETTINGS = {
    'input_window_size': 60,         # Past time steps used as input
    'forecast_horizon_size': 60      # Future steps to predict
}

# === Model Hyperparameters ===
MODEL_PARAMS = {
    'd_model': 64,
    'num_heads': 4,
    'ff_dim': 128,
    'dropout_rate': 0.1
}

# === Training Configuration ===
TRAINING_PARAMS = {
    'batch_size': 64,
    'learning_rate': 0.0001,
    'epochs_phase1': 120,     # Forecasting phase
    'epochs_phase2': 100,    # Reconstruction fine-tuning phase

    'early_stopping': {
        'enabled': True,         
        'patience': 10,           # Stop if no improvement for 10 epochs
        'patience_extension': 5  # Allow 5 extra epochs after plateau
    }
}

# === Training Control Flags ===
TRAINING_FLAGS = {
    'enable_phase1_training': True,
    'enable_phase2_training': True
}

# === Output Paths ===
OUTPUT_PATHS = {
    'phase1_model': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'phase1_model.pth'),
    'phase2_model': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'phase2_model.pth'),
    'normalization_params': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'normalization_params.pkl'),
    'threshold_params': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'threshold_params.pkl'),
    'forecast_results_csv': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_results', 'forecast_results.csv'),
    'forecast_errors_csv': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_results', 'forecast_errors.csv'),
    'log_dir': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_logs'),
    'forecast_plot_path': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_results', 'forecast_plot_train.png')
}
