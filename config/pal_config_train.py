import os

# === Base Directory ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_NAME = 'dt-ood-repo'

# === Data Paths ===
DATA_PATHS = {
    'train_data_dir': os.path.join(BASE_DIR, 'PAL', 'train'),
    'validation_data_dir': os.path.join(BASE_DIR, 'PAL', 'validation')
}

# === Features ===
INPUT_FEATURES = [
    'position_x', 'position_y', 
    'orientation_z', 'orientation_w',
    'linear_velocity_x', 'linear_velocity_y', 
    'angular_velocity_z'
]

OUTPUT_FEATURES = ['position_x', 'position_y']  # Only forecast x and y

# === Sequence Settings ===
SEQUENCE_SETTINGS = {
    'input_window_size': 60,
    'forecast_horizon_size': 60
}

# === Model Parameters ===
MODEL_PARAMS = {
    'd_model': 64,
    'num_heads': 4,
    'ff_dim': 128,
    'dropout_rate': 0.1
}

# === Training Parameters ===
TRAINING_PARAMS = {
    'batch_size': 64,
    'learning_rate': 0.00092,
    'epochs_phase1': 100,
    'epochs_phase2': 200,
    'early_stopping': {
        'enabled': False,
        'patience': 10,
        'patience_extension': 5
    }
}

# === Flags to Enable/Disable Phases ===
TRAINING_FLAGS = {
    'enable_phase1_training': True,
    'enable_phase2_training': False
}

# === Output Paths ===
OUTPUT_PATHS = {
    'phase1_model': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_models', 'phase1_model.pth'),
    'phase2_model': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_models', 'phase2_model.pth'),
    'normalization_params': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_models', 'scaler.pkl'),
    'threshold_params': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_models', 'thresholds.pkl'),
    'forecast_results_csv': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_results', 'forecast_results.csv'),
    'forecast_errors_csv': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_results', 'forecast_errors.csv'),
    'forecast_plot_path': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_results', 'forecast_plot.png'),
    'log_dir': os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_logs')
}
