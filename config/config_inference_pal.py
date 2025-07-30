import os

# === Base Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_NAME = 'dt-ood-repo'

# === Data Path ===
TEST_DATA_DIR = os.path.join(BASE_DIR, 'PAL', 'ood-test-sim') 

# === Feature Selection ===
INPUT_FEATURES = [
    'position_x', 'position_y', 
    'orientation_z', 'orientation_w',
    'linear_velocity_x', 'linear_velocity_y', 
    'angular_velocity_z'
]

OUTPUT_FEATURES = ['position_x', 'position_y']  # Forecast x, y trajectory

# === Model & Threshold Paths ===
MODEL_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_model', 'phase1_model.pth')
THRESHOLD_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_model', 'thresholds.pkl')
SCALER_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_model', 'scaler.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_model', 'thresholds.pkl')

# === Inference Results Directory ===
INFERENCE_RESULTS_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'pal_results', 'inference')

os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)

JSON_OUTPUT_PATH = os.path.join(INFERENCE_RESULTS_DIR, 'ood_forecast_window_diagnostics.json')

# === Inference Settings ===
SEQ_LEN = 60
FORECAST_HORIZON = 60
BATCH_SIZE = 16
