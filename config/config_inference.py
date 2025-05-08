import os

# === Base Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_NAME = 'dt-ood-repo'

# === Data Path ===
# TEST_DATA_DIR = os.path.join(BASE_DIR, 'NTNU', 'data_OOD_case_1')
# TEST_DATA_DIR = os.path.join(BASE_DIR, 'NTNU', 'data_OOD_case_2')
# TEST_DATA_DIR = os.path.join(BASE_DIR, 'NTNU', 'data_OOD_case_3')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'NTNU', 'data_OOD_case_all')

# === Feature Selection ===
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


# === Model & Threshold Paths ===
MODEL_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'phase2_model.pth')
SCALER_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'normalization_params.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_models', 'threshold_params.pkl')

# === Inference Results Directory ===
INFERENCE_RESULTS_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, 'ntnu_results', 'inference')

JSON_OUTPUT_PATH = os.path.join(INFERENCE_RESULTS_DIR, 'ood_forecast_window_diagnostics.json')

# === Inference Settings ===
SEQ_LEN = 60
FORECAST_HORIZON = 60
BATCH_SIZE = 64
