
# Digital Twin OOD Detection with Transformer

This repository implements a Transformer-based approach for proactive **out-of-distribution (OOD)** detection using **forecasting and reconstruction error analysis**. It is structured around a two-phase training process and supports confidence-aware inference and explainability.

---

## 🚀 Setup

1. **Clone the Repository**

```bash
git clone https://github.com/ErblinIsaku/dt-ood-detection.git
cd dt-ood-detection
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

- Modify `config/config_train.py` to set:
  - `INPUT_FEATURES`, `OUTPUT_FEATURES`
  - `DATA_PATHS`, `SEQUENCE_SETTINGS`, `TRAINING_PARAMS`

- Modify `config/config_inference.py` to set:
  - `TEST_DATA_DIR`, `MODEL_PATH`, `SCALER_PATH`, etc.

---

## 🧠 Model Training

```bash
python main_train.py
```

Or for PAL-specific config:

```bash
python pal_main_train.py
```

This performs:
- **Phase 1**: Joint training for forecasting and reconstruction
- **Phase 2**: Fine-tuning the reconstruction head only
- Computes and saves reconstruction and uncertainty thresholds

---

## 🔍 Inference & OOD Detection

```bash
python main_inference.py
```

This performs:
- Forecasting on test data
- Computes forecast reconstruction error and MC-dropout variance
- Applies thresholds to detect OOD
- Generates visualizations and JSON-based diagnostics

---

## 📊 Outputs

- `forecast_results.csv`: Ground truth vs forecasted values
- `forecast_errors.csv`: MSE/RMSE per feature
- `thresholds.pkl`: Saved thresholds for inference
- `ood_diagnostics.json`: Confidence-aware OOD decision metadata
- Plots:
  - Forecasted vs GT curves
  - Reconstruction/variance distributions
  - Quadrant-based OOD scatter

---

## 📘 Notes

- Ensure that your **input and output features are defined clearly** and the combined feature set used for normalization does **not contain duplicates**.
- Inference will reuse the training-time normalization statistics.
- The script supports datasets with different feature combinations (e.g., Ship dynamics, Mobile robot navigation).

---
