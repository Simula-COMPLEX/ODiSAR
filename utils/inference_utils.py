import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# === Dataset Loader ===
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_sequences(input_df, output_df, seq_len, forecast_horizon):
    sequences, targets = [], []
    step = seq_len + forecast_horizon
    for i in range(0, len(input_df) - step + 1, step):
        x_seq = input_df.iloc[i : i + seq_len].values
        y_seq = output_df.iloc[i + seq_len : i + seq_len + forecast_horizon].values
        sequences.append(x_seq)
        targets.append(y_seq)
    return np.array(sequences), np.array(targets)

# === Directory Loader ===
def load_directory_data(dir_path, feature_names):
    all_data, ood_labels = [], []
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                decimal = ',' if sample.count(',') > sample.count('.') else '.'

            try:
                df = pd.read_csv(file_path, sep=';', decimal=decimal)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            if 'OOD' in df.columns:
                ood_labels.extend(df['OOD'].values)
                df.drop(columns=['OOD'], inplace=True, errors='ignore')

            df = df[[col for col in feature_names if col in df.columns]]
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid data files found in {dir_path}")

    combined = pd.concat(all_data, axis=0).reset_index(drop=True)
    return (combined, np.array(ood_labels)) if ood_labels else (combined, None)

# === Forecast Reconstruction Error ===
def compute_forecast_recon_errors(model, loader, device, output_feature_names):
    model.eval()
    errors, forecasts = [], []
    feature_reconstruction_errors = {}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            forecast_out, recon_out = model(x, y)

            # Overall error per sample
            err = torch.sqrt(((forecast_out - recon_out) ** 2).mean(dim=(1, 2)))
            errors.extend(err.cpu().numpy())

            # Per-feature reconstruction errors
            for i, feature_name in enumerate(output_feature_names):
                feature_error = torch.sqrt(((forecast_out[:, :, i] - recon_out[:, :, i]) ** 2).mean(dim=1))
                feature_reconstruction_errors.setdefault(feature_name, []).extend(feature_error.cpu().numpy())

            forecasts.append(forecast_out.cpu().numpy())

    return np.array(errors), np.concatenate(forecasts), feature_reconstruction_errors


# === Forecast Variance with MC Dropout ===
def compute_forecast_variance(model, loader, device, n_passes=20):
    model.train()
    variances = []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        preds = [model(x, y)[0].detach().cpu().numpy() for _ in range(n_passes)]
        pred_stack = np.stack(preds)
        var = np.var(pred_stack, axis=0).mean(axis=(1, 2))
        variances.extend(var)
    return np.array(variances)

# === Evaluation Utility ===
def evaluate_ood_detection(recon_errors, forecast_variances, recon_thresh, var_thresh,
                           forecast_horizon, ood_labels, feature_names):
    if ood_labels is None:
        print("No ground truth labels provided.")
        return

    ood_flags = recon_errors > recon_thresh
    expanded_flags = np.repeat(ood_flags, forecast_horizon)

    gt_labels = []
    for i in range(0, len(ood_labels) - forecast_horizon, forecast_horizon + forecast_horizon):
        gt_labels.extend(ood_labels[i + forecast_horizon : i + forecast_horizon + forecast_horizon])
    gt_labels = np.array(gt_labels[:len(expanded_flags)])

    print(classification_report(gt_labels, expanded_flags))
    print(confusion_matrix(gt_labels, expanded_flags))

    if len(np.unique(gt_labels)) == 2:
        score = np.repeat(recon_errors, forecast_horizon)[:len(gt_labels)]
        auroc = roc_auc_score(gt_labels, score)
        fpr, tpr, _ = roc_curve(gt_labels, score)
        tnr_at_95 = 1.0 - fpr[np.argmin(np.abs(tpr - 0.95))]

        print(f"AUROC: {auroc:.4f}")
        print(f"TNR@TPR95: {tnr_at_95:.4f}")


# === JSON Output Utility ===
def export_ood_diagnostics_json(recon_errors, forecast_variances, recon_thresh, var_thresh,
                                forecast_horizon, feature_names, feature_recon_errors, output_path):
    json_results = []
    colors = []
    categories = []

    for i, (rerr, var) in enumerate(zip(recon_errors, forecast_variances)):
        is_ood = bool(rerr > recon_thresh)
        is_uncertain = bool(var > var_thresh)

        # Top-3 feature attribution
        sequence_errors = {
            feature: feature_recon_errors[feature][i]
            for feature in feature_names if feature in feature_recon_errors
        }
        sorted_features = sorted(sequence_errors.items(), key=lambda x: x[1], reverse=True)[:3]
        top_features = {feature: float(error) for feature, error in sorted_features}

        # Assign color and category
        if not is_ood and not is_uncertain:
            color = "green"
            category = "IND & Confident"
        elif not is_ood and is_uncertain:
            color = "yellow"
            category = "IND & Uncertain"
        elif is_ood and is_uncertain:
            color = "orange"
            category = "OOD & Uncertain"
        else:
            color = "red"
            category = "OOD & Confident"

        colors.append(color)
        categories.append(category)

        json_results.append({
            "sequence_index": int(i),
            "start_time_step": int(i * (forecast_horizon * 2) + forecast_horizon),
            "end_time_step": int(i * (forecast_horizon * 2) + forecast_horizon + forecast_horizon - 1),
            "is_OOD": is_ood,
            "reconstruction_error": float(rerr),
            "uncertainty_variance": float(var),
            "recon_exceeds_threshold": is_ood,
            "uncertainty_exceeds_threshold": is_uncertain,
            "category": color,  # Store only the color name
            "state_attribution": top_features if is_ood else None
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved detailed OOD diagnostics to {output_path}")
    return colors, categories
