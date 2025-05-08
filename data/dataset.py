import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_directory_data(dir_path, feature_names):
    """
    Loads and concatenates all CSV files in the given directory,
    selecting only the specified feature columns.
    """
    all_data = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
            df = df[[col for col in feature_names if col in df.columns]]
            all_data.append(df)
    if not all_data:
        raise ValueError(f"No valid data in {dir_path}")
    return pd.concat(all_data, axis=0).reset_index(drop=True)


def create_sequences(input_df, output_df, input_window_size, forecast_horizon_size, overlap=True):
    sequences = []
    targets = []
    step_size = 1 if overlap else (input_window_size + forecast_horizon_size)

    for i in range(0, len(input_df) - input_window_size - forecast_horizon_size + 1, step_size):
        x_seq = input_df.iloc[i : i + input_window_size].values
        y_seq = output_df.iloc[i + input_window_size : i + input_window_size + forecast_horizon_size].values
        sequences.append(x_seq)
        targets.append(y_seq)

    return np.array(sequences), np.array(targets)


