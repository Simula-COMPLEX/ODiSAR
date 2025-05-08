import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from data.dataset import TimeSeriesDataset

# === Logger Setup ===
def setup_logger(log_file_path):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

# === Phase 1: Forecast + Reconstruction Training ===
def train_phase1(model, train_data, input_window_size, forecast_horizon_size,
                 batch_size, learning_rate, num_epochs, device, logger, early_stopping=None,
                 ):
    model.train()
    dataset = TimeSeriesDataset(*train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    logger.info("Starting Phase 1: Forecast + Reconstruction Training")

    best_loss = float('inf')
    patience_counter = 0
    max_epochs = num_epochs

    if early_stopping and early_stopping.get('enabled', False):
        patience = early_stopping['patience']
        extension = early_stopping['patience_extension']
        logger.info(f"Early stopping enabled: patience={patience}, extension={extension}")
    else:
        patience = extension = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            forecast_out, recon_out = model(x, y)
            # forecast_target = y[:, :, output_indices]
            forecast_target = y
            loss = criterion(forecast_out, forecast_target) + criterion(recon_out, forecast_out.detach())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"[Phase 1] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

        if patience > 0:
            if avg_loss >= best_loss:
                patience_counter += 1
                if patience_counter == patience:
                    logger.info(f"No improvement for {patience} epochs. Extending training by {extension} more epochs.")
                    max_epochs = epoch + extension + 1
                elif patience_counter > patience and epoch + 1 >= max_epochs:
                    logger.info("Early stopping triggered. Ending Phase 1 training.")
                    break
            else:
                best_loss = avg_loss
                patience_counter = 0

# === Phase 2: Reconstruction Fine-tuning ===
def train_phase2(model, fine_tune_data, batch_size, learning_rate, num_epochs, device, logger, early_stopping=None):
    for name, param in model.named_parameters():
        if 'reconstruction_head' not in name:
            param.requires_grad = False

    model.train()
    dataset = TimeSeriesDataset(*fine_tune_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    logger.info("Starting Phase 2: Reconstruction Head Fine-tuning")

    best_loss = float('inf')
    patience_counter = 0
    max_epochs = num_epochs

    if early_stopping and early_stopping.get('enabled', False):
        patience = early_stopping['patience']
        extension = early_stopping['patience_extension']
        logger.info(f"Early stopping enabled: patience={patience}, extension={extension}")
    else:
        patience = extension = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            forecast_out, recon_out = model(x, y)
            loss = criterion(recon_out, forecast_out.detach())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"[Phase 2] Epoch {epoch+1}/{num_epochs} - Recon Loss: {avg_loss:.6f}")

        if patience > 0:
            if avg_loss >= best_loss:
                patience_counter += 1
                if patience_counter == patience:
                    logger.info(f"No improvement for {patience} epochs. Extending training by {extension} more epochs.")
                    max_epochs = epoch + extension + 1
                elif patience_counter > patience and epoch + 1 >= max_epochs:
                    logger.info("Early stopping triggered. Ending Phase 2 training.")
                    break
            else:
                best_loss = avg_loss
                patience_counter = 0

# === Validation: Forecast-Reconstruction Error ===
def compute_forecast_recon_errors(model, val_data, batch_size, device):
    model.eval()
    dataset = TimeSeriesDataset(*val_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            forecast_out, recon_out = model(x, y)
            forecast_target = y
            batch_errors = torch.sqrt(((forecast_out - recon_out) ** 2).mean(dim=(1, 2)))
            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)

# === Validation: Monte Carlo Forecast Variance ===
def compute_mc_forecast_variance(model, val_data, batch_size, device, n_passes):
    model.train()
    dataset = TimeSeriesDataset(*val_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    variances = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mc_outputs = []

        for _ in range(n_passes):
            forecast_out, _ = model(x, y)
            mc_outputs.append(forecast_out.detach().cpu())

        stacked = torch.stack(mc_outputs, dim=0)
        std_per_sample = torch.std(stacked, dim=0)
        mean_std = std_per_sample.mean(dim=(1, 2))
        variances.extend(mean_std.numpy())

    return np.array(variances)