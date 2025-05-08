import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MergedForecastReconModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, ff_dim,
                 input_window_size, forecast_horizon_size, dropout=0.1):
        super().__init__()
        self.input_window_size = input_window_size
        self.forecast_horizon_size = forecast_horizon_size

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max(input_window_size, forecast_horizon_size), d_model)
        )
        nn.init.xavier_uniform_(self.positional_encoding)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.forecast_input_projection = nn.Linear(output_dim, d_model)  # NEW
        self.output_projection = nn.Linear(d_model, output_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout),
            num_layers=3
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout),
            num_layers=3
        )

        mlp_hidden_dim = d_model * 2
        self.reconstruction_head = TinyMLP(d_model, mlp_hidden_dim, output_dim)

    def forward(self, x_input_seq, y_forecast_seq):
        # x_input_seq: [batch, input_window_size, input_dim]
        # y_forecast_seq: [batch, forecast_horizon_size, output_dim]

        x_proj = self.input_projection(x_input_seq) + self.positional_encoding[:, :self.input_window_size, :]
        memory = self.encoder(x_proj)

        y_proj = self.forecast_input_projection(y_forecast_seq) + self.positional_encoding[:, :self.forecast_horizon_size, :]
        forecast_hidden = self.decoder(y_proj, memory)

        forecast_pred = self.output_projection(forecast_hidden)
        reconstruction_pred = self.reconstruction_head(forecast_hidden)

        return forecast_pred, reconstruction_pred
