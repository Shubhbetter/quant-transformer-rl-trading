import random

import numpy as np
import torch
import torch.nn as nn


class PriceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, model_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_transformer_predicted_return(df, price_col="Close", window=20, seed=42):
    """Adds deterministic transformer-based predicted_return feature."""
    set_global_seed(seed)
    prices = df[price_col].astype(float).to_numpy()

    model = PriceTransformer(input_dim=1)
    model.eval()

    preds = np.full(len(df), np.nan, dtype=float)

    with torch.no_grad():
        for i in range(window, len(df)):
            seq = prices[i - window : i]
            x = torch.tensor(seq, dtype=torch.float32).view(1, window, 1)
            pred = model(x).item()
            preds[i] = pred

    out = df.copy()
    out["predicted_price"] = preds
    out["predicted_return"] = (out["predicted_price"] - out[price_col]) / out[price_col].replace(0, np.nan)
    out["predicted_return"] = out["predicted_return"].fillna(0.0)
    return out
