import random

import numpy as np
import torch
import torch.nn as nn


class PriceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        return self.fc(x)


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_transformer_predicted_return(df, price_col="Close", window=64, seed=42):
    """Adds deterministic transformer-based predicted_return from sequence features."""
    set_global_seed(seed)

    feature_cols = [
        c
        for c in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "rsi",
            "macd",
            "atr",
            "momentum_20",
            "rolling_volatility_20",
            "sentiment",
            "market_index_trend",
            "vix_trend_20",
        ]
        if c in df.columns
    ]

    if not feature_cols:
        out = df.copy()
        out["predicted_return"] = 0.0
        return out

    feats = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats - mean) / std

    model = PriceTransformer(input_dim=feats.shape[1])
    model.eval()

    preds = np.full(len(df), np.nan, dtype=float)
    with torch.no_grad():
        for i in range(window, len(df)):
            seq = feats[i - window : i]
            x = torch.tensor(seq, dtype=torch.float32).view(1, window, feats.shape[1])
            pred = model(x).item()
            preds[i] = pred

    out = df.copy()
    # squeeze raw model outputs into a bounded return forecast for realism
    out["predicted_return"] = np.tanh(pd_series(preds).fillna(0.0).to_numpy()) * 0.05
    if price_col in out.columns:
        out["predicted_price"] = out[price_col] * (1.0 + out["predicted_return"])
    return out


def pd_series(values):
    try:
        import pandas as pd

        return pd.Series(values)
    except Exception:
        # fallback without pandas import issues
        class _Arr:
            def __init__(self, arr):
                self.arr = np.asarray(arr, dtype=float)

            def fillna(self, val):
                a = self.arr.copy()
                a[np.isnan(a)] = val
                return _Arr(a)

            def to_numpy(self):
                return self.arr

        return _Arr(values)
