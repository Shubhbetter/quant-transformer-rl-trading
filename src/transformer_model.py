import torch
import torch.nn as nn

class PriceTransformer(nn.Module):

    def __init__(self, input_dim, model_dim=64):

        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=4
            ),
            num_layers=2
        )

        self.fc = nn.Linear(model_dim,1)

        self.input_layer = nn.Linear(input_dim, model_dim)

    def forward(self, x):

        x = self.input_layer(x)

        x = self.encoder(x)

        x = x.mean(dim=1)

        return self.fc(x)
