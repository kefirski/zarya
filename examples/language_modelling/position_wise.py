import torch.nn as nn

import zarya.nn as znn


class PositionWise(nn.Module):
    def __init__(self, size, inner_size, manifild, dropout=0.1):
        super(PositionWise, self).__init__()

        self.mf = manifild

        self.fc = znn.Hyperbolic(
            nn.Sequential(
                nn.Linear(size, inner_size),
                nn.ReLU(),
                nn.Linear(inner_size, size),
                nn.Dropout(dropout),
            ),
            self.mf,
        )

        self.layer_norm = znn.Hyperbolic(nn.LayerNorm(size, eps=1e-12), self.mf)

    def forward(self, input, residual=None):
        if residual is None:
            residual = input

        result = self.fc(input)

        return self.layer_norm(self.mf.add(result, residual))
