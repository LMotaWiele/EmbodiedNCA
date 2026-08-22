"""Pure residual NCA: 3×3 perception conv, 1×1 update net, clamp."""

import torch
import torch.nn as nn


class SimpleNCA(nn.Module):
    """Pure dynamic Neural Cellular Automata."""

    def __init__(self, channels=32, hidden_dim=64, grid_size=8):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size

        self.perception = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.update_net = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )

        nn.init.zeros_(self.update_net[-1].weight)
        nn.init.zeros_(self.update_net[-1].bias)

    def forward(self, x, steps=1):
        for _ in range(steps):
            perception_out = self.perception(x)
            update = self.update_net(perception_out)
            x = x + update
            x = torch.clamp(x, -10, 10)
        return x

    def init_grid(self, batch_size, device="cpu"):
        return torch.randn(
            batch_size, self.channels, self.grid_size, self.grid_size,
            device=device
        ) * 0.1