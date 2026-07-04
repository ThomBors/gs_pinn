import torch.nn as nn
import torch

"""
Networks for the Kovasznay equation
"""


class KovasznayNet(nn.Module):

    def __init__(self, channel_basics=50, n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh()))
        for i in range(n_layers):
            self.net.append(
                nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh())
            )
        self.net = nn.Sequential(*self.net)
        self.shared_out = nn.Sequential(nn.Linear(channel_basics, 3))
        # self.loss_layer = nn.ModuleList([nn.Linear(3, 3) for _ in range(2)])

    def forward(self, x, y, loss_idx):
        y = torch.stack([x, y], dim=-1)
        y = self.net(y)
        y = self.shared_out(y)

        # if loss_idx is not None:

        #     if loss_idx not in [0, 1, 2]:
        #         raise ValueError(f"Invalid loss_idx: {loss_idx}")
        #     y = self.loss_layer[loss_idx](y)

        return y

    def shared_parameters(self):

        return (
            p for module in [self.net, self.shared_out] for p in module.parameters()
        )

    def task_specific_parameters(self):

        # return (p for layer in self.loss_layer for p in layer.parameters())
        return []

    def last_shared_parameters(self):

        # return self.shared_out.parameters()
        return []
