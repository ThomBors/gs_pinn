import torch.nn as nn
import torch


class SchrodingerNet(nn.Module):

    def __init__(self, channel_basics=50, n_layes=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh()))
        for i in range(n_layes):
            self.net.append(
                nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh())
            )
        self.net = nn.Sequential(*self.net)
        self.shared_out = nn.Linear(channel_basics, 2)
        # self.loss_layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])

    def forward(self, x, t, loss_idx):
        y = torch.stack([x, t], dim=-1)
        y = self.net(y)
        y = self.shared_out(y)
        # if loss_idx is not None:

        #     if loss_idx not in [0, 1, 2, 3]:
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
