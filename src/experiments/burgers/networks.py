import torch.nn as nn
import torch

"""
Networks for the Burgers equation
"""


class BurgersNetRes(nn.Module):
    def __init__(
        self,
        num_channels_input=2,
        channel_basics=20,
        channel_multiplier=[1] * 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = len(channel_multiplier) - 1
        self.layer_in = nn.Sequential(
            nn.Linear(num_channels_input, channel_multiplier[0] * channel_basics),
            nn.GroupNorm(
                num_groups=4, num_channels=channel_multiplier[0] * channel_basics
            ),
            nn.LeakyReLU(),
        )
        self.res_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_channel = channel_multiplier[i] * channel_basics
            out_channel = channel_multiplier[i + 1] * channel_basics
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channel, in_channel),
                    nn.GroupNorm(num_groups=4, num_channels=in_channel),
                    nn.LeakyReLU(),
                    nn.Linear(in_channel, out_channel),
                    nn.GroupNorm(num_groups=4, num_channels=out_channel),
                )
            )
        self.layer_out = nn.Linear(channel_multiplier[-1] * channel_basics, 1)

    def forward(self, x, t):
        ini_shape = x.shape
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)
        y = self.layer_in(y)
        for res_block in self.res_blocks:
            y = res_block(y)
            y = nn.functional.leaky_relu(y)
        y = self.layer_out(y)
        return y.view(ini_shape)


# class BurgersNet(nn.Module):

#     def __init__(self, channel_basics=50, n_layers=4, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.net = []
#         self.net.append(nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh()))
#         for i in range(n_layers):
#             self.net.append(
#                 nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh())
#             )
#         self.net = nn.Sequential(*self.net)
#         self.shared_out = nn.Linear(channel_basics, 1)
#         self.loss_layer = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])

#     def forward(self, x, t, loss_idx):
#         ini_shape = x.shape
#         y = torch.stack([x.view(-1), t.view(-1)], dim=-1)
#         y = torch.stack([x, t], dim=-1)
#         y = self.net(y)
#         y = self.shared_out(y)

#         if loss_idx is not None:

#             if loss_idx not in [0, 1, 2]:
#                 raise ValueError(f"Invalid loss_idx: {loss_idx}")
#             y = self.loss_layer[loss_idx](y)

#         return y.view(ini_shape)

#     def shared_parameters(self):

#         return (
#             p for module in [self.net, self.shared_out] for p in module.parameters()
#         )

#     def task_specific_parameters(self):

#         return (p for layer in self.loss_layer for p in layer.parameters())

#     def last_shared_parameters(self):

#         return self.shared_out.parameters()


class BurgersNet(nn.Module):

    def __init__(self, channel_basics=50, n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh()))
        for i in range(n_layers):
            self.net.append(
                nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh())
            )
        self.net = nn.Sequential(*self.net)
        self.shared_out = nn.Linear(channel_basics, 1)

    def forward(self, x, t, loss_idx=0):
        ini_shape = x.shape
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)
        y = torch.stack([x, t], dim=-1)
        y = self.net(y)
        y = self.shared_out(y)

        return y.view(ini_shape)

    def shared_parameters(self):

        return (
            p for module in [self.net, self.shared_out] for p in module.parameters()
        )

    def task_specific_parameters(self):

        return []

    def last_shared_parameters(self):

        return []
