import torch
import numpy as np
from src.experiments.burgers.simulation_param import *


# Adapted from:https://github.com/erfanhamdi/pinn-torch/blob/main/Schrodingers_Equation/main.py
def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        y = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
    return y


def physical_residual(u, x, t, nu=NU):
    x.grad = None
    t.grad = None
    """ Physics-based loss function with Burgers equation """
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_xx = derivative(u_x, x, order=1)
    return u_t + u * u_x - nu * u_xx
