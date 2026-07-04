#!/usr/bin/env python

##################################################################
# Adaptation of code from: https://github.com/Cranial-XIX/CAGrad #
##################################################################

import torch
import numpy as np
from scipy.optimize import minimize_scalar

from matplotlib import pyplot as plt
from matplotlib import path as mpath

import seaborn as sns
from PIL import Image

import logging
from tqdm import tqdm


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def conflict_grads(grads):
    g1 = grads[:, 0]  # First column (gradient for task 1)
    g2 = grads[:, 1]  # Second column (gradient for task 2)
    return torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2))


def gradient_magnitude_similarity(grads):
    g1 = grads[:, 0]  # First column (gradient for task 1)
    g2 = grads[:, 1]  # Second column (gradient for task 2)

    norm_gi = torch.norm(g1)
    norm_gj = torch.norm(g2)
    return (2 * norm_gi * norm_gj) / (norm_gi**2 + norm_gj**2)


def curvarute_bounding_measure(grads):
    g1 = grads[:, 0]
    g2 = grads[:, 1]

    # Calculate the cosine of the angle between g1 and g2
    cos_phi_12 = conflict_grads(grads)

    # Calculate the components of ξ(g1, g2)
    cos2_phi_12 = cos_phi_12**2
    norm_diff_squared = torch.norm(g1 - g2) ** 2
    norm_sum_squared = torch.norm(g1 + g2) ** 2

    # Calculate ξ(g1, g2)
    return (1 - cos2_phi_12) * (norm_diff_squared / norm_sum_squared)


# def multitask_curvature():
#     H =


def convert_to_serializable(obj):
    """
    Recursively convert NumPy arrays, PyTorch tensors, and other non-serializable objects
    in a nested dictionary or list structure to JSON-serializable formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    else:
        return obj


# plot optimization


def gradient_aggegation(method, tt, obs):
    grads = tt["grad"][obs]

    if method == "cagrad":
        grad_combination = cagrad(grads=grads, c=0.5)
    elif method == "pcgrad":
        grad_combination = pcgrad(grads)
    elif method == "graddrop":
        grad_combination = graddrop(grads=grads)
    elif method == "famo":
        weight = tt["weight"][obs]["weights"].detach().numpy().copy()
        grad_combination = famo(grads=grads, weight=weight)
    elif method == "alignedmtl":
        grad_combination = alignedmtl(grads)
    elif method == "l2bmgrad":
        grad_combination = l2balance_magnitude(grads)
    elif method == "logbmgrad":
        grad_combination = logbalance_magnitude(grads)
    elif method == "minbmgrad":
        grad_combination = minbalance_magnitude(grads)
    elif method == "thbmgrad":
        grad_combination = thbalance_magnitude(grads)
    elif method == "gsb":
        grad_combination = gsb(grads)
    else:
        grad_combination = uws(grads=grads)

    return grad_combination


# https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py
def uws(grads):
    """uniformed weighted sum"""
    g1 = grads[:, 0]
    g2 = grads[:, 1]
    return g1 + g2


def cagrad(grads, c=0.5):
    g1 = grads[:, 0]
    g2 = grads[:, 1]
    g0 = (g1 + g2) / 2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12 + 1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return (
            coef
            * np.sqrt(x**2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-4)
            + 0.5 * x * (g11 + g22 - 2 * g12)
            + (0.5 + x) * (g12 - g22)
            + g22
        )

    res = minimize_scalar(obj, bounds=(0, 1), method="bounded")
    x = res.x

    gw = x * g1 + (1 - x) * g2
    gw_norm = np.sqrt(x**2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-4)

    lmbda = coef / (gw_norm + 1e-4)
    g = g0 + lmbda * gw
    return g / (1 + c)


def pcgrad(grads):
    g1 = grads[:, 0]
    g2 = grads[:, 1]
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1 - g12 / g11) * g1 + (1 - g12 / g22) * g2) / 2
    else:
        return (g1 + g2) / 2


def graddrop(grads):
    grads = torch.from_numpy(grads)
    P = 0.5 * (1.0 + grads.sum(0) / (grads.abs().sum(0) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)

    return g


def famo(grads, weight):
    g1 = grads[:, 0]
    g2 = grads[:, 1]
    z1 = weight[0]
    z2 = weight[1]

    return g1 * z1 + g2 * z2


def alignedmtl(grads, weights=None):
    grads = torch.from_numpy(grads)
    M = grads.T @ grads  # Compute Gram matrix

    # Compute eigenvalues and eigenvectors
    singulars, basis = torch.linalg.eigh(M)

    # Sorting in descending order
    sorted_indices = torch.argsort(singulars, descending=True)
    singulars = singulars[sorted_indices]
    basis = basis[:, sorted_indices]

    # Compute tolerance and rank
    tol = singulars.max() * max(M.shape) * torch.finfo(singulars.dtype).eps
    rank = (singulars > tol).sum().item()

    # Filter based on rank
    singulars = singulars[:rank]
    basis = basis[:, :rank]

    # Compute the inverse square root of singular values
    inv_sqrt_singulars = torch.diag(1.0 / torch.sqrt(singulars))
    inv_sqrt_singulars[torch.isinf(inv_sqrt_singulars)] = 0  # Handle inf

    # Compute the transformation matrix B
    # Compute balance transformation matrix B = √λRV Σ−1V⊤
    sqrt_singulars = torch.sqrt(singulars)
    B = sqrt_singulars[-1].view(1, -1) * basis @ inv_sqrt_singulars @ basis.T

    # If no user-provided weights, default to equal weights
    if weights is None:
        weights = torch.diag(torch.ones(grads.shape[1], device=grads.device))

    # Compute the final weighted gradient: α = Bw
    alpha = torch.matmul(B, weights)

    # Apply the transformation to the gradients: Gα
    aligned_grads = grads @ alpha

    return aligned_grads.sum(1)


def l2balance_magnitude(grads):
    w = torch.from_numpy(grads.copy())
    # Calculate magnitudes
    mg = torch.linalg.norm(w, dim=0)

    # Normalize magnitudes for similarity computation
    mg_matrix = mg.unsqueeze(1)  # Shape: (N, 1)
    mg_matrix_T = mg_matrix.T  # Shape: (1, N)

    # Compute pairwise similarities using broadcasting
    similarities_matrix = (
        2 * (mg_matrix @ mg_matrix_T) / (mg_matrix**2 + mg_matrix_T**2)
    )

    # Extract upper triangle (or lower triangle) without diagonal to get pairwise similarities
    # Since similarities_matrix is symmetric, we can use the upper triangle
    mask = torch.triu(
        torch.ones_like(similarities_matrix, dtype=torch.bool), diagonal=1
    )
    similarities = similarities_matrix[mask]

    # Display all pairwise similarities

    # Check if any similarity is below the threshold
    threshold = 0.01
    if (similarities < threshold).any():

        #### L2 Normalization ####
        # Compute L2 norms of the vectors
        l2_norms = torch.norm(w, dim=0, p=2)

        # Rescale each vector to have unit L2 norm
        scaled_w_l2 = w / l2_norms

        # Sum of L2-normalized vectors
        g = scaled_w_l2.sum(1)
    else:
        g = w.sum(1)

    return g


def minbalance_magnitude(grads):
    w = torch.from_numpy(grads.copy())
    # Calculate magnitudes
    mg = torch.linalg.norm(w, dim=0)

    # Normalize magnitudes for similarity computation
    mg_matrix = mg.unsqueeze(1)  # Shape: (N, 1)
    mg_matrix_T = mg_matrix.T  # Shape: (1, N)

    # Compute pairwise similarities using broadcasting
    similarities_matrix = (
        2 * (mg_matrix @ mg_matrix_T) / (mg_matrix**2 + mg_matrix_T**2)
    )

    # Extract upper triangle (or lower triangle) without diagonal to get pairwise similarities
    # Since similarities_matrix is symmetric, we can use the upper triangle
    mask = torch.triu(
        torch.ones_like(similarities_matrix, dtype=torch.bool), diagonal=1
    )
    similarities = similarities_matrix[mask]

    # Display all pairwise similarities

    # Check if any similarity is below the threshold
    threshold = 0.01
    if (similarities < threshold).any():

        #### min Normalization ####
        target_magnitude = torch.min(mg)

        # Scale the vectors based on the target magnitude
        scaling_factors = target_magnitude / mg

        # Reshape scaling factors to broadcast across vectors
        scaled_w = w * scaling_factors  # Broadcasting the scaling factors

        # Sum of L2-normalized vectors
        g = scaled_w.sum(1)
    else:
        g = w.sum(1)

    return g


def logbalance_magnitude(grads):
    w = torch.from_numpy(grads.copy())
    # Calculate magnitudes
    mg = torch.linalg.norm(w, dim=0)

    # Normalize magnitudes for similarity computation
    mg_matrix = mg.unsqueeze(1)  # Shape: (N, 1)
    mg_matrix_T = mg_matrix.T  # Shape: (1, N)

    # Compute pairwise similarities using broadcasting
    similarities_matrix = (
        2 * (mg_matrix @ mg_matrix_T) / (mg_matrix**2 + mg_matrix_T**2)
    )

    # Extract upper triangle (or lower triangle) without diagonal to get pairwise similarities
    # Since similarities_matrix is symmetric, we can use the upper triangle
    mask = torch.triu(
        torch.ones_like(similarities_matrix, dtype=torch.bool), diagonal=1
    )
    similarities = similarities_matrix[mask]

    # Display all pairwise similarities

    # Check if any similarity is below the threshold
    threshold = 0.01
    if (similarities < threshold).any():

        #### min Normalization ####
        log_mg = torch.log1p(mg)  # log(1 + mg) to avoid log(0)

        # Scaling factor based on logarithmic magnitudes
        scaling_factors_log = log_mg / mg

        # Reshape for broadcasting
        scaled_w_log = w * scaling_factors_log

        # Sum of normalized vectors
        g = scaled_w_log.sum(1)
    else:
        g = w.sum(1)

    return g


def thbalance_magnitude(grads):
    w = torch.from_numpy(grads).clone().detach()
    # Calculate magnitudes
    mg = torch.linalg.norm(w, dim=0)

    # Normalize magnitudes for similarity computation
    mg_matrix = mg.unsqueeze(1)  # Shape: (N, 1)
    mg_matrix_T = mg_matrix.T  # Shape: (1, N)

    # Compute pairwise similarities using broadcasting
    similarities_matrix = (
        2 * (mg_matrix @ mg_matrix_T) / (mg_matrix**2 + mg_matrix_T**2)
    )

    # Extract upper triangle (or lower triangle) without diagonal to get pairwise similarities
    # Since similarities_matrix is symmetric, we can use the upper triangle
    mask = torch.triu(
        torch.ones_like(similarities_matrix, dtype=torch.bool), diagonal=1
    )
    similarities = similarities_matrix[mask]
    # compute conflict
    g1 = w[:, 0]  # First column (gradient for task 1)
    g2 = w[:, 1]  # Second column (gradient for task 2)

    norm_gi = torch.norm(g1)
    norm_gj = torch.norm(g2)
    conf = (2 * norm_gi * norm_gj) / (norm_gi**2 + norm_gj**2)

    # Check if any similarity is below the threshold
    threshold = 0.01
    if ((similarities < threshold).any()) & (conf < 0.5):
        #### min standardization ####
        target_magnitude = torch.min(mg)

        # Scale the vectors based on the target magnitude
        scaling_factors = target_magnitude / mg

        # Reshape scaling factors to broadcast across vectors
        scaled_w = w * scaling_factors  # Broadcasting the scaling factors

        # Sum of L2-normalized vectors
        g = scaled_w.sum(1)
        aligned_w = scaled_w
        w = scaling_factors

    elif ((similarities > 0.95).any()) & (conf < 0.5):
        log_mg = torch.log1p(mg)  # log(1 + mg) to avoid log(0)

        # Scaling factor based on logarithmic magnitudes
        scaling_factors_log = log_mg / mg

        # Reshape for broadcasting
        scaled_w_log = w * scaling_factors_log

        # Sum of normalized vectors
        g = scaled_w_log.sum(1)
        aligned_w = scaled_w_log
        w = scaling_factors_log

    elif ((similarities > 0.75).all()) & (conf > 0.5):
        g = w.sum(1) * 100
        aligned_w = None
        w = None

    else:
        g = w.sum(1)
        aligned_w = None
        w = None

    return g


def gsb(vv):
    """https://github.com/legendongary/pytorch-gram-schmidt"""

    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    vv = torch.from_numpy(vv)
    t, n = vv.size()
    uu = torch.zeros_like(vv, device=vv.device)

    for k in range(t):
        vk = vv[k, :].clone()
        uk = torch.zeros_like(vk)

        # Calculate projections onto already orthogonalized vectors
        for j in range(k):
            uj = uu[j, :].clone()
            uk += projection(uj, vk)

        # Subtract projections to get orthogonal vector
        uk = vk - uk

        # Check for linear dependence
        if uk.norm() < 1e-10:  # Small threshold to handle numerical stability
            print(f"Warning: Vector {k} is linearly dependent on previous vectors.")
            # If the vector is dependent, we can skip normalization or handle it as needed
            uu = vv
        else:
            # Normalize the orthogonal vector
            uu[k, :] = uk / uk.norm()

    return uu.sum(1)


def list_to_layer(layer_num):
    """
    Convert a layer index into a list of corresponding parameter names.
    (Specific to SimpleMTL architecture.)

    Args:
        layer_num (int): The index of the layer of interest.

    Returns:
        list[str]: A list of parameter names corresponding to that layer.
    """
    if layer_num < 3:
        return [f"shared_base.{layer_num}.0.weight"]
    elif layer_num == 3:
        return [
            "out_layer.0.0.weight",
            "out_layer.1.0.weight",
            "out_layer.2.0.weight",
        ]
    elif layer_num == 4:
        return [
            "out_layer.0.2.weight",
            "out_layer.1.2.weight",
            "out_layer.2.2.weight",
        ]
    else:
        raise ValueError(f"Unsupported layer number: {layer_num}")
