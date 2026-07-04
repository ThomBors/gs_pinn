import logging
import random
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch
import math
import wandb
import gc
from typing import Optional, List, Union, Dict, Any
from omegaconf import DictConfig


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def build_run_name(cfg: DictConfig) -> str:
    method = cfg.optimization.method

    parts = [
        method,
        f"sd{cfg.random_seed}",
    ]

    # dynamically add all optimizer params
    for key, value in cfg.optimization.items():
        if key == "method":
            continue

        parts.append(f"{key}{value}")

    return "_".join(parts)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def cleanup():
    # Finish WandB
    if wandb.run:
        wandb.finish()

    # Free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Clear any remaining references
    gc.collect()

    logging.info(f"{'='*40} Training complete {'='*40}")


def loggerwandb(
    cfg: Dict[str, Any],
    epoch: int,
    metrics: Dict[int, Dict[str, float]],
    random_seed: Optional[int] = 42,
) -> None:
    if wandb.run is None:
        return

    log_data = dict(cfg)  # avoid mutating cfg

    log_data["random_seed"] = random_seed
    log_data["epoch"] = epoch

    # add epoch metrics
    if epoch in metrics:
        log_data.update(metrics[epoch])

    wandb.log(log_data)


def to_numpy(x: Union[np.ndarray, torch.Tensor, List[float]]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_cosine_lambda(initial_lr, final_lr, epochs, warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the cosine schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """

    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1 - (
                1
                - (
                    math.cos(
                        (idx_epoch - warmup_epoch) / (epochs - warmup_epoch) * math.pi
                    )
                    + 1
                )
                / 2
            ) * (1 - final_lr / initial_lr)

    return cosine_lambda


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
                max_norm=args.max_norm,
            ),
            stl=dict(main_task=args.main_task),
            dwa=dict(temp=args.dwa_temp),
            cagrad=dict(c=args.c, max_norm=args.max_norm),
            log_cagrad=dict(c=args.c, max_norm=args.max_norm),
            famo=dict(
                gamma=args.gamma, w_lr=args.method_params_lr, max_norm=args.max_norm
            ),
        )
    )
    return weight_methods_parameters


def extract_weight_method_parameters_from_cfg(method_cfg):

    # Create a dictionary to store parameters for each weight method
    weight_methods_parameters = defaultdict(dict)

    # Update with the necessary parameters for different methods
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=method_cfg.update_weights_every,
                optim_niter=method_cfg.nashmtl_optim_niter,
                max_norm=method_cfg.max_norm,
            ),
            stl=dict(main_task=method_cfg.main_task),
            dwa=dict(temp=method_cfg.dwa_temp),
            cagrad=dict(c=method_cfg.c, max_norm=method_cfg.max_norm),
            log_cagrad=dict(c=method_cfg.c, max_norm=method_cfg.max_norm),
            famo=dict(
                gamma=method_cfg.gamma,
                w_lr=method_cfg.method_params_lr,
                max_norm=method_cfg.max_norm,
            ),
        )
    )
    return weight_methods_parameters


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, 1)
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.fill_(0.001)
