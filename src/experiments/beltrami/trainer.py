from pathlib import Path
import rootutils
from omegaconf import DictConfig
from tqdm import trange
import logging
import hydra


import time

import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.experiments.beltrami.data_sampler import (
    BeltramiValidationDataSet,
    BeltramiSampler,
    BeltramiValidationDataLoader,
)
from src.experiments.beltrami.networks import BeltramiNet
from src.utils.utils import (
    extract_weight_method_parameters_from_cfg,
    get_device,
    set_logger,
    set_seed,
    build_run_name,
    cleanup,
    get_cosine_lambda,
)
from src.methods.weight_methods import WeightMethods
from src.experiments.beltrami.physical_residual import physical_residual
from src.experiments.beltrami.simulation_paras import *
from src.methods.conflict import ConflictsMetrics, MetricMethods
from src.experiments.beltrami.run_test import run_test

set_logger()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Setup
    set_seed(cfg.random_seed)

    device = torch.device(
        "cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Using device: {device}")

    # Paths
    res_path = Path(cfg.paths.out_path)
    res_path.mkdir(parents=True, exist_ok=True)

    chk_path = Path(cfg.paths.checkpoint_path)
    chk_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.paths.data_path) / "simulation_data.npy"

    # Model
    model = BeltramiNet().to(device)

    # Data
    train_dataloader = BeltramiSampler(
        n_internal=cfg.data.n_internal,
        n_initial=cfg.data.n_initial,
        n_boundary=cfg.data.n_boundary,
        device=device,
        update_data=cfg.data.update_training_data,
        seed=cfg.random_seed,
        data_sampler=cfg.data.data_sampler,
        x_start=cfg.data.x_start,
        x_end=cfg.data.x_end,
        y_start=cfg.data.y_start,
        y_end=cfg.data.y_end,
        z_start=cfg.data.z_start,
        z_end=cfg.data.z_end,
        simulation_time=cfg.data.simulation_time,
    )

    validation_dataset = BeltramiValidationDataSet(n_point=cfg.data.n_validation_point)

    vali_dataloader = BeltramiValidationDataLoader(
        validation_dataset,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.trainer.lr,
    )
    epochs = cfg.trainer.n_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_cosine_lambda(
            initial_lr=cfg.trainer.lr,
            final_lr=cfg.trainer.final_lr,
            epochs=epochs,
            warmup_epoch=min(100, int(epochs * 0.01)),
        ),
    )

    # Weighting method
    method_params = dict(cfg.optimization)
    method_params.pop("method", None)

    weight_method = WeightMethods(
        method=cfg.optimization.method,
        n_tasks=3,
        device=device,
        **method_params,
    )
    # Resume checkpoint
    latest_epoch = 0

    # Metrics
    metrics = {}
    conflict_metric = ConflictsMetrics()
    wm = MetricMethods("gc", n_tasks=3, device=device)
    nan_streak = 0
    
    # Training loop
    pbar = trange(latest_epoch, epochs)
    for epoch in pbar:

        model.train()

        t0 = time.time()

        # Sample collocation points
        x_internal, y_internal, z_internal, t_internal = (
            train_dataloader.sample_internal()
        )
        x_boundary, y_boundary, z_boundary, t_boundary, uvwp_boundary = (
            train_dataloader.sample_boundary()
        )
        x_initial, y_initial, z_initial, t_initial, uvwp_initial = (
            train_dataloader.sample_initial()
        )

        # Forward pass
        u_internal = model(x_internal, y_internal, z_internal, t_internal, 0)
        u_boundary = model(x_boundary, y_boundary, z_boundary, t_boundary, 1)
        u_initial = model(x_initial, y_initial, z_initial, t_initial, 2)

        # PDE residual
        loss_mx, loss_my, loss_mz, loss_c = physical_residual(
            u_internal, x_internal, y_internal, z_internal, t_internal
        )

        # Losses
        loss_mx = torch.mean(loss_mx**2)
        loss_my = torch.mean(loss_my**2)
        loss_mz = torch.mean(loss_mz**2)
        loss_c = torch.mean(loss_c**2)
        internal_loss = loss_mx + loss_my + loss_mz + loss_c
        boundary_loss = torch.mean((u_boundary - uvwp_boundary) ** 2)
        initial_loss = torch.mean((u_initial - uvwp_initial) ** 2)

        losses = torch.stack(
            [
                internal_loss,
                boundary_loss,
                initial_loss,
            ]
        )

        ### conflicts computation
        metrics_all = wm(
            losses=losses,
            shared_parameters=list(model.shared_parameters()),
            task_specific_parameters=list(model.task_specific_parameters()),
            metric="all",
        )
        conflict_metric.add("train", epoch, 0, metrics_all)
        ###

        # Zero gradients
        optimizer.zero_grad()

        # Backward
        loss, extra_outputs = weight_method.backward(
            losses=losses,
            shared_parameters=list(model.shared_parameters()),
            task_specific_parameters=list(model.task_specific_parameters()),
            last_shared_parameters=list(model.last_shared_parameters()),
        )

        optimizer.step()
        scheduler.step()
        t1 = time.time()

        # Testing
        (
            mse_value,
            mse,
            prediction,
        ) = run_test(model, device=device)

        is_invalid = (
            mse_value is None
            or (np.isnan(mse_value) if np.isscalar(mse_value) else False)
            or (prediction is not None and np.isnan(prediction).all())
        )

        if is_invalid:
            nan_streak += 1
            if nan_streak >= 5:
                break
            continue
        else:
            nan_streak = 0

            mse_history = mse_value
            pred_history = prediction

        metrics[epoch] = {
            "internal_loss_train": internal_loss.item(),
            "boundary_loss_train": boundary_loss.item(),
            "initial_loss_train": initial_loss.item(),
            "test_mse": mse_value,
        }

        t2 = time.time()

        name = build_run_name(cfg)

        if epoch % 1000 == 0:
            save_path = res_path / f"{name}.stats"
            torch.save(
                {"metric": metrics, "conflicts": conflict_metric.get_metrics()},
                save_path,
            )

        # Logging
        pbar.set_description(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.3e} "
            f"MSE: {mse_value:.3e}"
        )

    save_path = res_path / f"{name}_prediction.stats"
    torch.save({"mse": mse_history, "prediction": pred_history}, save_path)

    if cfg.save_model:
        model_path = chk_path / f"{name}.pt"
        torch.save(model.state_dict(), model_path)

    cleanup()


if __name__ == "__main__":
    main()
