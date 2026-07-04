from pathlib import Path
import rootutils
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
import logging
import hydra
import time
import torch
import wandb
from dotenv import load_dotenv
import tracemalloc

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.utils import (
    extract_weight_method_parameters_from_cfg,
    get_device,
    set_logger,
    set_seed,
    cleanup,
    build_run_name,
    loggerwandb,
    get_cosine_lambda,
)
from src.methods.weight_methods import WeightMethods
from src.methods.conflict import ConflictsMetrics, MetricMethods

from src.experiments.burgers.data_sampler import (
    BurgersValidationDataSet,
    BurgersSampler,
    BurgersValidationDataLoader,
)
from src.experiments.burgers.networks import BurgersNet
from src.experiments.burgers.physical_residual import physical_residual
from src.experiments.burgers.simulation_param import *
from src.experiments.burgers.run_test import run_test
from src.utils.memory import memory_report

set_logger()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Setup
    set_seed(cfg.random_seed)
    tracemalloc.start()

    device = torch.device(
        "cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Using device: {device}")

    # set wandb
    if cfg.logger.type == "wandb":
        load_dotenv()
        wandb.init(
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="offline" if cfg.logger.offline else "online",
        )

    # Paths
    res_path = Path(cfg.paths.out_path)
    res_path.mkdir(parents=True, exist_ok=True)

    chk_path = Path(cfg.paths.checkpoint_path)
    chk_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.paths.data_path) / "simulation_data.npy"

    # Model
    model = BurgersNet().to(device)

    # Data
    train_dataloader = BurgersSampler(
        n_internal=cfg.data.n_internal,
        n_initial=cfg.data.n_initial,
        n_boundary=cfg.data.n_boundary,
        device=device,
        update_data=cfg.data.update_training_data,
        seed=cfg.random_seed,
        data_sampler=cfg.data.data_sampler,
        x_start=cfg.data.x_start,
        x_end=cfg.data.x_end,
        simulation_time=cfg.data.simulation_time,
    )

    validation_dataset = BurgersValidationDataSet(dataset_path=data_path)

    vali_dataloader = BurgersValidationDataLoader(
        validation_dataset,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.trainer.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_cosine_lambda(
            initial_lr=cfg.trainer.lr,
            final_lr=cfg.trainer.final_lr,
            epochs=cfg.trainer.n_epochs,
            warmup_epoch=cfg.trainer.warmup_epoch,
        ),
    )

    epochs = cfg.trainer.n_epochs
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

    # Training loop
    pbar = trange(latest_epoch, epochs)
    for epoch in pbar:
        memory_report("start_epoch")

        model.train()

        t0 = time.time()

        # Sample collocation points
        x_internal, t_internal = train_dataloader.sample_internal()
        x_boundary, t_boundary, value_boundary = train_dataloader.sample_boundary()
        x_initial, t_initial, value_initial = train_dataloader.sample_initial()

        # Forward pass
        u_initial = model(x_initial, t_initial)
        u_boundary = model(x_boundary, t_boundary)
        u_internal = model(x_internal, t_internal)

        # PDE residual
        residual = physical_residual(
            u_internal,
            x_internal,
            t_internal,
        )

        # Losses
        internal_loss = torch.mean(residual**2)
        boundary_loss = torch.mean((u_boundary - value_boundary) ** 2)
        initial_loss = torch.mean((u_initial - value_initial) ** 2)

        losses = torch.stack(
            [
                internal_loss,
                boundary_loss,
                initial_loss,
            ]
        )
        memory_report("after_losses")

        ### conflicts computation
        metrics_all = wm(
            losses=losses,
            shared_parameters=list(model.shared_parameters()),
            task_specific_parameters=list(model.task_specific_parameters()),
            metric="all",
        )
        memory_report("after_conflicts")
        conflict_metric.add("train", epoch, 0, metrics_all)
        ###

        memory_report("after_conflict_store")

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
        mse_value, mse, prediction = run_test(
            model, vali_dataloader.simulation_data, device=device
        )

        metrics[epoch] = {
            "internal_loss": internal_loss.item(),
            "boundary_loss": boundary_loss.item(),
            "initial_loss": initial_loss.item(),
            "loss": loss.item(),
            "test_mse": mse_value,
        }
        memory_report("after_metrics_store")

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
            f"MSE: {mse_value}"
        )
        # logger
        if cfg.logger.type == "wandb":
            OmegaConf.resolve(cfg)
            cfg_wandb = OmegaConf.to_container(cfg, resolve=True)
            keys_to_remove = ["paths", "hydra", "logger"]
            for key in keys_to_remove:
                cfg_wandb.pop(key, None)
            loggerwandb(
                cfg=cfg_wandb,
                random_seed=cfg.random_seed,
                epoch=epoch,
                metrics=metrics,
            )
    save_path = res_path / f"{name}_prediction.stats"
    torch.save({"mse": mse, "prediction": prediction}, save_path)

    if cfg.save_model:
        model_path = chk_path / f"{name}.pt"
        torch.save(model.state_dict(), model_path)

    cleanup()


if __name__ == "__main__":
    main()
