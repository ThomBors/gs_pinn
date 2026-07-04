import wandb
from pathlib import Path
import rootutils
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
import logging
import hydra
import time
from dotenv import load_dotenv
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.experiments.kovasznay.data_sampler import (
    KovasznayValidationDataSet,
    KovasznaySampler,
    KovasznayValidationDataLoader,
)
from src.experiments.kovasznay.networks import KovasznayNet
from src.utils.utils import (
    extract_weight_method_parameters_from_cfg,
    get_device,
    set_logger,
    set_seed,
    cleanup,
    loggerwandb,
    build_run_name,
    get_cosine_lambda,
)
from src.methods.weight_methods import WeightMethods
from src.experiments.kovasznay.physical_residual import physical_residual
from src.experiments.kovasznay.simulation_paras import *
from src.methods.conflict import ConflictsMetrics, MetricMethods
from src.experiments.kovasznay.run_test import run_test

set_logger()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Setup
    set_seed(cfg.random_seed)

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

    # Model
    model = KovasznayNet().to(device)

    # Data
    train_dataloader = KovasznaySampler(
        n_internal=cfg.data.n_internal,
        n_boundary=cfg.data.n_boundary,
        device=device,
        update_data=cfg.data.update_training_data,
        seed=cfg.random_seed,
        data_sampler=cfg.data.data_sampler,
        x_start=cfg.data.x_start,
        x_end=cfg.data.x_end,
        y_start=cfg.data.y_start,
        y_end=cfg.data.y_end,
    )

    validation_dataset = KovasznayValidationDataSet(n_point=cfg.data.n_validation_point)

    vali_dataloader = KovasznayValidationDataLoader(
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
        n_tasks=2,
        device=device,
        **method_params,
    )
    # Resume checkpoint
    latest_epoch = 0

    # Metrics
    metrics = {}
    conflict_metric = ConflictsMetrics()
    wm = MetricMethods("gc", n_tasks=2, device=device)
    nan_streak = 0
    
    # Training loop
    pbar = trange(latest_epoch, epochs)
    for epoch in pbar:

        model.train()

        t0 = time.time()

        # Sample collocation points
        x_internal, t_internal = train_dataloader.sample_internal()
        x_boundary, t_boundary, uvp_boundary = train_dataloader.sample_boundary()

        # Forward pass
        u_internal = model(x_internal, t_internal, 0)
        u_boundary = model(x_boundary, t_boundary, 1)

        # PDE residual
        loss_mx, loss_my, loss_c = physical_residual(
            u_internal,
            x_internal,
            t_internal,
        )

        # Losses
        loss_mx = torch.mean(loss_mx**2)
        loss_my = torch.mean(loss_my**2)
        loss_c = torch.mean(loss_c**2)
        internal_loss = loss_mx + loss_my + loss_c
        bound_loss = torch.mean((u_boundary - uvp_boundary) ** 2)

        losses = torch.stack([internal_loss, bound_loss])

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
        mse_value, mse, prediction = run_test(model, device=device)

        metrics[epoch] = {
            "internal_loss": internal_loss.item(),
            "boundary_loss": bound_loss.item(),
            "test_mse": mse_value,
        }
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

        mse_history = mse
        pred_history = prediction

        t2 = time.time()

        name = build_run_name(cfg)

        if epoch % 100 == 0:
            save_path = res_path / f"{name}.stats"
            torch.save(
                {"metric": metrics, "conflicts": conflict_metric.get_metrics()},
                save_path,
            )

            if cfg.chk_save:
                train_path = chk_path / "training" / name
                train_path.mkdir(parents=True, exist_ok=True)
                save_path = train_path / f"model_{epoch}.pt"
                torch.save(
                    model.state_dict(),
                    save_path
                )
                

        # Logging
        pbar.set_description(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.3e} "
            f"MSE: {mse_value:.3e}"
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
    torch.save({"mse": mse_history, "prediction": pred_history}, save_path)

    if cfg.final_model_save:
        model_path = chk_path / f"{name}.pt"
        torch.save(model.state_dict(), model_path)

    cleanup()


if __name__ == "__main__":
    main()
