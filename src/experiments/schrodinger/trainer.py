from pathlib import Path
import rootutils
from omegaconf import DictConfig
from tqdm import trange
import logging
import hydra
import time
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
from src.methods.conflict import ConflictsMetrics, MetricMethods

from src.experiments.schrodinger.physical_residual import *
from src.experiments.schrodinger.simulation_paras import *
from src.experiments.schrodinger.run_test import run_test
from src.experiments.schrodinger.data_sampler import (
    SchrodingerValidationDataSet,
    SchrodingerSampler,
    SchrodingerValidationDataLoader,
)
from src.experiments.schrodinger.networks import SchrodingerNet

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

    data_path = Path(cfg.paths.data_path) / "NLS.mat"

    # Model
    model = SchrodingerNet().to(device)

    # Data
    train_dataloader = SchrodingerSampler(
        n_internal=cfg.data.n_internal,
        n_initial=cfg.data.n_initial,
        n_boundary_left=cfg.data.n_boundary // 2,
        n_boundary_right=cfg.data.n_boundary // 2,
        device=device,
        update_data=cfg.data.update_training_data,
        seed=cfg.random_seed,
        data_sampler=cfg.data.data_sampler,
        x_start=cfg.data.x_start,
        x_end=cfg.data.x_end,
        simulation_time=cfg.data.simulation_time,
    )

    validation_dataset = SchrodingerValidationDataSet(dataset_path=data_path)

    vali_dataloader = SchrodingerValidationDataLoader(
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
        n_tasks=4,
        device=device,
        **method_params,
    )
    # Resume checkpoint
    latest_epoch = 0

    # Metrics
    metrics = {}
    conflict_metric = ConflictsMetrics()
    wm = MetricMethods("gc", n_tasks=4, device=device)

    # Training loop
    pbar = trange(latest_epoch, epochs)
    for epoch in pbar:

        model.train()

        t0 = time.time()

        # Sample collocation points
        x_internal, t_internal = train_dataloader.sample_internal()
        (
            xl_boundary,
            tl_boundary,
            xr_boundary,
            tr_boundary,
        ) = train_dataloader.sample_boundary()
        x_initial, t_initial, h_initial = train_dataloader.sample_initial()

        # Forward pass
        u_internal = model(x_internal, t_internal, 0)
        ul_boundary = model(xl_boundary, tl_boundary, 1)
        ur_boundary = model(xr_boundary, tr_boundary, 2)
        u_initial = model(x_initial, t_initial, 3)

        # PDE residual
        # Losses
        internal_loss = squared_physical_residual_internal(
            u_internal, x_internal, t_internal
        ).mean()
        boundary_loss_v, boundary_loss_f = squared_residual_boundary(
            ul_boundary, ur_boundary, xl_boundary, xr_boundary
        )
        boundary_loss_f = boundary_loss_f.mean()
        boundary_loss_v = boundary_loss_v.mean()
        initial_loss = squared_residual_initial(u_initial, h_initial).mean()

        losses = torch.stack(
            [
                internal_loss,
                boundary_loss_f,
                boundary_loss_v,
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
        mse_value, mse, prediction, ground_truth_data = run_test(
            model, data_path, device=device
        )

        metrics[epoch] = {
            "internal_loss": internal_loss.item(),
            "boundary_loss_f": boundary_loss_f.item(),
            "boundary_loss_v": boundary_loss_v.item(),
            "initial_loss": initial_loss.item(),
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
            f"MSE: {mse_value}"
        )
    save_path = res_path / f"{name}_prediction.stats"
    torch.save({"mse": mse, "prediction": prediction}, save_path)
    cleanup()

    save_path = res_path / f"{name}_prediction.stats"
    torch.save({"mse": mse, "prediction": prediction}, save_path)

    if cfg.save_model:
        model_path = chk_path / f"{name}.pt"
        torch.save(model.state_dict(), model_path)

    cleanup()


if __name__ == "__main__":
    main()
