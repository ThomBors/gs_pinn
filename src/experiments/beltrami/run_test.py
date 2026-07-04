import torch
from src.experiments.beltrami.simulation_paras import *
from src.experiments.beltrami.data_sampler import BeltramiValidationDataSet
from src.experiments.beltrami.physical_residual import physical_residual


def run_test(
    network,
    x_start: float = X_START,
    x_end=X_END,
    y_start=Y_START,
    y_end=Y_END,
    z_start=Z_START,
    z_end=Z_END,
    simulation_time=SIMULATION_TIME,
    n_t=5,
    n_point=2001,
    device="cuda:0",
    return_ground_truth=False,
):
    network.eval()
    network.to(device)
    dataset = BeltramiValidationDataSet(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        z_start=z_start,
        z_end=z_end,
        simulation_time=simulation_time,
        n_t=n_t,
        n_point=n_point,
    )

    with torch.no_grad():
        prediction = network(
            torch.from_numpy(dataset.x).float().to(device),
            torch.from_numpy(dataset.y).float().to(device),
            torch.from_numpy(dataset.z).float().to(device),
            torch.from_numpy(dataset.t).float().to(device),
            0,
        )
        prediction = prediction.detach().cpu().numpy()
    mse = (dataset.uvwp - prediction) ** 2
    mse_value = np.mean(mse)
    shape = dataset.mesh_shape + (4,)
    if return_ground_truth:
        return (
            mse_value,
            mse.reshape(shape),
            prediction.reshape(shape),
            dataset.uvwp.reshape(shape),
        )
    else:
        return mse_value, mse.reshape(shape), prediction.reshape(shape)


def run_validation_loss(
    model,
    val_dataloader,
    device="cuda:0",
):
    model.eval()
    model.to(device)

    # Sample validation points
    x_internal, y_internal, z_internal, t_internal = val_dataloader.sample_internal()
    x_boundary, y_boundary, z_boundary, t_boundary, uvwp_boundary = (
        val_dataloader.sample_boundary()
    )
    x_initial, y_initial, z_initial, t_initial, uvwp_initial = (
        val_dataloader.sample_initial()
    )

    # PDE residual requires gradients
    x_internal.requires_grad_(True)
    y_internal.requires_grad_(True)
    z_internal.requires_grad_(True)
    t_internal.requires_grad_(True)

    # Forward passes
    u_internal = model(x_internal, y_internal, z_internal, t_internal, 0)
    u_boundary = model(x_boundary, y_boundary, z_boundary, t_boundary, 1)
    u_initial = model(x_initial, y_initial, z_initial, t_initial, 2)

    # Physics residuals
    loss_mx, loss_my, loss_mz, loss_c = physical_residual(
        u_internal,
        x_internal,
        y_internal,
        z_internal,
        t_internal,
    )

    loss_mx = torch.mean(loss_mx**2)
    loss_my = torch.mean(loss_my**2)
    loss_mz = torch.mean(loss_mz**2)
    loss_c = torch.mean(loss_c**2)

    internal_loss = loss_mx + loss_my + loss_mz + loss_c
    boundary_loss = torch.mean((u_boundary - uvwp_boundary) ** 2)
    initial_loss = torch.mean((u_initial - uvwp_initial) ** 2)

    total_loss = internal_loss + boundary_loss + initial_loss

    return (
        total_loss.item(),
        internal_loss.item(),
        boundary_loss.item(),
        initial_loss.item(),
    )
