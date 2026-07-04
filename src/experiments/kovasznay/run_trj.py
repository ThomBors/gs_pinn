from pathlib import Path
import rootutils
from omegaconf import DictConfig
from tqdm import trange
import logging
import hydra

import gc
import os
import copy
import wandb
import shutil
import socket

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

from src.experiments.kovasznay.sharpness import (
    SharpnessMetrics,
    eval_APGD_sharpness,
    eval_classification_sharpness,
)
import src.loss_landscape as ll

set_logger()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.random_seed)

    device = torch.device(
        "cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Using device: {device}")

    # Paths
    method_path = Path(cfg.optimization.method)

    chk_path = Path(cfg.paths.checkpoint_path)
    chk_path.mkdir(parents=True, exist_ok=True)

    ll_dir = Path(cfg.paths.ll_path)
    res_dir = ll_dir / method_path
    res_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.paths.data_path) / "simulation_data.npy"

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    name = build_run_name(cfg)
    model_path = chk_path / f"{name}.pt"

    model = KovasznayNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    w = ll.net_plotter.get_weights(model)
    s = model.state_dict()


    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files = []
    for epoch in range(0, cfg.trainer.n_epochs,100):
        model_dir = chk_path / 'training' / name
        model_file = model_dir / f"model_{epoch}.pt"
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)


    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    dir_file = ll.projection.setup_PCA_directions(res_dir, model_files,KovasznayNet().to(device), w, s,dir_type='weights',device=device)

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = ll.projection.project_trajectory(dir_file, w, s, KovasznayNet().to(device), model_files,device, 'lstsq')
    

    cleanup()


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



if __name__ == "__main__":
    main()
