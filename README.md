# Gradient Similarity Surgery in Multi-task Deep Learning
Official implementation of the Gradient surgery methods for training Physics-Informed Neural Networks (under review at ACML)

Physics-Informed Neural Networks (PINNs) optimise a composite objective that couples data fitting with physics-based constraints, yielding a highly imbalanced multi-task learning (MTL) problem. In practice, naive loss aggregation leads to conflicting gradient signals, slow convergence, and unstable training dynamics, particularly for stiff or high-frequency partial differential equations (PDEs).
In this work, we conduct an extensive evaluation of modern MTL optimisation methods in the context of PINNs and provide an analysis of gradient conflicts during training. 
Based on these insights, we propose PAM-GS, a physics-tailored gradient surgery method that leverages a landscape-aware strategy to mitigate gradient interference and improve optimisation performance.
Empirically, we evaluate our method on canonical PDE benchmarks and show consistent improvements in convergence speed and solution accuracy over existing optimisation strategies. 
This study provides a systematic study of MTL optimisation for PINNs and highlights the importance of landscape-aware gradient manipulation.


---

<p align="center"> 
    <img src="./extra/loss_landscape_traj_0.gif" width="800">
</p>
<p align="center"> 
    <img src="./extra/loss_landscape_traj_1.gif" width="800">
</p>
<p align="center"> 
    <img src="./extra/loss_landscape_traj_MTL.gif" width="800">
</p>

Loss landscape of the 2D Kovasznay flow across optimisers. Panels show $\log_{10}$ training loss over a 2D PCA subspace with iso-contours and learning trajectories (blue) on the loss 3D surface and 2D projection.

---




## Experiments

### Setup environment

Create the virtual environment and install project dependencies:

```bash
cd gs_pinn
uv sync
```

### Project structure

The file hierarchy should look like:

```
gs_pinn
 ├─ data                          # dataset folder containing MTL data
 ├─ src
 │  ├─ experiments
 │  │  ├─ beltrami
 │  │  │  ├─ conf                # configuration files for experiments
 │  │  │  └─ trainer.py          # main training script
 │  │  ├─ burgers
 │  │  │  ├─ conf
 │  │  │  └─ trainer.py
 │  │  ├─ kovasznay
 │  │  │  ├─ conf
 │  │  │  └─ trainer.py
 │  │  └─ schrodinger
 │  │     ├─ conf
 │  │     └─ main.py
 │  └─ methods
 │     └─ weight_methods.py      # different MTL optimizers
 ├─ run.sh                       # script to run experiments
```

### Run experiments

The `run.sh` script acts as a template for launching experiments. It uses Hydra for configuration management.

Available parameters and options can be found in the `conf` directories under each experiment module.

To run an experiment:

```bash
bash run.sh
```

### MTL methods

We support the following MTL methods with a unified API. To run experiment with MTL method `X` simply run:
```bash
python trainer.py --method=X
```

| Method (code name) | Paper (notes) |
| :---: | :---: |
| Aligned-MTL (`alignedmtl`) | [Independent Component Alignment for Multi-Task Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf) |
| FAMO (`famo`) | [Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792.pdf) |
| Nash-MTL (`nashmtl`) | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf) |
| CAGrad (`cagrad`) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) |
| IMTL-G (`imtl`) | [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr) |
| MGDA (`mgda`) | [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) |
| DWA (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) |
| Uncertainty weighting (`uw-so`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| SAM-gs (`smgs`) | [Gradient Similarity Surgery in Multi-task Deep Learning](https://arxiv.org/abs/2506.06130) |
| Linear scalarization (`ls`) | - (equal weighting) |
| Scale-invariant baseline (`scaleinvls`) | - (see Nash-MTL paper for details) |

