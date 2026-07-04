import torch
from typing import List, Union
from abc import abstractmethod
import torch.nn.functional as F
import numpy as np


class ConflictsMetrics:
    def __init__(self, partitions=None):
        # e.g. ['train', 'val', 'test']
        if partitions is None:
            partitions = ["train", "val", "test"]
        self.partitions = partitions
        # { partition: { metric_name: [(epoch, batch, value), ...] } }
        self.data = {p: {} for p in self.partitions}

    def add(self, partition, epoch, batch, metrics):
        """
        Add metrics for one partition, epoch, and batch.

        :param partition: str, e.g. 'train'/'val'/'test'
        :param epoch: int
        :param batch: int
        :param metrics: dict or list of (metric_name, value)
        """
        if partition not in self.data:
            raise ValueError(f"Unknown partition: {partition}")

        # normalize input
        if isinstance(metrics, dict):
            items = metrics.items()
        else:
            items = metrics

        for name, values in items:
            # convert tensor -> numpy -> float if needed
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            if isinstance(values, np.ndarray) and values.size == 1:
                values = values.item()

            if name not in self.data[partition]:
                self.data[partition][name] = []
            self.data[partition][name].append((epoch, batch, values))

    def get_metrics_partition(self, partition, name):
        """
        Get all recorded values for one metric in one partition.
        Returns list of (epoch, batch, value).
        """
        return self.data.get(partition, {}).get(name, [])

    def aggregate_epoch(self, partition, name, agg="mean"):
        """
        Aggregate batch-level metrics per epoch.

        :param partition: str
        :param name: metric name
        :param agg: 'mean' | 'max' | 'min'
        :return: dict {epoch: aggregated_value}
        """
        values = self.get_metrics_partition(partition, name)
        if not values:
            return {}

        epoch_dict = {}
        for epoch, batch, val in values:
            if epoch not in epoch_dict:
                epoch_dict[epoch] = []
            epoch_dict[epoch].append(val)

        if agg == "mean":
            return {e: float(np.mean(v)) for e, v in epoch_dict.items()}
        elif agg == "max":
            return {e: float(np.max(v)) for e, v in epoch_dict.items()}
        elif agg == "min":
            return {e: float(np.min(v)) for e, v in epoch_dict.items()}
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    def get_metrics(self):
        """Return full nested dict of all data."""
        return self.data

    def summary(self):
        """Print summary of stored metrics."""
        for p in self.partitions:
            print(f"Partition: {p}")
            for name, values in self.data[p].items():
                epochs = sorted(set(e for e, _, _ in values))
                total_batches = len(values)
                print(
                    f"  Metric: {name}, epochs={epochs}, total_batches={total_batches}"
                )


class MetricMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def compute_conflicts(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor] = None,
        **kwargs,
    ):
        """Compute gradient-based conflict metrics."""
        pass

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.compute_conflicts(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )


class GradientConflict(MetricMethod):
    """Gradient Conflict metrics (magnitude similarity and angle)."""

    def __init__(self, n_tasks: int, device: torch.device, max_norm: float = 1.0):
        super().__init__(n_tasks=n_tasks, device=device, max_norm=max_norm)

    @staticmethod
    def _grad2vec(grads_task, grads, grad_dims, task, shared_parameters):
        grads[:, task].fill_(0.0)
        cnt_task = 0
        for cnt_param, p in enumerate(shared_parameters):
            if p.requires_grad:
                g = grads_task[cnt_task]
                cnt_task += 1
                if g is None:
                    g = torch.zeros_like(p).view(-1)
            else:
                g = torch.zeros_like(p).view(-1)

            beg = 0 if cnt_param == 0 else sum(grad_dims[:cnt_param])
            en = sum(grad_dims[: cnt_param + 1])
            grads[beg:en, task].copy_(g.view(-1))

    def compute_conflicts(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        metric: str = "all",
        **kwargs,
    ):
        grad_dims = [p.numel() for p in shared_parameters]
        grads = torch.zeros(sum(grad_dims), self.n_tasks, device=self.device)

        # Compute gradients per task
        for i in range(self.n_tasks):
            retain = i < self.n_tasks - 1
            grads_task = torch.autograd.grad(
                losses[i],
                shared_parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            self._grad2vec(grads_task, grads, grad_dims, i, shared_parameters)

        if metric == "magnitude":
            return {"magnitude_similarity": self._magnitude_similarity(grads).item()}
        elif metric == "angle":
            conf, ang = self._conflict_grads(grads)
            return {"average_angle": ang, "conflicting_angle": conf}
        elif metric == "all":
            conf, ang = self._conflict_grads(grads)
            return {
                "magnitude_similarity": self._magnitude_similarity(grads).item(),
                "average_angle": ang,
                "conflicting_angle": conf,
            }
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    def _magnitude_similarity(self, grads: torch.Tensor) -> torch.Tensor:
        """Mean pairwise similarity of gradient magnitudes."""
        norms = grads.norm(dim=0)  # (n_tasks,)
        norm_matrix = norms.unsqueeze(1)  # (n_tasks, 1)

        sims = (2 * (norm_matrix @ norm_matrix.T)) / (
            norm_matrix**2 + norm_matrix.T**2 + 1e-12
        )
        mask = torch.triu(torch.ones_like(sims, dtype=torch.bool), diagonal=1)
        return sims[mask].mean()

    def _conflict_grads(self, grads: torch.Tensor, any_conflict: bool = True):
        n_tasks = grads.shape[1]
        cos_sims = []
        for i in range(n_tasks):
            gi = grads[:, i]
            for j in range(i + 1, n_tasks):
                gj = grads[:, j]
                cos_sim = torch.dot(gi, gj) / (gi.norm() * gj.norm() + 1e-12)
                cos_sims.append(cos_sim)
        if not cos_sims:  # only one task return False,
            torch.tensor(1.0, device=grads.device)
        cos_sims = torch.stack(cos_sims)
        conflicts = cos_sims < 0
        conflict = conflicts.any() if any_conflict else conflicts.all()
        avg_cos_sim = cos_sims.mean()
        return conflict.detach().item(), avg_cos_sim.detach()


class MetricMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        assert method in METHODS, f"Unknown method {method}."
        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def compute_conflicts(self, losses, **kwargs):
        return self.method.compute_conflicts(losses, **kwargs)

    def __call__(self, losses, **kwargs):
        return self.compute_conflicts(losses, **kwargs)


# Registry of available conflict computation methods
METHODS = {
    "gc": GradientConflict,
}
