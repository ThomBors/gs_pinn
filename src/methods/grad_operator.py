# usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
from typing import Optional, Sequence, Union, Tuple, Literal
from torch import Tensor
import numpy as np


def get_para_vector(network: torch.nn.Module) -> torch.Tensor:
    """
    Returns the parameter vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The parameter vector of the network.
    """
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed = par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec


def get_gradient_vector(
    network: torch.nn.Module, none_grad_mode: Literal["raise", "zero", "skip"] = "skip"
) -> torch.Tensor:
    """
    Returns the gradient vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.
        none_grad_mode (Literal['raise', 'zero', 'skip']): The mode to handle None gradients. default: 'skip'
            - 'raise': Raise an error when the gradient of a parameter is None.
            - 'zero': Replace the None gradient with a zero tensor.
            - 'skip': Skip the None gradient.
                        The None gradient usually occurs when part of the network is not trainable (e.g., fine-tuning)
            or the weight is not used to calculate the current loss (e.g., different parts of the network calculate different losses).
            If all of your losses are calculated using the same part of the network, you should set none_grad_mode to 'skip'.
            If your losses are calculated using different parts of the network, you should set none_grad_mode to 'zero' to ensure the gradients have the same shape.

    Returns:
        torch.Tensor: The gradient vector of the network.
    """
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "raise":
                    raise RuntimeError("None gradient detected.")
                elif none_grad_mode == "zero":
                    viewed = torch.zeros_like(par.data.view(-1))
                elif none_grad_mode == "skip":
                    continue
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                viewed = par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec


def apply_gradient_vector(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
    none_grad_mode: Literal["zero", "skip"] = "skip",
    zero_grad_mode: Literal["skip", "pad_zero", "pad_value"] = "pad_value",
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    This function requires the network contains the some gradient information in order to apply the gradient vector.
    If your network does not contain the gradient information, you should consider using `apply_gradient_vector_para_based` function.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
        none_grad_mode (Literal['zero', 'skip']): The mode to handle None gradients.
            You should set this parameter to the same value as the one used in `get_gradient_vector` method.
        zero_grad_mode (Literal['padding', 'skip']): How to set the value of the gradient if your `none_grad_mode` is "zero". default: 'skip'
            - 'skip': Skip the None gradient.
            - 'padding': Replace the None gradient with a zero tensor.
            - 'pad_value': Replace the None gradient using the value in the gradient.
            If you set `none_grad_mode` to 'zero', that means you padded zero to your `grad_vec` if the gradient of the parameter is None when getting the gradient vector.
            When you apply the gradient vector back to the network, the value in the `grad_vec` corresponding to the previous None gradient may not be zero due to the applied gradient operation.
                        Thus, you need to determine whether to recover the original None value, set it to zero, or set the value according to the value in `grad_vec`.
            If you are not sure what you are doing, it is safer to set it to 'pad_value'.

    """
    if none_grad_mode == "zero" and zero_grad_mode == "pad_value":
        apply_gradient_vector_para_based(network, grad_vec)
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "skip":
                    continue
                elif none_grad_mode == "zero":
                    start = start + par.data.view(-1).shape[0]
                    if zero_grad_mode == "pad_zero":
                        par.grad = torch.zeros_like(par.data)
                    elif zero_grad_mode == "skip":
                        continue
                    else:
                        raise ValueError(f"Invalid zero_grad_mode '{zero_grad_mode}'.")
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                end = start + par.data.view(-1).shape[0]
                par.grad.data = grad_vec[start:end].view(par.data.shape)
                start = end


def apply_gradient_vector_para_based(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    Please only use this function when you are sure that the length of `grad_vec` is the same of your network's parameters.
    This happens when you use `get_gradient_vector` with `none_grad_mode` set to 'zero'.
    Or, the 'none_grad_mode' is 'skip' but all of the parameters in your network is involved in the loss calculation.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.grad = grad_vec[start:end].view(par.data.shape)
            start = end


def apply_para_vector(network: torch.nn.Module, para_vec: torch.Tensor) -> None:
    """
    Applies a parameter vector to the network's parameters.

    Args:
        network (torch.nn.Module): The network to apply the parameter vector to.
        para_vec (torch.Tensor): The parameter vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.data = para_vec[start:end].view(par.data.shape)
            start = end


def get_cos_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine angle between two vectors.

    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The cosine angle between the two vectors.
    """
    with torch.no_grad():
        return torch.dot(vector1, vector2) / vector1.norm() / vector2.norm()


def unit_vector(vector: torch.Tensor, warn_zero: bool = False) -> torch.Tensor:
    """
    Compute the unit vector of a given tensor.

    Parameters:
        vector (torch.Tensor): The input tensor.
        warn_zero (bool): Whether to print a warning when the input tensor is zero. default: False

    Returns:
        torch.Tensor: The unit vector of the input tensor.
    """
    with torch.no_grad():
        if vector.norm() == 0:
            if warn_zero:
                print("Detected zero vector when doing normalization.")
            return torch.zeros_like(vector)
        else:
            return vector / vector.norm()


def transfer_coef_double(
    weights: torch.tensor,
    unit_vec_1: torch.tensor,
    unit_vec_2: torch.tensor,
    or_unit_vec_1: torch.tensor,
    or_unit_vec_2: torch.tensor,
) -> tuple:
    """
    Transfer the angle weights to a length coefficient for ConFIG method with two vectors.

    This function will return a coefficient matrix [c_1,c_2] so that
    $$
    \frac{(c_1 o_1+c_2 o_2)\dot \mathbf{v}_1}{(c_1 o_1+c_2 o_2) \dot \mathbf{v}_2}
    =\frac{w_1}{w_2}
    $$
    where w_1 and w_2 are the angle weights,
    v_1 and v_2 are unit vectors of the two gradients,
    and o_1 and o_2 are the unit orthogonal components of the two gradients.

    Args:
        weights (torch.tensor): Angle weights for the two vectors. It should have shape (2,).
        unit_vecs (torch.tensor): Unit vectors of the orthogonal components. It should have shape (2,k) where k is the length of the vector.

    Returns:
        tuple: The coefficients for the two vectors.

    Raises:
        ValueError: If the number of weights or unit vectors is not equal to 2.
    """

    return (
        torch.dot(or_unit_vec_2, unit_vec_1)
        / (weights[0] / weights[1] * torch.dot(or_unit_vec_1, unit_vec_2)),
        1,
    )


def estimate_conflict(gradients: torch.Tensor) -> torch.Tensor:
    """
    Estimates the degree of conflict of gradients.

    Args:
        gradients (torch.Tensor): A tensor containing gradients.

    Returns:
        torch.Tensor: A tensor consistent of the dot products between the sum of gradients and each sub-gradient.
    """
    direct_sum = unit_vector(gradients.sum(dim=0))
    unit_grads = gradients / torch.norm(gradients, dim=1).view(-1, 1)
    return unit_grads @ direct_sum


def has_zero(lists: Sequence) -> bool:
    """
    Check if any element in the list is zero.

    Args:
        lists (Sequence): A list of elements.

    Returns:
        bool: True if any element is zero, False otherwise.
    """
    for i in lists:
        if i == 0:
            return True
    return False


class OrderedSliceSelector:
    """
    Selects a slice of the source sequence in order.
    Usually used for selecting loss functions/gradients/losses in momentum-based method if you want to update more tha one gradient in a single iteration.

    """

    def __init__(self):
        self.start_index = 0

    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        """
        Selects a slice of the source sequence in order.

        Args:
            n (int): The length of the target slice.
            source_sequence (Sequence): The source sequence to select from.

        Returns:
            Tuple[Sequence,Union[float,Sequence]]: A tuple containing the indexes of the selected slice and the selected slice.
        """
        if n > len(source_sequence):
            raise ValueError(
                "n must be less than or equal to the length of the source sequence"
            )
        end_index = self.start_index + n
        if end_index > len(source_sequence) - 1:
            new_start = end_index - len(source_sequence)
            indexes = list(range(self.start_index, len(source_sequence))) + list(
                range(0, new_start)
            )
            self.start_index = new_start
        else:
            indexes = list(range(self.start_index, end_index))
            self.start_index = end_index
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]


class RandomSliceSelector:
    """
    Selects a slice of the source sequence randomly.
    Usually used for selecting loss functions/gradients/losses in momentum-based method if you want to update more tha one gradient in a single iteration.
    """

    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        """
        Selects a slice of the source sequence randomly.

        Args:
            n (int): The length of the target slice.
            source_sequence (Sequence): The source sequence to select from.

        Returns:
            Tuple[Sequence,Union[float,Sequence]]: A tuple containing the indexes of the selected slice and the selected slice.
        """
        assert n <= len(
            source_sequence
        ), "n can not be larger than or equal to the length of the source sequence"
        indexes = np.random.choice(len(source_sequence), n, replace=False)
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]


class WeightModel:
    """
    Base class for weight models.
    """

    def __init__(self):
        pass

    def get_weights(
        self,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ):
        """_summary_

        Args:
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method must be implemented by the subclass.")


class EqualWeight(WeightModel):
    """
    A weight model that assigns equal weights to all gradients.
    """

    def __init__(self):
        super().__init__()

    def get_weights(
        self,
        gradients: torch.Tensor,
        losses: Optional[Sequence] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """
        Calculate the weights for the given gradients.

        Args:
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            torch.Tensor: A tensor of equal weights for all gradients.

        Raises:
            ValueError: If gradients is None.
        """
        assert gradients is not None, "The EqualWeight model requires gradients"
        return torch.ones(gradients.shape[0], device=device)


class LengthModel:
    """
    The base class for length model.

    Methods:
        get_length: Calculates the length based on the given parameters.
        rescale_length: Rescales the length of the target vector based on the given parameters.
    """

    def __init__(self):
        pass

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> Union[torch.Tensor, float]:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def rescale_length(
        self,
        target_vector: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Rescales the length of the target vector based on the given parameters.
        It calls the get_length method to calculate the length and then rescales the target vector.

        Args:
            target_vector (torch.Tensor): The final update gradient vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses.

        Returns:
            torch.Tensor: The rescaled target vector.
        """
        unit_target_vector = unit_vector(target_vector)
        return (
            self.get_length(
                target_vector=target_vector,
                unit_target_vector=unit_target_vector,
                gradients=gradients,
                losses=losses,
            )
            * unit_target_vector
        )


class ProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector:

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_i|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        assert (
            gradients is not None
        ), "The ProjectionLength model requires gradients information."
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        return torch.sum(
            torch.stack([torch.dot(grad_i, unit_target_vector) for grad_i in gradients])
        )


class _FlexibleTrackProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    The length each loss-specific gradient will be rescaled to the same length as the tracked value before projection.
    The tracked value is calculated by the _tracked_value method.
    """

    def __init__(self):
        super().__init__()

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        assert (
            gradients is not None
        ), "The ProjectLength model requires gradients information."
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        norms = torch.norm(gradients, dim=1)
        tracked_value = self._tracked_value(norms)
        return sum(
            [
                torch.dot(grad_i / norm_i, unit_target_vector) * tracked_value
                for grad_i, norm_i in zip(gradients, norms)
            ]
        )

    def _tracked_value(self, grad_norms: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method must be implemented by the subclass.")


class TrackMinimum(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the minimum gradient before projection, i.e., the minimum gradient will be the same length as the target vector.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_{min}|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.min()


class TrackMaximum(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the maximum gradient before projection, i.e., the maximum gradient will be the same length as the target vector.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_{max}|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.max()


class TrackHarmonicAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the harmonic average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{harm}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$

    where

    $$
    \overline{|\mathbf{g}|}_{harm}=\frac{m}{\sum_{i=1}^m \frac{1}{|\mathbf{g}_i|}}
    $$

    The harmonic average can be used to avoid the influence of the large gradients.
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.shape[0] / torch.sum(1 / grad_norms)


class TrackArithmeticAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the arithmetic average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{arith}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$

    where

    $$
    \overline{|\mathbf{g}|}_{arith}=\frac{1}{m}\sum_{i=1}^m |\mathbf{g}_i|
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.mean()


class TrackGeometricAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the geometric average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{geom}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$

    where

    $$
    \overline{|\mathbf{g}|}_{geom}=\left(\prod_{i=1}^m |\mathbf{g}_i|\right)^{\frac{1}{m}}
    $$

    The geometric average can be used to avoid the influence of the large gradients.
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return torch.prod(grad_norms) ** (1 / grad_norms.shape[0])


class TrackSpecific(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the specific gradient before projection.
    E.g., if the track_id is 2, then all the gradients will be rescaled to the same length as the third gradient before projection.

    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{track_id}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$

    """

    def __init__(self, track_id: int):
        super().__init__()
        self.track_id = track_id

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms[self.track_id]


def ConFIG_update_double(
    grad_1: torch.Tensor,
    grad_2: torch.Tensor,
    weight_model: WeightModel = EqualWeight(),
    length_model: LengthModel = ProjectionLength(),
    losses: Optional[Sequence] = None,
) -> torch.Tensor:
    """
    ConFIG update for two gradients where no inverse calculation is needed.

    Args:
        grad_1 (torch.Tensor): The first gradient.
        grad_2 (torch.Tensor): The second gradient.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        losses (Optional[Sequence], optional): The losses associated with the gradients.
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update_double
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in [loss_fn1, loss_fn2]:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update_double(grads) # calculate the conflict-free direction
            apply_gradient_vector(network,g_config) # set the conflict-free direction to the network
            optimizer.step()
        ```

    """
    with torch.no_grad():
        norm_1 = grad_1.norm()
        norm_2 = grad_2.norm()
        unit_1 = grad_1 / norm_1
        unit_2 = grad_2 / norm_2
        cos_angle = get_cos_similarity(grad_1, grad_2)
        or_2 = grad_1 - norm_1 * cos_angle * unit_2
        or_1 = grad_2 - norm_2 * cos_angle * unit_1
        unit_or1 = unit_vector(or_1)
        unit_or2 = unit_vector(or_2)
        coef_1, coef_2 = transfer_coef_double(
            weight_model.get_weights(
                gradients=torch.stack([grad_1, grad_2]),
                losses=losses,
                device=grad_1.device,
            ),
            unit_1,
            unit_2,
            unit_or1,
            unit_or2,
        )
        best_direction = coef_1 * unit_or1 + coef_2 * unit_or2
        return length_model.rescale_length(
            target_vector=best_direction,
            gradients=torch.stack([grad_1, grad_2]),
            losses=losses,
        )


def ConFIG_update(
    grads: Union[torch.Tensor, Sequence[torch.Tensor]],
    weight_model: WeightModel = EqualWeight(),
    length_model: LengthModel = ProjectionLength(),
    use_least_square: bool = True,
    losses: Optional[Sequence] = None,
) -> torch.Tensor:
    """
    Performs the standard ConFIG update step.

    Args:
        grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
            It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction.
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.
        losses (Optional[Sequence], optional): The losses associated with the gradients.
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update(grads) # calculate the conflict-free direction
            apply_gradient_vector(network,g_config) # set the conflict-free direction to the network
            optimizer.step()
        ```
    """
    if not isinstance(grads, torch.Tensor):
        grads = torch.stack(grads)
    with torch.no_grad():
        weights = weight_model.get_weights(
            gradients=grads, losses=losses, device=grads.device
        )
        units = torch.nan_to_num((grads / (grads.norm(dim=1)).unsqueeze(1)), 0)
        if use_least_square:
            best_direction = torch.linalg.lstsq(units, weights).solution
        else:
            best_direction = torch.linalg.pinv(units) @ weights
        return length_model.rescale_length(
            target_vector=best_direction,
            gradients=grads,
            losses=losses,
        )


class GradientOperator:
    """
    A base class that represents a gradient operator.

    Methods:
        calculate_gradient: Calculates the gradient based on the given gradients and losses.
        update_gradient: Updates the gradient of the network based on the calculated gradient.

    """

    def __init__(self):
        pass

    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient based on the given gradients and losses.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients.
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("calculate_gradient method must be implemented")

    def update_gradient(
        self,
        network: torch.nn.Module,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> None:
        """
        Calculate the gradient and apply the gradient to the network.

        Args:
            network (torch.nn.Module): The target network.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients.
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            None

        """
        apply_gradient_vector(network, self.calculate_gradient(grads, losses))


class ConFIGOperator(GradientOperator):
    """
    Operator for the ConFIG algorithm.

    Args:
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        allow_simplified_model (bool, optional): Whether to allow simplified model for calculating the gradient.
            If set to True, will use simplified form of ConFIG method when there are only two losses (ConFIG_update_double). Defaults to True.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction.
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIGOperator
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        operator=ConFIGOperator() # initialize operator
        for input_i in dataset:
            grads=[]
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network))
            g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
            apply_gradient_vector(network,g_config) # or simply use `operator.update_gradient(network,grads)` to calculate and set the conflict-free direction to the network
            optimizer.step()
        ```

    """

    def __init__(
        self,
        weight_model: WeightModel = EqualWeight(),
        length_model: LengthModel = ProjectionLength(),
        allow_simplified_model: bool = True,
        use_least_square: bool = True,
    ):
        super().__init__()
        self.weight_model = weight_model
        self.length_model = length_model
        self.allow_simplified_model = allow_simplified_model
        self.use_least_square = use_least_square

    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the ConFIG algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients.
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        if grads.shape[0] == 2 and self.allow_simplified_model:
            return ConFIG_update_double(
                grads[0],
                grads[1],
                weight_model=self.weight_model,
                length_model=self.length_model,
                losses=losses,
            )
        else:
            return ConFIG_update(
                grads,
                weight_model=self.weight_model,
                length_model=self.length_model,
                use_least_square=self.use_least_square,
                losses=losses,
            )


class PCGradOperator(GradientOperator):
    """
    PCGradOperator class represents a gradient operator for PCGrad algorithm.

    @inproceedings{yu2020gradient,
    title={Gradient surgery for multi-task learning},
    author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
    booktitle={34th International Conference on Neural Information Processing Systems},
    year={2020},
    url={https://dl.acm.org/doi/abs/10.5555/3495724.3496213}
    }

    """

    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the PCGrad algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using PCGrad method.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        with torch.no_grad():
            grads_pc = torch.clone(grads)
            length = grads.shape[0]
            for i in range(length):
                for j in range(length):
                    if j != i:
                        dot = grads_pc[i].dot(grads[j])
                        if dot < 0:
                            grads_pc[i] -= dot * grads[j] / ((grads[j].norm()) ** 2)
            return torch.sum(grads_pc, dim=0)


class IMTLGOperator(GradientOperator):
    """
    PCGradOperator class represents a gradient operator for IMTL-G algorithm.

    @inproceedings{
    liu2021towards,
    title={Towards Impartial Multi-task Learning},
    author={Liyang Liu and Yi Li and Zhanghui Kuang and Jing-Hao Xue and Yimin Chen and Wenming Yang and Qingmin Liao and Wayne Zhang},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=IMPnRXEWpvr}
    }

    """

    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the IMTL-G algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using IMTL-G method.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        with torch.no_grad():
            ut_norm = grads / grads.norm(dim=1).unsqueeze(1)
            ut_norm = torch.nan_to_num(ut_norm, 0)
            ut = torch.stack(
                [ut_norm[0] - ut_norm[i + 1] for i in range(grads.shape[0] - 1)], dim=0
            ).T
            d = torch.stack(
                [grads[0] - grads[i + 1] for i in range(grads.shape[0] - 1)], dim=0
            )
            at = grads[0] @ ut @ torch.linalg.pinv(d @ ut)
            return (1 - torch.sum(at)) * grads[0] + torch.sum(
                at.unsqueeze(1) * grads[1:], dim=0
            )
