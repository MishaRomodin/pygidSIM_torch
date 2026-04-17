import torch
from itertools import product
from typing import List, Tuple


def get_unique_directions(
        min_index: int = -10,
        max_index: int = 10,
        *,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Get all the (normalized) orientations within the given range.

    Parameters
    ----------
    min_index : int
        The minimum index.
    max_index : int
        The maximum index.
    device : torch.device
    dtype: torch.dtype

    Returns
    -------
    torch.Tensor
        The orientations of shape (num_orientations, 3).
    """
    orient = get_mi(min_index, max_index, device=device, dtype=dtype)
    orient = orient[torch.argsort(torch.linalg.norm(orient, dim=1))]
    return torch.nn.functional.normalize(orient, dim=1).unique(dim=0)


def get_mi(
        min_index: int, max_index: int,
        *,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Get all the Laue indices within the given range excluding (0, 0, 0).

    Parameters
    ----------
    min_index : int
        The minimum index.
    max_index : int
        The maximum index.
    device : torch.device, optional
    dtype: torch.dtype, optional

    Returns
    -------
    torch.Tensor
        The Laue indices of shape (num_reflections, 3).
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.tensor(
        _get_all_nonzero_combinations(min_index, max_index),
        device=device, dtype=dtype,
    )


def _get_all_nonzero_combinations(min_index: int, max_index: int) -> List[Tuple]:
    return list(filter(any, product(list(range(min_index, max_index + 1)), repeat=3)))
