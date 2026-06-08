import torch
from torch import Tensor
from typing import Optional
from pygidsim_torch.utils import define_device


def get_unique_directions(
        min_index: int = -10,
        max_index: int = 10,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate unique orientations within the given range.

    Removes:
    - inversion symmetry:
        (h,k,l) == (-h,-k,-l)
    - collinear duplicates:
        (1,1,1) == (2,2,2)

    Parameters
    ----------
    min_index : int
        The minimum index.
    max_index : int
        The maximum index.
    device : Optional[torch.device], optional
        The device to store the orientations.
        If None, use CUDA if available, otherwise CPU. Default is None.
    dtype: torch.dtype, optional
         The data type of the orientations. Default is torch.float32.

    Returns
    -------
    Tensor
        The orientations of shape (num_orientations, 3).
    """
    device = define_device(device)

    hkl = get_mi(
        min_index=min_index,
        max_index=max_index,
        device=device,
        dtype=torch.int64,
    )

    hkl = _reduce_directions(hkl)
    return hkl.to(dtype)


def get_mi(
        min_index: int,
        max_index: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.int64,
) -> Tensor:
    """
    Generate all the Laue indices within the given range excluding (0, 0, 0).
    No symmetry reduction is applied.

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
    Tensor
        The Laue indices of shape (num_reflections, 3).
    """
    device = define_device(device)

    r = torch.arange(min_index, max_index + 1, device=device)
    hkl = torch.cartesian_prod(r, r, r)

    # remove (0,0,0)
    hkl = hkl[(hkl != 0).any(dim=1)]

    return hkl.to(dtype)


def _reduce_directions(hkl: Tensor) -> Tensor:
    """
    Remove:
    - collinear multiples
    - inversion symmetry

    Examples
    --------
    (2,2,2) -> (1,1,1)

    (-1,-2,-3) -> (1,2,3)
    """

    hkl = hkl.clone()

    # reduce by gcd
    g = torch.gcd(
        torch.gcd(hkl[:, 0].abs(), hkl[:, 1].abs()),
        hkl[:, 2].abs(),
    )

    hkl = hkl // g[:, None]

    # canonical sign:
    # first nonzero component must be positive
    sign = torch.where(
        hkl[:, 0] != 0,
        torch.sign(hkl[:, 0]),
        torch.where(
            hkl[:, 1] != 0,
            torch.sign(hkl[:, 1]),
            torch.sign(hkl[:, 2]),
        )
    )

    hkl[sign < 0] *= -1

    return torch.unique(hkl, dim=0)
