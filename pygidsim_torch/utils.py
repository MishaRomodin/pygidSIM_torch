import torch
from math import pi


def convert_angles_to_degrees(
        lattice_params: torch.Tensor,
        inplace: bool = False,
):
    """
    Convert the angles in the lattice parameters to degrees.

    Parameters
    ----------
    lattice_params : torch.Tensor
        The lattice parameters. Shape (B, 6). Angles in radians.
    inplace : bool
        If True, convert the angles in-place. Defaults to False.

    Returns
    -------
    lattice_params : torch.Tensor
        The lattice parameters. Shape (B, 6). Angles in degrees.
    """
    if not inplace:
        lattice_params = lattice_params.clone()
    lattice_params[..., 3:6] *= 180 / pi
    return lattice_params


def convert_angles_to_radians(
        lattice_params: torch.Tensor,
        inplace: bool = False,
):
    """
    Convert the angles in the lattice parameters to radians.

    Parameters
    ----------
    lattice_params : torch.Tensor
        The lattice parameters. A tensor of shape (B, 6). Angles in degrees.
    inplace : bool
        If True, convert the angles in-place. Defaults to False.

    Returns
    -------
    lattice_params : torch.Tensor
        The lattice parameters. Shape (B, 6). Angles in radians.
    """
    if not inplace:
        lattice_params = lattice_params.clone()
    lattice_params[..., 3:6] *= pi / 180
    return lattice_params


def calculate_volume(
        lat_par: torch.Tensor,  # (B, 6)
        min_unit_volume: float = 0.1,
        deg: bool = True,
) -> torch.Tensor:
    """Calculate the volume of a unit cell."""
    if deg:
        lat_par = convert_angles_to_radians(lat_par, inplace=False)
    a, b, c, alpha, beta, gamma = lat_par.unbind(dim=-1)

    unit_volume = calculate_unit_volume(lat_par)

    unit_volume[torch.isnan(unit_volume)] = 0
    unit_volume[unit_volume < min_unit_volume] = 0

    return a * b * c * unit_volume


def calculate_unit_volume(lat_par: torch.Tensor,  # (B, 6)
                          ):
    """
    Calculate unit volume of a unit cell.

    Parameters
    ----------
    lat_par : torch.Tensor
        The lattice parameters. Shape (B, 6). Angles in radians.
    """
    a, b, c, alpha, beta, gamma = lat_par.unbind(dim=-1)
    return torch.sqrt(
        1.0
        + 2.0 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma)
        - torch.cos(alpha) ** 2
        - torch.cos(beta) ** 2
        - torch.cos(gamma) ** 2,
    )
