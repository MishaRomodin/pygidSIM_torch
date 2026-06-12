import warnings
from typing import Optional
import torch
from torch import Tensor

from .utils import define_device, calculate_volume


class Crystal:
    """
    A class to represent the crystal structure.

    Parameters
    ----------
    lat_par : Tensor
        Lattice parameters of shape (B, 6). Columns: a, b, c, alpha, beta, gamma.
    deg : bool
        Whether the angles in lat_par are in degrees.
        If True, expects the angles to be in degrees and returns in degrees.
        If False, expects the angles to be in radians and returns in radians. Default is True.
    vol_min : float, optional
        Minimum volume of the unit cell. Default is 10.0 Å^3.
        Unit cells with volume below this threshold will be considered invalid.
    device : Optional[torch.device]
        Device on which the tensor is stored. If None, use CUDA if available, otherwise CPU.
    """

    def __init__(self,
                 lat_par: Tensor,
                 # spgr: Tensor[int],
                 # atoms: Optional[Tensor] = None,
                 # atom_positions: Optional[Tensor] = None,
                 # occ: Optional[Tensor] = None,
                 # scale: Optional[Tensor] = None, # (B, 3)
                 deg: bool = True,
                 vol_min: float = 10.0,
                 device: Optional[torch.device] = None
                 ):
        self.deg = deg
        self.device = define_device(device)
        self._lat_par_all = lat_par.to(
            device=self.device,
            dtype=torch.float32
        )
        if self.lat_par.ndim == 1:
            self.lattice_params = self.lat_par.unsqueeze(0)
        # self.spgr = spgr.to(device)
        self._volumes_all = self._calc_volume(vol_min=vol_min)
        self._valid = self._volumes_all > 0
        if not self._valid.all():
            warnings.warn('Warning: Some of the lattice parameters are not valid.')

    @property
    def lat_par(self) -> Tensor:
        return self._lat_par_all[self._valid]

    @property
    def invalid_lat_par(self) -> Tensor:
        return self._lat_par_all[~self._valid]

    @property
    def volume(self) -> Tensor:
        return self._volumes_all[self._valid]

    def _calc_volume(self, vol_min: float):
        """Calculate the volume of the unit cell."""
        unit_volume = calculate_volume(self._lat_par_all, deg=self.deg)
        unit_volume[unit_volume < vol_min] = 0
        return unit_volume
