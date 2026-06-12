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
                 device: Optional[torch.device] = None
                 ):
        self.device = define_device(device)
        self.lat_par = lat_par.to(
            device=self.device,
            dtype=torch.float32
        )
        if self.lat_par.ndim == 1:
            self.lattice_params = self.lat_par.unsqueeze(0)
        # self.spgr = spgr.to(device)
        self.volume = self._calc_volume(vol_min=10)
        self.valid = self.volume > 0

    def _calc_volume(self, vol_min=10):
        """Calculate the volume of the unit cell."""
        unit_volume = calculate_volume(self.lat_par, deg=True)
        unit_volume[unit_volume < vol_min] = 0
        return unit_volume
