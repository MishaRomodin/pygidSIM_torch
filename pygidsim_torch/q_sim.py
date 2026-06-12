from typing import Union, Optional
from math import pi
import torch
from torch import Tensor
import torch.nn.functional as F

from .directions import get_unique_directions
from .utils import convert_angles_to_radians


class Qpos:
    """
    A class to calculate the q positions in 3d reciprocal space.

    Attributes
    ----------
    lat_par : Tensor
        Lattice parameters. Shape (B, 6). Angles in grad.
    deg : bool
        If True, angles in lat_par are in degrees and will be converted to radians for calculations.
    _rec : Tensor
        Reciprocal vectors. Shape (B, 3, 3).

    Methods
    -------
    calculate_q3d(mi):
        Calculates the q vectors in 3d reciprocal space.
    rotate_vect(orientation, baz):
        Rotate the crystal.
    """

    def __init__(self, lat_par: Tensor, deg: bool = True):
        self.lat_par = lat_par
        if deg:
            self.lat_par = convert_angles_to_radians(self.lat_par)
        self.deg = deg
        self._B = len(lat_par)
        self.device = lat_par.device
        self.dtype = lat_par.dtype

        self._rec = self._calculate_rec()

    @property
    def rec(self) -> Tensor:
        """Return reciprocal lattice vectors."""
        return self._rec

    def calculate_q3d(self, mi: Tensor):
        """
        Calculate scattering vectors.

        Parameters
        ----------
        mi : Tensor
            The Miller indices. Shape (num_reflections, 3).

        Returns
        -------
        Tensor
            The scattering vectors. Shape (B, num_reflections, 3).
        """
        # if len(mi) == 0:
        #     return torch.tensor([], dtype=self.dtype).reshape(self._B, 0, 3)
        return torch.matmul(mi, self._rec)

    def rotate_vect(self,
                    q_3d: Tensor,
                    orientation: Optional[Union[Tensor, str]] = None):
        """
        Rotate crystal

        Parameters
        ----------
        q_3d : Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).
        orientation : Union[Tensor, str]
            Orientation of the crystal growth:
            - Tensor: Specific orientation vector. Tensor of shape (B, 3) or (3,) in case of same orientation
            for all samples.
            - 'random': Random orientation for each calculation (2D pattern).
            Default is [001] for all samples.

        Returns
        -------
        q_3d : Tensor
            Rotated peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).
        """
        if q_3d.shape[0] != self._B or q_3d.shape[2] != 3:
            raise ValueError(
                "q_3d must have shape (B, num_reflections, 3), where B should be the same as the number "
                "of lattice parameter sets."
            )
        if orientation is None:
            return q_3d
        elif isinstance(orientation, str):
            if orientation != 'random':
                raise ValueError("orientation is not correct - use Tensor with size (3,) or 'random'")
            directions = get_unique_directions(-10, 10, device=self.device)
            perm_idx = torch.multinomial(
                torch.ones(directions.shape[0]),
                q_3d.shape[0],
                replacement=True
            )
            orientation = directions[perm_idx]  # choose possible orientations for the samples
        else:
            if orientation.ndim == 1:
                orientation = orientation.unsqueeze(0).expand(self._B, -1)
            assert orientation.shape[1] == 3, \
                "orientation must have shape (B, 3) or (3,) in case of same orientation for all samples."

        orientation = F.normalize(orientation, dim=-1)
        assert orientation.shape == (self._B, 3), "finally, the orientation must have shape (B, 3)."

        R = self._rotation_matrix(orientation=orientation.to(self.device))
        return torch.matmul(q_3d, R)

    def _rotation_matrix(
            self,
            orientation: Optional[Tensor] = None,
            *,
            baz: Optional[Tensor] = None,
    ):
        """
        Rotate crystal

        Parameters
        ----------
        orientation : Optional[Tensor]
            Crystallographic orientations.orientation of the crystal growth.
            Tensor of shape (B, 3) or (3,) in case of same orientation for all samples.
            Default is [001] for all samples.
        baz : Optional[Tensor]
            Shape (3,) or (B, 3). Basis vector for the default orientation.
            Tensor of shape (B, 3) or (3,) in case of same orientation for all samples.
            Default is [001] for all samples.

        Return
        -------
        R : Tensor
            Rotation matrix. Shape (B, 3, 3).
        """
        baz = self._norm_orientations(baz)  # (B, 3)
        orientation = self._norm_orientations(orientation)  # (B, 3)

        same_mask = (orientation == baz).all(dim=-1)  # (B,)

        # Compute oriented vectors in the reciprocal basis for all batch entries
        orient = torch.matmul(orientation.unsqueeze(1), self._rec).squeeze(1)  # (B, 3)

        v1 = self._norm_orientations(orient)  # (B, 3)
        v2 = baz  # (B, 3)

        n_raw = torch.cross(v1, v2, dim=-1)  # (B, 3)
        zero_mask = (n_raw.norm(dim=-1) < 1e-8)
        n_raw = torch.where(zero_mask.unsqueeze(-1), baz, n_raw)

        n = n_raw / n_raw.norm(dim=-1, keepdim=True)

        cos_phi = torch.sum(v1 * v2, dim=-1)  # (B,)
        sin_phi = torch.sqrt(torch.clamp(1 - cos_phi ** 2, min=0.0))  # (B,)

        nx, ny, nz = n.unbind(dim=-1)

        one_minus_cos = 1 - cos_phi

        a1 = torch.stack(
            [
                nx * nx * one_minus_cos + cos_phi,
                nx * ny * one_minus_cos + nz * sin_phi,
                nx * nz * one_minus_cos - ny * sin_phi
            ], dim=-1
        )

        a2 = torch.stack(
            [
                nx * ny * one_minus_cos - nz * sin_phi,
                ny * ny * one_minus_cos + cos_phi,
                ny * nz * one_minus_cos + nx * sin_phi
            ], dim=-1
        )

        a3 = torch.stack(
            [
                nx * nz * one_minus_cos + ny * sin_phi,
                ny * nz * one_minus_cos - nx * sin_phi,
                nz * nz * one_minus_cos + cos_phi
            ], dim=-1
        )

        R = torch.stack([a1, a2, a3], dim=1)  # (B, 3, 3)

        # Where orientation equals baz, use identity rotation
        I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self._B, -1, -1)
        R = torch.where(same_mask.unsqueeze(-1).unsqueeze(-1), I, R)

        return R

    def _norm_orientations(self, orient: Optional[Tensor]) -> Tensor:
        """
        Normalize orientations.

        Parameters
        ----------
        orient : Optional[Tensor]
            Orientation tensor of shape (B, 3) or (3,) in case of same orientation for all samples.
            If None, default orientation [001] is used.

        Returns
        -------
        Tensor
            Normalized orientation tensor of shape (B, 3).
        """
        if orient is None:
            orient = torch.tensor([0., 0., 1.], dtype=self.dtype, device=self.device)
        if orient.ndim == 1:
            orient = orient.unsqueeze(0).expand(self._B, -1)
        assert orient.shape[0] == self._B, "Orientation tensor must have the same batch size as the lattice parameters."
        orient = F.normalize(orient, dim=-1)
        return orient

    def _calculate_rec(self) -> Tensor:
        """Calculate reciprocal lattice vectors."""
        a1, a2, a3 = self._lattice_vectors_from_parameters()
        return self._calc_reciprocal_vectors(a1, a2, a3)

    def _lattice_vectors_from_parameters(self):
        """
        Calculate lattice vectors corresponding to lattice parameters.

        Returns
        -------
        a1, a2, a3: (B, 3)
            Unit cell vectors. Invalid rows are filled with NaN.
        """
        a, b, c, alpha, beta, gamma = self.lat_par.unbind(dim=-1)

        a1 = torch.stack(
            [
                a,
                torch.zeros_like(a),
                torch.zeros_like(a)
            ], dim=-1
        )

        a2 = torch.stack(
            [
                b * torch.cos(gamma),
                b * torch.sin(gamma),
                torch.zeros_like(b)
            ], dim=-1
        )

        a31 = c * torch.cos(beta)
        a32 = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        a33 = torch.sqrt(torch.clamp(c ** 2 - a31 ** 2 - a32 ** 2, min=0.0))

        a3 = torch.stack([a31, a32, a33], dim=-1)

        return a1, a2, a3

    @staticmethod
    def _calc_reciprocal_vectors(
            a1: Tensor,  # (B, 3)
            a2: Tensor,  # (B, 3)
            a3: Tensor,  # (B, 3)
    ):
        """
        Calculate the reciprocal unit cell vectors from the real-space unit cell vectors.

        Parameters
        ----------
        a1, a2, a3 : (B, 3)
            Unit cell vectors.

        Returns
        -------
        (B, 3, 3)
            Reciprocal unit cell vectors. [b1, b2, b3]
        """
        b1 = torch.cross(a2, a3, dim=-1)
        b2 = torch.cross(a3, a1, dim=-1)
        b3 = torch.cross(a1, a2, dim=-1)

        unit_volume = torch.sum(a1 * b1, dim=-1, keepdim=True)
        return torch.stack([b1, b2, b3], dim=1) * 2 * pi / unit_volume.unsqueeze(-1)
