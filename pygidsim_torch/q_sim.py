import torch
from math import pi
from typing import Union, Optional

from pygidsim_torch.utils import calculate_volume
from pygidsim_torch.directions import get_unique_directions


class Q_pos:
    """
    A class to calculate the q positions in 3d reciprocal space.

    Attributes
    ----------
    lat_par : torch.Tensor
        Lattice parameters. Shape (B, 6). Angles in grad.
    _rec : torch.Tensor
        Reciprocal vectors. Shape (B, 3, 3).

    Methods
    -------
    calculate_q3d(mi):
        Calculates the q vectors in 3d reciprocal space.
    rotate_vect(orientation, baz):
        Rotate the crystal.
    """

    def __init__(self, lat_par: torch.Tensor):
        self.lat_par = lat_par
        self._B = len(lat_par)
        self.device = lat_par.device
        self.dtype = lat_par.dtype

        self.volume = self._calc_volume(vol_min=10)
        self.valid = self.volume > 0
        self._rec = self._calculate_rec()

    def calculate_q3d(self, mi: torch.Tensor):
        """
        Calculate scattering vectors.

        Parameters
        ----------
        mi : torch.Tensor
            The Miller indices. Shape (num_reflections, 3).

        Returns
        -------
        torch.Tensor
            The scattering vectors. Shape (B, num_reflections, 3).
        """
        # if len(mi) == 0:
        #     return torch.tensor([], dtype=self.dtype).reshape(self._B, 0, 3)
        return torch.matmul(mi, self._rec)

    def rotate_vect(self,
                    q_3d: torch.Tensor,
                    orientation: Optional[Union[torch.Tensor, str]] = None, ):
        """
        Rotate crystal

        Parameters
        ----------
        q_3d : torch.Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).
        orientation : Union[torch.Tensor, str]
            Orientation of the crystal growth:
            - torch.Tensor: Specific orientation vector. Tensor of shape (B, 3) or (3,) in case of same orientation
            for all samples.
            - 'random': Random orientation for each calculation (2D pattern).
            Default is [001] for all samples.

        Returns
        -------
        q_3d : torch.Tensor
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
                raise ValueError("orientation is not correct - use ArrayLike with size (3,) or 'random'")
            directions = get_unique_directions(-10, 10, device=self.device, dtype=self.dtype, )
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
            orientation = torch.nn.functional.normalize(orientation, dim=1)

        assert orientation.shape == (self._B, 3), "finally, the orientation must have shape (B, 3)."

        R = self._rotation_matrix(orientation=orientation.to(self.device))
        return torch.matmul(q_3d, R)

    def _rotation_matrix(
            self,
            orientation: torch.Tensor = None,
            *,
            baz: torch.Tensor = None,
    ):
        """
        Rotate crystal

        Parameters
        ----------
        orientation : torch.Tensor, optional
            Crystallographic orientations.orientation of the crystal growth.
            Tensor of shape (B, 3) or (3,) in case of same orientation for all samples.
            Default is [001] for all samples.
        baz : torch.Tensor, optional
            Shape (3,) or (B, 3). Basis vector for the default orientation.
            Tensor of shape (B, 3) or (3,) in case of same orientation for all samples.
            Default is [001] for all samples.

        Return
        -------
        R : torch.Tensor
            Rotation matrix. Shape (B, 3, 3).
        """
        # B = self.valid.sum().item()
        R = torch.full((self._B, 3, 3), float('nan'), device=self.device, dtype=self.dtype)

        baz = self._filter_orientations(baz)  # (B, 3)
        orientation = self._filter_orientations(orientation)  # (B, 3)

        same_mask = (orientation == baz).all(dim=-1)  # (B,)

        # if orientation == baz → identity
        if same_mask.all():
            R[self.valid] = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
            return R

        orient = torch.matmul(orientation.unsqueeze(1), self._rec[self.valid]).squeeze(1)  # (B, 3)

        v1 = orient / orient.norm(dim=-1, keepdim=True)
        v2 = baz / baz.norm(dim=-1, keepdim=True)

        n_raw = torch.cross(v1, v2, dim=-1)  # (B, 3)
        zero_mask = (n_raw.norm(dim=-1) < 1e-8)
        n_raw[zero_mask] = baz[zero_mask]

        n = n_raw / n_raw.norm(dim=-1, keepdim=True)

        cos_phi = torch.sum(v1 * v2, dim=-1)  # (B,)
        sin_phi = torch.sqrt(torch.clamp(1 - cos_phi ** 2, min=0))  # (B,)

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

        R[self.valid] = torch.stack([a1, a2, a3], dim=1)  # (B, 3, 3)

        # if orientation == baz → identity
        if same_mask.any():
            I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
            # Create a full-size mask combining same_mask (for valid entries) with self.valid
            full_same_mask = torch.zeros(self._B, dtype=torch.bool, device=self.device)
            full_same_mask[self.valid] = same_mask
            R[full_same_mask] = I

        return R

    def _filter_orientations(self, orient: torch.Tensor):
        """Filter orientations to only valid entries if full batch is provided."""
        if orient is None:
            orient = torch.tensor([0., 0., 1.], dtype=self.dtype, device=self.device)
        if orient.ndim == 1:
            orient = orient.unsqueeze(0).expand(self._B, -1)
        assert orient.shape[0] == self._B, "Orientation tensor must have the same batch size as the lattice parameters."
        orient = torch.nn.functional.normalize(orient, dim=1)
        return orient[self.valid]

    @property
    def rec(self) -> torch.Tensor:
        """Return reciprocal lattice vectors."""
        return self._rec

    def _calculate_rec(self) -> torch.Tensor:
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
        a, b, c = self.lat_par[self.valid, :3].unbind(dim=-1)
        alpha, beta, gamma = (self.lat_par[self.valid, 3:] * pi / 180).unbind(dim=-1)

        a1 = torch.full((self._B, 3), float('nan'), device=self.device, dtype=self.dtype)
        a1[self.valid] = torch.stack(
            [
                a,
                torch.zeros_like(a),
                torch.zeros_like(a)
            ], dim=-1
        )

        a2 = torch.full((self._B, 3), float('nan'), device=self.device, dtype=self.dtype)
        a2[self.valid] = torch.stack(
            [
                b * torch.cos(gamma),
                b * torch.sin(gamma),
                torch.zeros_like(b)
            ], dim=-1
        )

        a31 = c * torch.cos(beta)
        a32 = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        a33 = torch.sqrt(c ** 2 - a31 ** 2 - a32 ** 2)

        a3 = torch.full((self._B, 3), float('nan'), device=self.device, dtype=self.dtype)
        a3[self.valid] = torch.stack([a31, a32, a33], dim=-1)

        return a1, a2, a3

    def _calc_volume(self, vol_min=10):
        """Calculate the volume of the unit cell."""
        unit_volume = calculate_volume(self.lat_par, deg=True)
        unit_volume[unit_volume < vol_min] = 0
        return unit_volume

    @staticmethod
    def _calc_reciprocal_vectors(
            a1: torch.Tensor,  # (B, 3)
            a2: torch.Tensor,  # (B, 3)
            a3: torch.Tensor,  # (B, 3)
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

# if __name__ == "__main__":
#     lattice_params =
#     q_pos = Q_pos(self.crystal.lattice_params)
