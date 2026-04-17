import torch
from typing import Optional, Union, Tuple
from pygidsim_torch.experiment import ExpParameters
from pygidsim_torch.q_sim import Q_pos


class Crystal:
    """
    A class to represent the crystal structure.

    Parameters
    ----------
    lattice_params : torch.Tensor
        Lattice parameters of shape (B, 6). Columns: a, b, c, alpha, beta, gamma.
    device : torch.device
        Device on which the tensor is stored.
    """

    def __init__(self,
                 lattice_params: torch.Tensor,
                 # spgr: torch.Tensor[int],
                 # atoms: Optional[torch.Tensor] = None,
                 # atom_positions: Optional[torch.Tensor] = None,
                 # occ: Optional[torch.Tensor] = None,
                 # scale: Optional[torch.Tensor] = None, # (B, 3)
                 device: torch.device = None
                 ):
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lattice_params = lattice_params.to(self.device)
        if self.lattice_params.ndim == 1:
            self.lattice_params = self.lattice_params.unsqueeze(0)
        # self.spgr = spgr.to(device)


class GIWAXS:
    """
    A class to calculate the GIWAXS pattern from the crystal structure and experimental parameters

    Attributes
    ----------
    crystal : Crystal
        Crystal structure representation.
    exp : ExpParameters
        Experiment parameters representation.
    _mi : ArrayLike
        Allowed miller indices, optional. Shape (N, 3)
        If None - calculate via xrayutilities
    q_3d : torch.Tensor
        Peak positions in 3d reciprocal space. (with default orientation [001] for all samples])
        Tensor of shape (B, num_reflections, 3).

    Methods
    -------
    giwaxs_sim(orientation):
        Calculates the GIWAXS pattern.
    mi:
        Return allowed miller indices.
    rec:
        Return reciprocal vectors.
    """

    def __init__(self,
                 crystal: Crystal,
                 exp: ExpParameters,
                 mi: Optional[torch.Tensor] = None, ):
        self.crystal = crystal
        self.B = self.crystal.lattice_params.shape[0]

        self.exp = exp
        if self.exp.q_xy_range.shape[0] == 1:
            self.exp.q_xy_range = self.exp.q_xy_range.expand(self.B, -1)
            self.exp.q_z_range = self.exp.q_z_range.expand(self.B, -1)
        assert self.exp.q_xy_range.shape[
                   0] == self.B, "q_xy_range must have the same batch size as the crystal lattice parameters."

        self._q_sim = Q_pos(self.crystal.lattice_params)
        if mi is not None:
            self._mi = mi.to(self.crystal.lattice_params.device)
        else:
            # TODO: calculate allowed miller indices
            raise NotImplementedError(
                "Calculation of allowed miller indices is not implemented yet. Please provide mi tensor."
            )
        self.q_3d = self._q_sim.calculate_q3d(self.mi)

    @property
    def mi(self) -> torch.Tensor:
        """Return Miller indices."""
        return self._mi

    def giwaxs_sim(self,
                   orientation: Union[torch.Tensor, str, None] = torch.tensor([0, 0, 1]),
                   move_fromMW: bool = True, ):
        """
        Calculates peak positions and TODO: their intensities in the GIWAXS pattern.

        Parameters
        ----------
        orientation : Union[torch.Tensor, str, None], optional
            Orientation of the crystal growth:
            - None: Powder diffraction (1D pattern).
            - 'random': Random orientation for each calculation (2D pattern).
            - torch.Tensor: Specific orientation vector.
            Default is [001].
        move_fromMW : bool, optional
            True if move peaks from missing wedge to visible area, default = False.

        Returns
        -------
        q : torch.Tensor
            Peak positions. Tensor of shape (B, peaks_num, 2).
        """
        if orientation is None:
            q_1d, mask = self.giwaxs_1d(self.q_3d)
            return q_1d, mask
        else:
            q_3d_rot = self._q_sim.rotate_vect(self.q_3d, orientation)  # (B, peaks_num, 3)
            q_2d, mask = self.giwaxs_2d(
                q_3d=q_3d_rot,
                q_xy_range=self.exp.q_xy_range,
                q_z_range=self.exp.q_z_range,
                move_fromMW=move_fromMW
            )  # (B, peaks_num, 2)
            return q_2d, mask

    @staticmethod
    def giwaxs_1d(q_2d: torch.Tensor, ):
        # TODO
        raise NotImplementedError("Powder diffraction is not implemented yet.")

    @staticmethod
    def giwaxs_2d(q_3d: torch.Tensor,
                  q_xy_range: torch.Tensor,
                  q_z_range: torch.Tensor,
                  move_fromMW=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert q_3d to q_2d GIWAXS pattern, applying the limits for the visible area and moving peaks from missing
        wedge if needed.

        Parameters
        ----------
        q_3d : torch.Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).
        q_xy_range : torch.Tensor
            Range for the q in xy direction, Å^{-1}.
            Tensor of shape (B, 2).
        q_z_range : torch.Tensor
            Range for the q in z direction, Å^{-1}.
            Tensor of shape (B, 2).
        move_fromMW : bool, optional
            Whether to move peaks from missing wedge to the visible area. Default is False.

        Returns
        -------
        q_2d : torch.Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        q_mask : torch.Tensor
            Mask for peaks in the visible area.
            Tensor of shape (B, num_reflections).
        """
        if q_xy_range.any() < 0 or q_z_range.any() < 0:
            # TODO: implement negative ranges
            raise NotImplementedError("Negative ranges are not implemented yet.")
        q_2d = GIWAXS.q3d_q2d(q_3d)
        q_mask = GIWAXS.limit_q2d(
            q_2d=q_2d,
            q_xy_range=q_xy_range.to(q_2d.device),
            q_z_range=q_z_range.to(q_2d.device),
            use_abs=False
        )
        if move_fromMW:
            q_2d = GIWAXS._move_from_MW(q_2d)
        return q_2d, q_mask

    @staticmethod
    def q3d_q2d(q_3d: torch.Tensor) -> torch.Tensor:
        """
        Convert q_3d to q_2d

        Parameters
        ----------
        q_3d : torch.Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).

        Returns
        -------
        q_2d : torch.Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        """
        q_xy = torch.sqrt(q_3d[..., 0] ** 2 + q_3d[..., 1] ** 2)  # (B, peaks_num)
        q_z = q_3d[..., 2]  # (B, peaks_num)

        q_2d = torch.stack((q_xy, q_z)).permute(1, 2, 0).contiguous()  # shape (B, peaks_num, 2)
        q_2d[q_2d.abs() < 1e-4] = 0
        return q_2d

    @staticmethod
    def limit_q2d(q_2d: torch.Tensor,
                  q_xy_range: torch.Tensor,
                  q_z_range: torch.Tensor,
                  use_abs: bool = False):
        """Calculate the mask for peaks in the visible area.

        Parameters
        ----------
        q_2d : torch.Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        q_xy_range : torch.Tensor
            Range for the q in xy direction, Å^{-1}.
            Tensor of shape (B, 2).
        q_z_range : torch.Tensor
            Range for the q in z direction, Å^{-1}.
            Tensor of shape (B, 2).
        use_abs : bool, optional
            Whether to take the absolute value of q_2d before applying the range limits. Default is False.
        """
        assert q_2d.device == q_xy_range.device, "q_2d and q_xy_range must be on the same device."
        q_mask = (
                (q_2d[..., 1] >= q_z_range[:, 0].unsqueeze(1)) &
                (q_2d[..., 1] <= q_z_range[:, 1].unsqueeze(1))
        )
        if use_abs:
            q_xy_max = q_xy_range.abs().max(dim=1).values.unsqueeze(1)
            q_mask &= (q_2d[..., 0].abs() <= q_xy_max)
        else:
            q_mask &= ((q_2d[..., 0] >= q_xy_range[:, 0].unsqueeze(1)) &
                       (q_2d[..., 0] <= q_xy_range[:, 1].unsqueeze(1)))

        return q_mask

    @staticmethod
    def _move_from_MW(q_2d: torch.Tensor,  # (B, peaks_num, 2)
                      wavelength: float = 12398 / 18000,  # wavelength, Angstrom,
                      ) -> torch.Tensor:
        """Move peaks from Missimg Wedge to the visible area"""
        k = 2 * torch.pi / wavelength

        q_xy = q_2d[..., 0]  # (B, peaks_num)
        q_z = q_2d[..., 1]  # (B, peaks_num)

        # condition if peaks are in Missing Wedge
        condition_inMW = (k - torch.abs(q_xy)) ** 2 > (k ** 2 - q_z ** 2)  # (B, peaks_num)

        q_abs_mod = torch.sqrt(q_xy ** 2 + q_z ** 2)  # (B, peaks_num)

        new_q_xy = q_abs_mod ** 2 / (2 * k)
        new_q_z = q_abs_mod * torch.sqrt(4 * (k ** 2) - q_abs_mod ** 2) / (2 * k)

        q_2d[..., 0] = torch.where(condition_inMW, new_q_xy, q_xy)
        q_2d[..., 1] = torch.where(condition_inMW, new_q_z, q_z)

        return q_2d
