from typing import Optional
import torch


class ExpParameters:
    """
    A class to represent the experiment parameters.

    Attributes
    ----------
    q_xy_range : torch.Tensor, optional
        Range for the q in xy direction, Å^{-1}. Tensor of shape (B, 2) or (2,)
    q_z_range : torch.Tensor, optional
        Range for the q in z direction, Å^{-1}. Tensor of shape (B, 2) or (2,)
    q_xy_max : torch.Tensor, optional
        Upper limit for q in xy direction, Å^{-1}. Tensor of shape (B,) or (1,).
        Used if 'q_xy_range' is not provided.
    q_z_max : torch.Tensor, optional
        Upper limit for q in z direction, Å^{-1}. Tensor of shape (B,) or (1,).
        Used if 'q_z_range' is not provided.
    wavelength : float
        Beam wavelength, Å.
    """

    def __init__(self,
                 q_xy_range: Optional[torch.Tensor] = None,  # (B, 2) or (2,)
                 q_z_range: Optional[torch.Tensor] = None,  # (B, 2) or (2,)
                 q_xy_max: Optional[torch.Tensor] = None,  # (B,) or (1,)
                 q_z_max: Optional[torch.Tensor] = None,  # (B,) or (1,)
                 # ai: float = 0.3,  # Incidence angle, deg
                 en: float = 18000,  # Energy, eV
                 # create_FF: bool = True,  # If True create database with form-factors.
                 ):
        if q_xy_range is None:
            if q_xy_max is not None:
                if not isinstance(q_xy_max, torch.Tensor):
                    raise TypeError('q_xy_max must be a torch.Tensor.')
                B = q_xy_max.shape[0]
                if B == 1:
                    q_xy_range = torch.tensor([0.0, q_xy_max]).unsqueeze(0)
                else:
                    q_xy_range = torch.zeros((B, 2), dtype=q_xy_max.dtype)
                    q_xy_range[:, 1] = q_xy_max
            else:
                raise ValueError('q_xy_range or q_xy_max must be provided.')
        else:
            if q_xy_range.ndim == 1:
                q_xy_range = q_xy_range.unsqueeze(0)

        B = q_xy_range.shape[0]
        if q_z_range is None:
            if q_z_max is not None:
                if not isinstance(q_z_max, torch.Tensor):
                    raise TypeError('q_z_max must be a torch.Tensor.')
                assert B == q_z_max.shape[0], "q_xy_max and q_z_max should have the same batch size."
                if B == 1:
                    q_z_range = torch.tensor([0.0, q_z_max]).unsqueeze(0)
                else:
                    q_z_range = torch.zeros((B, 2), dtype=q_xy_max.dtype)
                    q_z_range[:, 1] = q_z_max
            else:
                raise ValueError('q_z_range or q_z_max must be provided.')
        else:
            if q_z_range.ndim == 1:
                q_z_range = q_z_range.unsqueeze(0)
        assert B == q_z_range.shape[0], "q_xy_range and q_z_range should have the same batch size."

        assert q_xy_range.shape[1] == q_z_range.shape[1] == 2, "q_xy_range and q_z_range should have shape (B, 2)."
        assert (q_xy_range[:, 1] > q_xy_range[
            :, 0]).all(), "q_xy_range should be in the form (min, max) with max > min."
        assert (q_z_range[:, 1] > q_z_range[:, 0]).all(), "q_z_range should be in the form (min, max) with max > min."

        self.q_xy_range = q_xy_range
        self.q_z_range = q_z_range
        # q_xy_max = max(abs(self.q_xy_range[0]), abs(self.q_xy_range[1]))
        # q_z_max = max(abs(self.q_z_range[0]), abs(self.q_z_range[1]))
        # self.q_max = math.sqrt(q_xy_max ** 2 + q_z_max ** 2)
        # self.ai = ai
        self.wavelength = 12398 / en

        # self.database = Database(en=int(en), q_max=self.q_max) if create_FF else None
