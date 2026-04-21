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
        def _ensure_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor.')
            return value

        def _normalize_max(name: str, value: torch.Tensor) -> torch.Tensor:
            value = _ensure_tensor(name, value)
            if value.ndim == 0:
                value = value.unsqueeze(0)
            elif value.ndim != 1:
                raise ValueError(f'{name} must have shape (B,) or (1,). Got {tuple(value.shape)}.')
            return value

        def _normalize_range(name: str, value: torch.Tensor) -> torch.Tensor:
            value = _ensure_tensor(name, value)
            if value.ndim == 1:
                if value.shape[0] != 2:
                    raise ValueError(f'{name} must have shape (2,) or (B, 2). Got {tuple(value.shape)}.')
                value = value.unsqueeze(0)
            elif value.ndim == 2:
                if value.shape[1] != 2:
                    raise ValueError(f'{name} must have shape (B, 2). Got {tuple(value.shape)}.')
            else:
                raise ValueError(f'{name} must have shape (2,) or (B, 2). Got {tuple(value.shape)}.')
            return value

        if en <= 0:
            raise ValueError('en must be positive.')

        if q_xy_range is None and q_xy_max is None:
            raise ValueError('q_xy_range or q_xy_max must be provided.')
        if q_z_range is None and q_z_max is None:
            raise ValueError('q_z_range or q_z_max must be provided.')

        if q_xy_range is None:
            q_xy_max = _normalize_max('q_xy_max', q_xy_max)
            q_xy_range = torch.stack([torch.zeros_like(q_xy_max), q_xy_max], dim=1)
        else:
            q_xy_range = _normalize_range('q_xy_range', q_xy_range)

        B = q_xy_range.shape[0]
        if q_z_range is None:
            q_z_max = _normalize_max('q_z_max', q_z_max)
            if q_z_max.shape[0] != B:
                raise ValueError('q_xy_max and q_z_max should have the same batch size.')
            q_z_range = torch.stack([torch.zeros_like(q_z_max), q_z_max], dim=1)
        else:
            q_z_range = _normalize_range('q_z_range', q_z_range)

        if B != q_z_range.shape[0]:
            raise ValueError('q_xy_range and q_z_range should have the same batch size.')

        if q_xy_range.device != q_z_range.device:
            raise ValueError('q_xy_range and q_z_range must be on the same device.')

        if not (q_xy_range[:, 1] > q_xy_range[:, 0]).all():
            raise ValueError('q_xy_range should be in the form (min, max) with max > min.')
        if not (q_z_range[:, 1] > q_z_range[:, 0]).all():
            raise ValueError('q_z_range should be in the form (min, max) with max > min.')

        self.q_xy_range = q_xy_range
        self.q_z_range = q_z_range
        # q_xy_max = max(abs(self.q_xy_range[0]), abs(self.q_xy_range[1]))
        # q_z_max = max(abs(self.q_z_range[0]), abs(self.q_z_range[1]))
        # self.q_max = math.sqrt(q_xy_max ** 2 + q_z_max ** 2)
        # self.ai = ai
        self.wavelength = 12398 / en

        # self.database = Database(en=int(en), q_max=self.q_max) if create_FF else None
