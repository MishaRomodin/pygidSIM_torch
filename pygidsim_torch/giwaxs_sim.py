from typing import Optional, Union, Tuple
import torch
from torch import Tensor

from .crystal import Crystal
from .experiment import ExpParameters
from .q_sim import Qpos


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
    q_3d : Tensor
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

    # Clustering parameters
    CLUSTER_RADIUS_1D = 1e-2
    CLUSTER_RADIUS_2D = 2e-2

    def __init__(self,
                 crystal: Crystal,
                 exp: ExpParameters,
                 mi: Optional[Tensor] = None, ):
        self.crystal = crystal
        self.B = self.crystal.lat_par.shape[0]

        self.exp = exp
        if self.exp.q_xy_range.shape[0] == 1:
            self.exp.q_xy_range = self.exp.q_xy_range.expand(self.B, -1)
            self.exp.q_z_range = self.exp.q_z_range.expand(self.B, -1)
        assert self.exp.q_xy_range.shape[
                   0] == self.B, "q_xy_range must have the same batch size as the crystal lattice parameters."

        if mi is not None:
            self._mi = mi.to(
                device=self.crystal.lat_par.device,
                dtype=torch.float32
            )
        else:
            # TODO: calculate allowed miller indices
            raise NotImplementedError(
                "Calculation of allowed miller indices is not implemented yet. Please provide mi tensor."
            )
        self._q_sim = Qpos(self.crystal.lat_par, deg=self.crystal.deg)
        self.q_3d = self._q_sim.calculate_q3d(self.mi)

    @property
    def mi(self) -> Tensor:
        """Return Miller indices."""
        return self._mi

    @property
    def rec(self):
        return self._q_sim.rec

    def giwaxs_sim(self,
                   orientation: Union[Tensor, str, None] = Tensor([0, 0, 1]),
                   move_fromMW: bool = False, ):
        """
        Calculates peak positions and TODO: their intensities in the GIWAXS pattern.

        Parameters
        ----------
        orientation : Union[Tensor, str, None], optional
            Orientation of the crystal growth:
            - None: Powder diffraction (1D pattern).
            - 'random': Random orientation for each calculation (2D pattern).
            - Tensor: Specific orientation vector.
            Default is [001].
        move_fromMW : bool, optional
            True if move peaks from missing wedge to visible area, default = False.

        Returns
        -------
        q : Tensor
            Peak positions. Tensor of shape (B, peaks_num, 2).
        mask : Tensor
            Mask for peaks in the visible area. Tensor of shape (B, peaks_num).
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
            )
            return q_2d, mask

    @staticmethod
    def giwaxs_1d(q_1d: Tensor, ):
        """Calculate powder diffraction pattern for GIWAXS."""
        # TODO
        raise NotImplementedError("Powder diffraction is not implemented yet.")

    @staticmethod
    def giwaxs_2d(q_3d: Tensor,
                  q_xy_range: Tensor,
                  q_z_range: Tensor,
                  move_fromMW=False) -> Tuple[Tensor, Tensor]:
        """
        Convert q_3d to q_2d GIWAXS pattern, applying the limits for the visible area and moving peaks from missing
        wedge if needed.

        Parameters
        ----------
        q_3d : Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).
        q_xy_range : Tensor
            Range for the q in xy direction, Å^{-1}.
            Tensor of shape (B, 2).
        q_z_range : Tensor
            Range for the q in z direction, Å^{-1}.
            Tensor of shape (B, 2).
        move_fromMW : bool, optional
            Whether to move peaks from missing wedge to the visible area. Default is False.

        Returns
        -------
        q_2d : Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        q_mask : Tensor
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
        q_2d, q_mask = GIWAXS.cluster(q_2d, q_mask, r=GIWAXS.CLUSTER_RADIUS_2D)
        return q_2d, q_mask

    @staticmethod
    def q3d_q2d(q_3d: Tensor) -> Tensor:
        """
        Convert q_3d to q_2d

        Parameters
        ----------
        q_3d : Tensor
            Peak positions in 3d reciprocal space.
            Tensor of shape (B, num_reflections, 3).

        Returns
        -------
        q_2d : Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        """
        q_xy = torch.sqrt(q_3d[..., 0] ** 2 + q_3d[..., 1] ** 2)  # (B, peaks_num)
        q_z = q_3d[..., 2]  # (B, peaks_num)

        q_2d = torch.stack((q_xy, q_z)).permute(1, 2, 0).contiguous()  # shape (B, peaks_num, 2)
        q_2d[q_2d.abs() < 1e-4] = 0
        return q_2d

    @staticmethod
    def limit_q2d(q_2d: Tensor,
                  q_xy_range: Tensor,
                  q_z_range: Tensor,
                  use_abs: bool = False):
        """Calculate the mask for peaks in the visible area.

        Parameters
        ----------
        q_2d : Tensor
            Peak positions in 2d reciprocal space.
            Tensor of shape (B, num_reflections, 2).
        q_xy_range : Tensor
            Range for the q in xy direction, Å^{-1}.
            Tensor of shape (B, 2).
        q_z_range : Tensor
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
    def _move_from_MW(q_2d: Tensor,  # (B, peaks_num, 2)
                      wavelength: float = 12398 / 18000,  # wavelength, Angstrom,
                      ) -> Tensor:
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

    @staticmethod
    def cluster(
            q_sim: Tensor,
            mask: Tensor,
            r: float,
    ) -> Tuple[Tensor, Tensor]:
        """


        Parameters
        ----------
        q_sim : Tensor
            Peak positions in Q-space.
             Tensor of shape (B, N) for 1D or (B, N, 2) for 2D.
        mask : Tensor
            Mask for peaks in the visible area.
            Tensor of shape (B, num_reflections).
        r : float
            Clustering radius.
        """
        from torch_geometric.nn import radius_graph
        import cudf
        import cugraph

        dim = q_sim.ndim
        B, N = q_sim.shape[0], q_sim.shape[1]

        if dim == 2:
            # TODO
            raise NotImplementedError("Powder diffraction is not implemented yet.")
        elif dim == 3:
            q_valid = q_sim[mask]  # (num_valid_peaks, 2)
        else:
            raise ValueError(f"Wrong q_sim dimension: {dim}")

        device = q_valid.device
        batch = torch.arange(B, device=device)[:, None]
        batch = batch.expand(B, N)[mask]  # (num_valid_peaks, )

        if q_valid.shape[0] == 0:
            if dim == 2:
                # TODO
                raise NotImplementedError("Powder diffraction is not implemented yet.")
            elif dim == 3:
                q_pad = torch.full((B, 0, q_valid.shape[-1]), float("nan"), device=device)
            mask_out = torch.zeros((B, 0), dtype=torch.bool, device=device)
            return q_pad, mask_out

        edge_index = radius_graph(q_valid,
                                  r=r,
                                  loop=True,
                                  max_num_neighbors=256,
                                  batch=batch,
                                  )

        src, dst = edge_index

        df = cudf.DataFrame({
            "src": cudf.Series(src),
            "dst": cudf.Series(dst),
        })

        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(df, source="src", destination="dst")

        cc = cugraph.connected_components(G)

        vertices = torch.as_tensor(cc["vertex"].values, device=device)
        components = torch.as_tensor(cc["labels"].values, device=device)

        tmp = torch.empty_like(vertices)
        tmp[vertices] = components
        _, labels = torch.unique(tmp, return_inverse=True)

        counts_per_cluster = torch.bincount(labels).float()
        if dim == 2:
            # TODO
            raise NotImplementedError("Powder diffraction is not implemented yet.")
        elif dim == 3:
            sum_x = torch.bincount(labels, weights=q_valid[:, 0])  # (num_clusters)
            sum_y = torch.bincount(labels, weights=q_valid[:, 1])  # (num_clusters)
            q_fin = torch.stack((sum_x / counts_per_cluster,
                                 sum_y / counts_per_cluster,),
                                dim=1, )  # (num_clusters, 2)

        num_clusters = counts_per_cluster.shape[0]
        cluster_batch = torch.empty(num_clusters,
                                    dtype=batch.dtype,
                                    device=device)
        cluster_batch[labels] = batch  # (num_clusters)

        perm = torch.argsort(cluster_batch)
        q_fin = q_fin[perm]
        cluster_batch = cluster_batch[perm]

        counts = torch.bincount(cluster_batch, minlength=B)
        starts = torch.cumsum(counts, 0) - counts
        N_max = counts.max().item()

        idx = torch.arange(num_clusters, device=device)
        local_idx = idx - starts[cluster_batch]

        if dim == 2:
            # TODO
            q_pad = torch.full((B, N_max), float("nan"), device=device)
            raise NotImplementedError("Powder diffraction is not implemented yet.")
        elif dim == 3:
            q_pad = torch.full((B, N_max, 2), float("nan"), device=device)

        q_pad[cluster_batch, local_idx] = q_fin
        mask_out = ~torch.isnan(q_pad).any(dim=-1)

        return q_pad, mask_out
