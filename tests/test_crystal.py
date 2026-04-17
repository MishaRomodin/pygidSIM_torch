"""
Tests for Crystal-based GIWAXS simulations.
"""
import pytest
import numpy as np
from pygidsim_torch.giwaxs_sim import GIWAXS


class TestCrystalGIWAXS:
    """Test class for Crystal-based GIWAXS functionality."""

    def test_crystal(self, crystal_single, crystal_multiple):
        """Test Crystal inizialization."""
        assert crystal_single.lattice_params.shape == (1, 6), \
            f"Lattice parameters should be (1, 6), got {crystal_single.lattice_params.shape}"
        assert crystal_multiple.lattice_params.shape == (3, 6), \
            f"Lattice parameters should be (3, 6), got {crystal_multiple.lattice_params.shape}."

    def test_crystal_giwaxs_single(self, crystal_single, exp_parameters, test_mi, random_orientation_single):
        """Test GIWAXS simulation with a single crystal."""
        el = GIWAXS(crystal=crystal_single, exp=exp_parameters, mi=test_mi)
        q_2d, q_mask = el.giwaxs_sim(orientation=random_orientation_single)

        assert q_2d.ndim == 3, f"q_2d should be a 3D tensor, but got {q_2d.ndim}D."
        assert q_mask.ndim == 2, f"q_mask should be a 2D tensor, but got {q_mask.ndim}D"
        assert q_2d.shape[0] == 1, f"q_2d should have batch size 1, but got {q_2d.shape[0]}"
        assert q_2d.shape[2] == 2, f"q_2d should have 2 components in the last dimension, but got {q_2d.shape[2]}"
        assert q_mask.shape == q_2d.shape[:2], \
            (f"q_mask should have the same batch size and number of peaks as q_2d,"
             f" but got {q_mask.shape} and {q_2d.shape[:2]}")

    def test_crystal_giwaxs_single_random_or(self, crystal_single, exp_parameters, test_mi):
        """Test GIWAXS simulation with a single crystal."""
        el = GIWAXS(crystal=crystal_single, exp=exp_parameters, mi=test_mi)
        q_2d, q_mask = el.giwaxs_sim(orientation='random')

        assert q_2d.ndim == 3, f"q_2d should be a 3D tensor, but got {q_2d.ndim}D."
        assert q_mask.ndim == 2, f"q_mask should be a 2D tensor, but got {q_mask.ndim}D"
        assert q_2d.shape[0] == 1, f"q_2d should have batch size 1, but got {q_2d.shape[0]}"
        assert q_2d.shape[2] == 2, f"q_2d should have 2 components in the last dimension, but got {q_2d.shape[2]}"
        assert q_mask.shape == q_2d.shape[:2], \
            (f"q_mask should have the same batch size and number of peaks as q_2d,"
             f" but got {q_mask.shape} and {q_2d.shape[:2]}")

    def test_crystal_giwaxs_multiple(self, crystal_multiple, exp_parameters, test_mi, random_orientation_multiple):
        """Test GIWAXS simulation with a several crystals."""
        B = crystal_multiple.lattice_params.shape[0]
        el = GIWAXS(crystal=crystal_multiple, exp=exp_parameters, mi=test_mi)
        assert el.exp.q_xy_range.shape[0] == B, f"q_xy_range should have {B} rows."
        assert el.exp.q_z_range.shape[0] == B, f"q_z_range should have {B} rows."
        q_2d, q_mask = el.giwaxs_sim(orientation=random_orientation_multiple)

        assert q_2d.ndim == 3, f"q_2d should be a 3D tensor, but got {q_2d.ndim}D."
        assert q_mask.ndim == 2, f"q_mask should be a 2D tensor, but got {q_mask.ndim}D"
        assert q_2d.shape[0] == B, f"q_2d should have batch size {B}, but got {q_2d.shape[0]}"
        assert q_2d.shape[2] == 2, f"q_2d should have 2 components in the last dimension, but got {q_2d.shape[2]}"
        assert q_mask.shape == q_2d.shape[:2], \
            (f"q_mask should have the same batch size and number of peaks as q_2d,"
             f" but got {q_mask.shape} and {q_2d.shape[:2]}")

    def test_crystal_giwaxs_multiple_random_or(self, crystal_multiple, exp_parameters, test_mi):
        """Test GIWAXS simulation with a several crystals."""
        B = crystal_multiple.lattice_params.shape[0]
        el = GIWAXS(crystal=crystal_multiple, exp=exp_parameters, mi=test_mi)
        assert el.exp.q_xy_range.shape[0] == B, f"q_xy_range should have {B} rows."
        assert el.exp.q_z_range.shape[0] == B, f"q_z_range should have {B} rows."
        q_2d, q_mask = el.giwaxs_sim(orientation='random')

        assert q_2d.ndim == 3, f"q_2d should be a 3D tensor, but got {q_2d.ndim}D."
        assert q_mask.ndim == 2, f"q_mask should be a 2D tensor, but got {q_mask.ndim}D"
        assert q_2d.shape[0] == B, f"q_2d should have batch size {B}, but got {q_2d.shape[0]}"
        assert q_2d.shape[2] == 2, f"q_2d should have 2 components in the last dimension, but got {q_2d.shape[2]}"
        assert q_mask.shape == q_2d.shape[:2], \
            (f"q_mask should have the same batch size and number of peaks as q_2d,"
             f" but got {q_mask.shape} and {q_2d.shape[:2]}")
