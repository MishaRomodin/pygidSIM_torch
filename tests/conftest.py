import torch

from pygidsim_torch.experiment import ExpParameters
from pygidsim_torch.giwaxs_sim import Crystal
from pygidsim_torch.directions import get_mi
import pytest


@pytest.fixture
def exp_parameters():
    """Return a standard ExpParameters instance for testing."""
    return ExpParameters(
        q_xy_max=torch.tensor([5.0]),
        q_z_max=torch.tensor([3.0]),
        en=18_000,
    )


@pytest.fixture
def exp_parameters_small():
    """Return a tiny ExpParameters instance for testing."""
    return ExpParameters(
        q_xy_max=torch.tensor([0.2]),
        q_z_max=torch.tensor([0.3]),
        en=18_000,
    )


@pytest.fixture
def crystal_single():
    """Return a Crystal instance for testing."""
    lat_par = torch.tensor([6.3026, 6.3026, 6.3026, 90., 90., 90.], dtype=torch.float32)
    return Crystal(lat_par)


@pytest.fixture
def crystal_multiple():
    """Return a Crystal instance for testing."""
    lat_par = torch.tensor(
        [[6.3026, 6.3026, 6.3026, 90., 90., 90.],
         [3.6541, 10.3503, 11.8438, 89.9950, 90., 90.],
         [7.6111, 7.6105, 9.7373, 90.0325, 89.9363, 119.5385]], dtype=torch.float32
    )
    return Crystal(lat_par)


@pytest.fixture
def random_orientation_single():
    """Return a random orientation vector for testing."""
    return torch.tensor([5., 7., 1.])


@pytest.fixture
def random_orientation_multiple():
    """Return a set of random orientation vectors for testing."""
    return torch.tensor(
        [[5., 7., 1.],
         [3., 4., 2.],
         [6., 8., 3.]]
    )


@pytest.fixture
def test_mi():
    """Return a MI set for testing."""
    return get_mi(min_index=-3, max_index=2, dtype=torch.float32)
