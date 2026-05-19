import pytest
import torch

from pygidsim_torch.experiment import ExpParameters


class TestExpPar:
    """Test class for ExpParameters functionality."""

    def test_max_values(self):
        """Test maximum values of ExpParameters."""
        exp_par = ExpParameters(
            q_xy_max=torch.tensor([5.0]),
            q_z_max=torch.tensor([3.0])
        )
        assert exp_par.q_xy_range.shape == (1, 2), f"Expected q_xy_range shape (1, 2), got {exp_par.q_xy_range.shape}"
        assert exp_par.q_z_range.shape == (1, 2), f"Expected q_z_range shape (1, 2), got {exp_par.q_z_range.shape}"
        assert torch.all(exp_par.q_xy_range[0] == torch.tensor([0.0, 5.0])), (f"Expected q_xy_range values [0.0, 5.0],"
                                                                              f" got {exp_par.q_xy_range[0]}")
        assert torch.all(exp_par.q_z_range[0] == torch.tensor([0.0, 3.0])), (f"Expected q_z_range values [0.0, 3.0],"
                                                                             f" got {exp_par.q_z_range[0]}")

    def test_max_batch(self):
        """Test maximum values with batch size > 1."""
        q_xy_max = torch.tensor([5.0, 6.0, 1.0])
        q_z_max = torch.tensor([3.0, 4.0, 2.0])
        exp_par = ExpParameters(
            q_xy_max=q_xy_max,
            q_z_max=q_z_max
        )
        assert exp_par.q_xy_range.shape == (3, 2), f"Expected q_xy_range shape (3, 2), got {exp_par.q_xy_range.shape}"
        assert exp_par.q_z_range.shape == (3, 2), f"Expected q_z_range shape (3, 2), got {exp_par.q_z_range.shape}"
        assert torch.all(
            exp_par.q_xy_range[:, 0] == 0
        ), f"Expected all minimum values in q_xy_range to be 0, got {exp_par.q_xy_range[:, 0]}"
        assert torch.all(
            exp_par.q_z_range[:, 0] == 0
        ), f"Expected all minimum values in q_z_range to be 0, got {exp_par.q_xy_range[:, 0]}"

        assert torch.all(
            exp_par.q_xy_range[:, 1] == torch.tensor([5.0, 6.0, 1.0])
        ), f"Expected all maximum values in q_xy_range to be {q_xy_max}, got {exp_par.q_xy_range[:, 1]}"
        assert torch.all(
            exp_par.q_z_range[:, 1] == torch.tensor([3.0, 4.0, 2.0])
        ), f"Expected all maximum values in q_z_range to be {q_z_max}, got {exp_par.q_z_range[:, 1]}"

    def test_range_values(self):
        """Test providing explicit range values."""
        q_xy_range = torch.tensor([1.0, 6.0])
        q_z_range = torch.tensor([0.5, 4.0])
        exp_par = ExpParameters(
            q_xy_range=q_xy_range,
            q_z_range=q_z_range
        )
        assert exp_par.q_xy_range.shape == (1, 2), f"Expected q_xy_range shape (1, 2), got {exp_par.q_xy_range.shape}"
        assert exp_par.q_z_range.shape == (1, 2), f"Expected q_z_range shape (1, 2), got {exp_par.q_z_range.shape}"
        assert torch.all(
            exp_par.q_xy_range[0] == torch.tensor([1.0, 6.0])
        ), f"Expected q_xy_range [1.0, 6.0], got {exp_par.q_xy_range[0]}"
        assert torch.all(
            exp_par.q_z_range[0] == torch.tensor([0.5, 4.0])
        ), f"Expected q_z_range [0.5, 4.0], got {exp_par.q_z_range[0]}"

    def test_range_batch(self):
        """Test providing explicit range values with batch size > 1."""
        q_xy_range = torch.tensor([[1.0, 5.0], [2.0, 6.0], [-2.0, 6.0]])
        q_z_range = torch.tensor([[0.5, 3.0], [-1.0, 4.0], [1.0, 4.0]])
        exp_par = ExpParameters(
            q_xy_range=q_xy_range,
            q_z_range=q_z_range
        )
        assert exp_par.q_xy_range.shape == (3, 2), f"Expected q_xy_range shape (3, 2), got {exp_par.q_xy_range.shape}"
        assert exp_par.q_z_range.shape == (3, 2), f"Expected q_z_range shape (3, 2), got {exp_par.q_z_range.shape}"
        assert torch.all(
            exp_par.q_xy_range[:, 0] == torch.tensor([1.0, 2.0, -2.0])
        ), f"Expected q_xy_range [[1.0, 5.0], [-2.0, 6.0], [2.0, 6.0]], got {exp_par.q_xy_range}"
        assert torch.all(
            exp_par.q_z_range[:, 0] == torch.tensor([0.5, -1.0, 1.0])
        ), f"Expected q_z_range [[0.5, 3.0], [-1.0, 4.0], [1.0, 4.0]], got {exp_par.q_z_range}"
        assert torch.all(
            exp_par.q_xy_range[:, 1] == torch.tensor([5.0, 6.0, 6.0])
        ), f"Expected q_xy_range [[1.0, 5.0], [2.0, 6.0], [6.0, 6.0]], got {exp_par.q_xy_range}"
        assert torch.all(
            exp_par.q_z_range[:, 1] == torch.tensor([3.0, 4.0, 4.0])
        ), f"Expected q_z_range [[0.5, 3.0], [-1.0, 4.0], [1.0, 4.0]], got {exp_par.q_z_range}"

    def test_mixed_range_and_max(self):
        """Test mixed initialization: q_xy_range takes precedence over q_xy_max."""
        q_xy_range = torch.tensor([1.0, 6.0])
        q_z_max = torch.tensor([3.0])
        # q_xy_range should be used, q_z_max should be used to create q_z_range
        exp_par = ExpParameters(
            q_xy_range=q_xy_range,
            q_z_max=q_z_max
        )
        assert exp_par.q_xy_range.shape == (1, 2)
        assert exp_par.q_z_range.shape == (1, 2)
        # q_xy_range should be unchanged
        assert torch.all(exp_par.q_xy_range[0] == torch.tensor([1.0, 6.0]))
        # q_z_range should be created from q_z_max
        assert torch.all(exp_par.q_z_range[0] == torch.tensor([0.0, 3.0]))

    def test_batch_size_mismatch_error(self):
        """Test that mismatched batch sizes raise ValueError."""
        q_xy_max = torch.tensor([5.0, 6.0])  # batch size 2
        q_z_max = torch.tensor([3.0])  # batch size 1
        with pytest.raises(ValueError):
            ExpParameters(q_xy_max=q_xy_max, q_z_max=q_z_max)

    def test_invalid_q_xy_max_type(self):
        """Test that non-tensor q_xy_max raises TypeError."""
        with pytest.raises(TypeError):
            ExpParameters(q_xy_max=5.0, q_z_max=torch.tensor([3.0]))

    def test_invalid_q_z_max_type(self):
        """Test that non-tensor q_z_max raises TypeError."""
        with pytest.raises(TypeError):
            ExpParameters(q_xy_max=torch.tensor([5.0]), q_z_max=3.0)

    def test_missing_q_xy_parameters_error(self):
        """Test that ValueError is raised when neither q_xy_range nor q_xy_max is provided."""
        with pytest.raises(ValueError):
            ExpParameters(q_z_max=torch.tensor([3.0]))

    def test_missing_q_z_parameters_error(self):
        """Test that ValueError is raised when neither q_z_range nor q_z_max is provided."""
        with pytest.raises(ValueError):
            ExpParameters(q_xy_max=torch.tensor([5.0]))

    def test_invalid_range_order(self):
        """Test that ranges where min >= max raise ValueError."""
        # min > max
        with pytest.raises(ValueError):
            ExpParameters(
                q_xy_range=torch.tensor([5.0, 3.0]),
                q_z_range=torch.tensor([0.5, 4.0])
            )

    def test_invalid_range_order_batch(self):
        """Test that invalid range order is caught in batch mode with ValueError."""
        with pytest.raises(ValueError):
            ExpParameters(
                q_xy_range=torch.tensor([[1.0, 5.0], [6.0, 2.0]]),  # second batch invalid
                q_z_range=torch.tensor([[0.5, 3.0], [1.0, 4.0]])
            )

    def test_1d_tensor_expansion(self):
        """Test that 1D tensors are correctly expanded to 2D."""
        q_xy_range_1d = torch.tensor([1.0, 6.0])
        q_z_range_1d = torch.tensor([0.5, 4.0])
        exp_par = ExpParameters(
            q_xy_range=q_xy_range_1d,
            q_z_range=q_z_range_1d
        )
        # Should be expanded to (1, 2)
        assert exp_par.q_xy_range.shape == (1, 2)
        assert exp_par.q_z_range.shape == (1, 2)
        assert exp_par.q_xy_range.ndim == 2
        assert exp_par.q_z_range.ndim == 2

    def test_large_batch(self):
        """Test with a larger batch size."""
        batch_size = 50
        q_xy_max = torch.linspace(1.0, 10.0, batch_size)
        q_z_max = torch.linspace(0.5, 5.0, batch_size)
        exp_par = ExpParameters(q_xy_max=q_xy_max, q_z_max=q_z_max)
        assert exp_par.q_xy_range.shape == (batch_size, 2)
        assert exp_par.q_z_range.shape == (batch_size, 2)
        # Check that all minimums are 0
        assert torch.all(exp_par.q_xy_range[:, 0] == 0)
        assert torch.all(exp_par.q_z_range[:, 0] == 0)
        # Check that maximums match input
        assert torch.allclose(exp_par.q_xy_range[:, 1], q_xy_max)
        assert torch.allclose(exp_par.q_z_range[:, 1], q_z_max)

    def test_batch_size_mismatch_q_xy_range_q_z_range(self):
        """Test that mismatched batch sizes between ranges raise ValueError."""
        q_xy_range = torch.tensor([[1.0, 5.0], [2.0, 6.0]])  # batch size 2
        q_z_range = torch.tensor([[0.5, 3.0]])  # batch size 1
        with pytest.raises(ValueError):
            ExpParameters(q_xy_range=q_xy_range, q_z_range=q_z_range)
