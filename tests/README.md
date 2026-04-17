# Tests

This directory contains the test suite for pygidsim using pytest.

## Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_crystal.py` - Tests for Crystal-based GIWAXS simulations
- `test_exp_par.py` - Tests for Experimental parameters initialization

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cif.py

# Run specific test
pytest tests/test_crystal.py::TestCrystalGIWAXS::test_crystal_giwaxs_single
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=pygidsim_torch

# Generate HTML coverage report
pytest --cov=pygidsim_torch --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Parallel Testing

```bash
# Run tests in parallel (faster)
pytest -n auto
```

### Test Categories

```bash
# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Skip slow tests
pytest -m "not slow"
```

## Development Workflow

1. **Install development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

2. **Run tests before committing:**
   ```bash
   make test
   ```

3. **Check code quality:**
   ```bash
   make lint
   make format-check
   ```

4. **Format code:**
   ```bash
   make format
   ```
