# Testing Guide

This document provides comprehensive guidelines for testing in the TrackNet project.

## Testing Strategy

TrackNet follows a three-layer testing strategy to ensure code quality and reliability:

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions and classes in isolation

**Scope**:
- Utility functions (`tracknet.utils.*`)
- Configuration building (`tracknet.utils.config`)
- Logging functionality (`tracknet.utils.logging`)
- Model components
- Dataset components
- Individual tool functions

**Guidelines**:
- Use mocking for external dependencies
- Test edge cases and error conditions
- Keep tests fast and focused
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

**Example**:
```python
def test_build_cfg_with_custom_parameters(self):
    """Test building configuration with custom parameters."""
    cfg = build_cfg(
        data_name="tracknet",
        model_name="convnext_fpn_heatmap", 
        training_name="default",
        seed=123,
        dry_run=True
    )
    
    # Verify custom seed was used
    assert cfg.runtime.seed == 123
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test multiple components working together

**Scope**:
- Configuration loading and validation
- Data loading pipelines
- Model training setup
- Component interactions
- End-to-end data flows

**Guidelines**:
- Test realistic scenarios
- Use actual configuration files when possible
- Test error handling across components
- Validate component interfaces
- Use test data that mimics real data structure

**Example**:
```python
def test_config_override_integration(self):
    """Test configuration override system integration."""
    overrides = ["training.batch_size=8", "model.backbone=resnet50"]
    cfg = build_cfg(overrides=overrides, dry_run=True)
    
    # Verify overrides were applied
    assert cfg.training.batch_size == 8
    assert cfg.model.backbone == "resnet50"
```

### 3. E2E Tests (`tests/e2e/`)

**Purpose**: Test complete workflows from start to finish

**Scope**:
- Complete training pipelines
- Model inference workflows
- Data processing pipelines
- Model saving/loading cycles
- Configuration to execution flows

**Guidelines**:
- Use minimal data for reproducibility
- Test critical user workflows
- Validate end-to-end functionality
- Use `@pytest.mark.slow` for long-running tests
- Focus on integration points and user scenarios

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_minimal_training_workflow(self, temp_dir):
    """Test minimal training workflow with mock components."""
    # Create minimal config and test complete workflow
    # ...
```

## Test Organization

### Directory Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_utils.py       # Utility function tests
│   ├── test_config.py      # Configuration tests
│   └── test_models.py      # Model component tests
├── integration/            # Integration tests
│   ├── test_config_pipeline.py
│   └── test_data_pipeline.py
├── e2e/                    # End-to-end tests
│   ├── test_training_pipeline.py
│   └── test_inference_pipeline.py
├── tools/                  # Tool-specific tests
│   └── test_annotation_pipeline.py
├── conftest.py            # Shared fixtures
└── utils.py               # Test utilities
```

### Test Naming Conventions

- **Files**: `test_*.py` or `*_test.py`
- **Classes**: `Test*`
- **Functions**: `test_*`
- **Descriptive names**: `test_function_name_with_specific_scenario`

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.e2e          # End-to-end tests
@pytest.mark.slow        # Long-running tests
```

## Writing Good Tests

### 1. Test Structure (AAA Pattern)

```python
def test_tensor_creation(self):
    """Test basic tensor creation utilities."""
    # Arrange - Set up test data and mocks
    expected_shape = (3, 224, 224)
    
    # Act - Execute the function being tested
    tensor = torch.randn(expected_shape)
    
    # Assert - Verify the results
    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float32
```

### 2. Use Fixtures Effectively

```python
@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "model": {"name": "test_model", "backbone": "resnet18"},
        "training": {"batch_size": 2, "learning_rate": 0.001}
    }

def test_with_config(self, mock_config):
    # Use the fixture
    assert mock_config["model"]["name"] == "test_model"
```

### 3. Mock External Dependencies

```python
@patch('tracknet.utils.config._seed_all')
def test_config_seeding_integration(self, mock_seed):
    """Test configuration and random seeding integration."""
    cfg = build_cfg(seed=42, dry_run=True)
    
    # Verify seeding was called
    mock_seed.assert_called_once_with(42)
    assert cfg.runtime.seed == 42
```

### 4. Test Error Cases

```python
def test_build_cfg_nonexistent_config(self):
    """Test building configuration with nonexistent config file."""
    with pytest.raises(FileNotFoundError):
        build_cfg(data_name="nonexistent", dry_run=True)
```

## Coverage Requirements

### Coverage Targets

- **Overall Coverage**: 75% minimum
- **Core Modules**:
  - `tracknet.utils`: 90% minimum
  - `tracknet.models`: 85% minimum
  - `tracknet.datasets`: 85% minimum

### Coverage Exclusions

The following patterns are excluded from coverage:

```python
# pragma: no cover
def __repr__(self):
    return f"MyClass({self.attr})"

# Debug and development code
if self.debug:
    pass

# Abstract methods and protocols
class MyProtocol(Protocol):
    @abstractmethod
    def my_method(self) -> None: ...

# Error handling that's hard to test
raise NotImplementedError
```

### Running Coverage

```bash
# Generate coverage report
uv run pytest --cov=tracknet --cov-report=html

# Check coverage against thresholds
uv run pytest --cov=tracknet --cov-fail-under=75

# View detailed coverage report
open htmlcov/index.html
```

## Test Data Management

### Temporary Data

Use fixtures with `temp_dir` for temporary test data:

```python
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
```

### Mock Data

Create realistic mock data that matches production data structure:

```python
def create_mock_tensor(width=640, height=480):
    """Create a mock tensor for testing."""
    return torch.randn(1, 3, height, width)

def create_test_image(path: Path, size=(100, 100)):
    """Create a test image file at the given path."""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)
    return path
```

### Test Data Location

- Small test data: Include in repository under `tests/data/`
- Large test data: Use external storage or download on demand
- Generated data: Create dynamically in tests

## Running Tests

### Local Development

```bash
# Run all tests
uv run pytest

# Run specific test layers
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/e2e/

# Run with coverage
uv run pytest --cov=tracknet --cov-report=html

# Run specific markers
uv run pytest -m unit
uv run pytest -m "not slow"

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

### CI/CD Integration

Tests run automatically on:
- Pull requests
- Pushes to main/develop branches
- Python 3.11 and 3.12 matrices

Coverage is uploaded to Codecov for tracking.

## Best Practices

### 1. Keep Tests Independent
- Don't rely on test execution order
- Clean up after each test
- Use fresh fixtures for each test

### 2. Make Tests Readable
- Use descriptive names
- Add docstrings explaining test purpose
- Group related tests in classes

### 3. Test the Right Things
- Focus on behavior, not implementation
- Test public interfaces
- Use property-based testing for complex logic

### 4. Handle Flaky Tests
- Use retries for network-dependent tests
- Mock external services
- Mark flaky tests appropriately

### 5. Performance Considerations
- Keep unit tests fast (< 1 second)
- Use `@pytest.mark.slow` for integration/E2E tests
- Parallelize test execution when possible

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test function
uv run pytest tests/unit/test_utils.py::TestConfigUtils::test_build_cfg_with_default_parameters -v

# Run with debugger
uv run pytest --pdb tests/unit/test_utils.py::test_function

# Show local variables on failure
uv run pytest -l tests/unit/test_utils.py
```

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Missing Fixtures**: Check `conftest.py` for fixture definitions
3. **Configuration Issues**: Use `dry_run=True` to avoid file system operations
4. **Mock Issues**: Verify patch paths and mock return values

## Contributing Tests

When adding new features:

1. **Add unit tests** for new functions/classes
2. **Add integration tests** for component interactions
3. **Update documentation** if testing patterns change
4. **Ensure coverage** meets minimum requirements
5. **Run full test suite** before submitting PR

### Test Review Checklist

- [ ] Tests cover happy path and error cases
- [ ] Test names are descriptive
- [ ] Proper fixtures are used
- [ ] External dependencies are mocked
- [ ] Coverage requirements are met
- [ ] Tests are fast and reliable
- [ ] Documentation is updated if needed
