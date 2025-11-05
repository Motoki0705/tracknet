# Contributing Guidelines

This document provides guidelines for contributing to the TrackNet project.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Git
- Basic knowledge of PyTorch and computer vision

### Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/your-username/tracknet.git
cd tracknet
```

2. **Install dependencies**:
```bash
uv sync --dev
source .venv/bin/activate
```

3. **Set up pre-commit hooks**:
```bash
uv run pre-commit install
```

4. **Verify setup**:
```bash
uv run pytest
uv run ruff check .
uv run mypy tracknet
```

## Development Workflow

### 1. Create a Branch

Use conventional commits for branch names:

```bash
git checkout -b feat/your-feature-name
git checkout -b fix/bug-description
git checkout -b docs/update-documentation
git checkout -b test/add-test-coverage
```

### 2. Make Changes

Follow these guidelines while developing:

#### Code Style

- **Python**: Follow PEP 8, enforced by Ruff
- **Naming**: 
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Documentation**: Google-style docstrings for all public functions/classes
- **Type hints**: Add type hints for new code

#### Code Quality Tools

```bash
# Lint and fix issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Type checking
uv run mypy tracknet

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### 3. Add Tests

**All new features must include tests**. Follow our [testing strategy](testing.md):

```bash
# Run tests before committing
uv run pytest

# Run with coverage
uv run pytest --cov=tracknet

# Check specific test layers
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/e2e/
```

#### Test Requirements

- **Unit tests**: For new functions/classes
- **Integration tests**: For component interactions
- **Coverage**: Maintain 75% overall, 85-90% for core modules
- **Documentation**: Add docstrings explaining test purpose

### 4. Update Documentation

- Update relevant documentation in `docs/`
- Add examples for new features
- Update README if needed
- Document configuration changes

### 5. Commit Changes

Use [conventional commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat(models): add new resnet backbone"
git commit -m "fix(utils): resolve config loading error"
git commit -m "docs(testing): add pytest examples"
git commit -m "test(datasets): improve coverage for data loader"
```

### 6. Create Pull Request

#### PR Requirements

- [ ] All tests pass
- [ ] Coverage meets minimum requirements
- [ ] Code is properly formatted
- [ ] Type checking passes (or has valid exceptions)
- [ ] Documentation is updated
- [ ] PR description explains changes clearly

#### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Coverage requirements met

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## Testing Guidelines

### When to Add Tests

1. **New Features**: Always add comprehensive tests
2. **Bug Fixes**: Add regression tests
3. **Refactoring**: Ensure existing tests still pass
4. **Configuration Changes**: Test new/modified options

### Test Structure

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange, Act, Assert
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        pass
    
    def test_integration(self):
        """Test integration with other components."""
        pass
```

### Coverage Targets

- **Overall**: 75% minimum
- **Core modules**: 85-90% minimum
- **New code**: 90%+ expected

Run coverage locally:

```bash
uv run pytest --cov=tracknet --cov-report=html --cov-fail-under=75
open htmlcov/index.html
```

## Code Review Process

### Reviewer Guidelines

1. **Functionality**: Does the code work as intended?
2. **Testing**: Are tests comprehensive and reliable?
3. **Style**: Does code follow project conventions?
4. **Documentation**: Is code properly documented?
5. **Performance**: Are there obvious performance issues?

### Author Guidelines

1. **Respond promptly** to review comments
2. **Explain reasoning** for complex changes
3. **Update tests** based on feedback
4. **Keep PR focused** on single feature/fix
5. **Address all feedback** before requesting merge

## Project Structure

```
tracknet/
├── tracknet/          # Main package
│   ├── datasets/      # Data loading and processing
│   ├── models/        # Model definitions
│   ├── training/      # Training logic
│   ├── tools/         # Utility tools
│   └── utils/         # Common utilities
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── e2e/           # End-to-end tests
├── configs/           # Configuration files
├── docs/              # Documentation
├── demo/              # Example scripts
└── .github/           # CI/CD workflows
```

### Adding New Components

1. **Choose appropriate directory** based on functionality
2. **Follow existing patterns** for imports and structure
3. **Add comprehensive tests** in corresponding test directory
4. **Update configuration** if new parameters are needed
5. **Document usage** in appropriate docs file

## Configuration Management

### Adding New Configuration Options

1. **Update YAML configs** in `configs/`
2. **Add validation** in relevant code
3. **Document options** in configuration docs
4. **Add tests** for new options
5. **Update schema** if using validation

### Configuration Best Practices

- Use sensible defaults
- Validate configuration on load
- Document all options
- Provide examples
- Test configuration loading

## Performance Guidelines

### Code Performance

1. **Profile code** before optimizing
2. **Use vectorized operations** (NumPy/PyTorch)
3. **Avoid unnecessary loops**
4. **Cache expensive operations**
5. **Consider memory usage**

### Testing Performance

1. **Keep unit tests fast** (< 1 second)
2. **Mark slow tests** with `@pytest.mark.slow`
3. **Use mocks** for expensive operations
4. **Parallelize test execution** when possible

## Security Guidelines

### General Security

1. **Validate inputs** from external sources
2. **Use secure defaults** for configurations
3. **Avoid hardcoding secrets**
4. **Sanitize file paths** and user input
5. **Review dependencies** regularly

### Data Security

1. **Don't commit sensitive data**
2. **Use environment variables** for secrets
3. **Validate data formats** before processing
4. **Handle errors gracefully** without exposing details

## Release Process

### Version Management

- Use semantic versioning (semver)
- Update version numbers in `pyproject.toml`
- Create release notes
- Tag releases in Git

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version is incremented
- [ ] Change log is updated
- [ ] Release notes are prepared
- [ ] Tag is created and pushed

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory
- **Examples**: Look at `demo/` scripts
- **Issues**: Search GitHub issues first
- **Discussions**: Use GitHub Discussions for questions

### Contact

- **Maintainers**: Tag specific maintainers in issues/PRs
- **Community**: Use GitHub Discussions for general questions
- **Bugs**: Create detailed bug reports with reproduction steps

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to TrackNet! 🎾
