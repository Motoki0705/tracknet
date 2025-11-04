## Why
The current repository lacks the necessary tooling and configuration to enforce the Code Style requirements specified in `openspec/project.md`. The project requires Ruff + Black for linting/formatting, Mypy for type checking, and proper CI/CD setup to ensure code quality standards are maintained.

## What Changes
- Add Ruff and Black configuration files with project-specific settings
- Configure Mypy for type checking with appropriate settings
- Update `pyproject.toml` to include development dependencies (ruff, black, mypy, pre-commit)
- Create GitHub Actions workflow for automated linting, formatting, type checking, and testing
- Add pre-commit hooks configuration for local development
- Ensure directory structure follows the specified `src/` convention if needed

## Impact
- **Affected specs**: Development Environment, Code Quality
- **Affected code**: All Python files in the repository will need to conform to new style guidelines
- **Development workflow**: All commits will be automatically checked for code style compliance
- **CI/CD**: GitHub Actions will enforce code quality standards on pull requests
