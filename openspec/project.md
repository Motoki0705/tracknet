# Project Context

## Purpose

The goal is to train and run inference with models that detect various components (players, ball, court, etc.) from sequences of tennis images.
We aim to provide high-quality data and a stable training pipeline to streamline model development, validation, and operations.

Main objectives:

* Develop high-accuracy object detection models
* Build a highly reproducible environment for training, evaluation, and inference
* Make the inference pipeline lightweight and fast

---

## Tech Stack

* **Language**: Python
* **Frameworks**: PyTorch, PyTorch Lightning
* **Configuration Management**: OmegaConf
* **Logging/Visualization**: TensorBoard, tqdm
* **Model Tooling**: Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning), bitsandbytes (quantization)
* **CI/CD**: GitHub Actions

---

## Project Conventions

### Code Style

* **Linter/Formatter**: Ruff
* **Type Checking**: Mypy
* **Naming Rules**:

  * Variables/Functions: `snake_case`
  * Classes: `PascalCase`
  * Config keys: `lower_case_with_underscores`
* **Structure Rules**:

  * Under `tracknet/`: `datasets/`, `models/`, `training/`, `scripts/`, `configs/`, `tools/`, `utils/`
  * Clear separation of concerns by module (data processing, models, training, evaluation)

---

### Architecture Patterns

* **Pattern**: Module layout inspired by Clean Architecture

  * `domain`: Data types and model definitions
  * `application`: Training and inference flows
  * `infrastructure`: Data loading, persistence, logging
* **Characteristics**:

  * Explicit configuration management with OmegaConf
  * Training logic separated via `LightningModule`
  * Unified logging and metrics in TensorBoard

---

### Testing Strategy

* **Test Layers**:

  * **Unit**: Verify function/class logic (pytest)
  * **Integration/Contract**: Training/inference pipeline, config loading, model save/load, etc.
  * **E2E**: End-to-end from data → training → evaluation → inference (use a small dataset to confirm reproducibility)
* **Coverage Targets**:

  * Overall: 75–85%
  * Core logic: 90% (emphasis on branch coverage)
* **Policy**:

  * Maintain overall coverage while focusing especially on inference and evaluation logic
  * Always include at least one negative case (e.g., misconfigured settings, missing data, insufficient GPU resources)
* **Tools**:

  * Pytest + Coverage (`uv run pytest` ensures the shared virtual environment and `pyproject.toml` settings are respected).
  * Automatic execution in CI (GitHub Actions)
  * Default options (from `pyproject.toml`): `--cov=tracknet`, HTML/XML coverage reports, `--strict-markers`, `--disable-warnings`, coverage gate ≥75%.

  ```bash
  # Common local test runs
  uv run pytest                          # full suite with defaults
  uv run pytest -m "unit"               # unit-only
  uv run pytest -m "integration and not slow"
  uv run pytest -m "e2e" --maxfail=1
  ```

---

### Git Workflow

* **Branching Model**: Trunk-based (short-lived branches)

  * Naming: `feat/`, `fix/`, `refactor/`, `test/`
* **Commit Convention**: Conventional Commits

  * Example: `feat(model): add detection head`
* **Review Process**:

  * `main` branch accepts changes via PRs only
  * CI runs `pytest`, lint, and format checks automatically
* **CI/CD**: Automated test runs and artifact retention via GitHub Actions

---

## External Dependencies

* **Primary Libraries/APIs**:

  * PyTorch, Lightning, Hugging Face Transformers
  * bitsandbytes (quantization), PEFT (fine-tuning)
* **CI Service**: GitHub Actions
* **External Logging/Monitoring**: TensorBoard (local or remote logger)
* **Package Management & Execution Policy**:

  * Dependency resolution and execution are standardized with **Astral `uv`**.
  * Always launch **`python` / `pip` / `pytest` / scripts** via **`uv run`** to eliminate per-tool virtual-environment drift and ensure full reproducibility.
  * Examples:

    ```bash
    # Run scripts (training, evaluation, inference, etc.)
    uv run python tracknet.training.train

    # Interactive execution (including one-liners)
    uv run python -c "import torch; print(torch.__version__)"

    # Run tests
    uv run pytest -q

    # Lint/Format
    uv run ruff check .
    uv run ruff format .
    ```
