# Contributing to Tomato Sample Classificator

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project follows standard open-source collaboration practices. We expect all contributors to:

- Be respectful and constructive in discussions
- Focus on what is best for the project and the research community
- Accept constructive criticism gracefully
- Show empathy towards other contributors

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone <your-fork-url>
   cd tomato_sample_classificator
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title** describing the problem
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, package versions)
- **Error messages** or logs if applicable

### Suggesting Enhancements

For feature requests or enhancements:

- Check if the feature has already been suggested
- Provide a clear use case
- Explain why this feature would be useful
- If possible, suggest an implementation approach

### Pull Requests

We welcome pull requests for:

- Bug fixes
- Documentation improvements
- New features (please discuss in an issue first for major changes)
- Code refactoring
- Performance improvements

## Development Setup

### Prerequisites

- Python 3.11 or higher (3.13 tested)
- pip or conda package manager

### Installation

```bash
# Clone your fork
git clone <your-fork-url>
cd tomato_sample_classificator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

## Code Style Guidelines

### Python Style

- **PEP 8 compliance**: Follow Python's style guide
- **Line length**: Maximum 100 characters (88 for Black formatter)
- **Imports**: Group in order: standard library, third-party, local
- **Naming conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Documentation

- **Docstrings**: Use Google-style docstrings for all public functions and classes
  ```python
  def function_name(arg1: type, arg2: type) -> return_type:
      \"\"\"
      Brief description.
      
      More detailed description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
      
      Returns:
          Description of return value
      
      Raises:
          ErrorType: When and why this error is raised
      \"\"\"
  ```

- **Type hints**: Use type hints for function signatures
- **Comments**: Use inline comments sparingly, prefer self-documenting code

### Code Organization

- **Modularity**: Keep functions focused and single-purpose
- **File structure**: Follow existing module organization in `src/`
- **Configuration**: Add configurable parameters to `configs/reproducible_run.yaml`

### Example Good Code

```python
from typing import Tuple
import pandas as pd
import numpy as np


def calculate_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    top_n: int = 10
) -> Tuple[pd.DataFrame, np.ndarray]:
    \"\"\"
    Calculate and rank feature importance for a trained model.
    
    Args:
        model: Trained scikit-learn pipeline
        X: Feature matrix
        top_n: Number of top features to return
    
    Returns:
        Tuple of (feature_names_df, importance_scores)
    \"\"\"
    # Extract feature names
    feature_names = X.columns.tolist()
    
    # Get importance scores from model
    importance = model.named_steps['classifier'].feature_importances_
    
    # Create result dataframe
    results = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return results, importance
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_<module_name>.py`
- Name test functions as `test_<functionality>()`
- Use fixtures for common test data
- Aim for high code coverage

Example test:

```python
from tomato_classifier.features import select_features_for_fold


def test_select_features_for_fold_returns_expected_count():
    # Use a small synthetic dataset to verify fold selection behavior.
    ...
```

## Submitting Changes

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** if needed
3. **Add/update tests** for new functionality
4. **Run formatters/linters** used by the project
5. **Update CHANGELOG** (if exists) with your changes

### Pull Request Process

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: descriptive message"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - **Clear title** summarizing the change
   - **Description** explaining what and why
   - **Reference** to related issues (e.g., "Fixes #123")
   - **Testing** details showing how you tested the changes

4. **Respond to feedback** from reviewers
5. **Update your PR** if requested

### Commit Message Format

```
<type>: <short summary>

<optional detailed description>

<optional footer: issue references>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
- `feat: add SHAP waterfall plots for individual predictions`
- `fix: correct feature scaling in preprocessing pipeline`
- `docs: update installation instructions for conda users`

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Contact the maintainers (see README for contact info)

Thank you for contributing to scientific research! 🔬✨
