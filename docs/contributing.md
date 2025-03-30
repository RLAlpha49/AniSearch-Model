# Contributing Guide

Thank you for your interest in contributing to AniSearch Model! This guide will help you get started with contributing to the project.

## Setup for Development

1. Fork the repository on GitHub
2. Clone your fork locally:

    ```bash
    git clone https://github.com/YOUR-USERNAME/AniSearch-Model.git
    cd AniSearch-Model
    ```

3. Set up a development environment:

    ```bash
    # Create a virtual environment
    python -m venv venv
    
    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Install development dependencies
    pip install pytest mypy ruff
    ```

## Code Style

This project follows these coding standards:

- **PEP 8** for Python code style
- **Type Annotations** for all function definitions
- **Docstrings** in Google style format
- **Line Length** limited to 100 characters

We use the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pytest** for unit testing

You can run these tools locally:

```bash
# Run linters
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest
```

## Pull Request Process

1. **Create a branch** for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Write tests** for your changes to ensure they work correctly

4. **Run the checks** to ensure your code passes all tests:

   ```bash
   pytest
   mypy src/
   ruff check src/
   ```

5. **Commit your changes** with a clear and descriptive commit message:

   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** from your fork to the main repository

## Adding New Features

When adding new features:

1. **Document your code** with docstrings and type annotations
2. **Update documentation** if necessary
3. **Add tests** for your feature
4. Consider **backwards compatibility**

## Reporting Bugs

When reporting bugs:

1. Check if the bug has already been reported
2. Include detailed steps to reproduce
3. Describe the expected behavior
4. Include relevant logs or screenshots
5. Specify your environment (OS, Python version, etc.)

## Feature Requests

For feature requests:

1. Clearly describe the feature and its benefits
2. Explain how it aligns with the project's goals
3. If possible, outline how it might be implemented
