# Environment Setup Verification & Fix Guide

**Date**: 2025-12-22

I had multiple issues with installing the dependencies in `requirements.txt` on my M1 Mac. I was getting errors with `cffi`, `zstandard`, and `pandas`.

I resolved these issues by switching to Python 3.10.19 and manually installing the dependencies.

## Changes Made

1.  **Switched Python Version**: Changed `pyenv local` from 3.13.5 (and 3.11.9) to **3.10.19**. Python 3.10 is the stability sweet spot for the libraries used in ARENA 3.0.
2.  **Cleaned `requirements.txt` into `requirements-self-paced.txt**: Reverted the 3.13-specific hacks. Added a `pandas>=1.5.0` pin to prevent `pip` from backtracking to ancient versions.
3.  **Manual Build Dependencies**: Explicitly installed `cython<3`, `numpy`, and `poetry` with `--no-build-isolation`.
4.  **Anchored Installation**: Manually installed `pandas` and `streamlit` first.

## Reproduction Steps (Commands)

If you need to recreate this environment, follow these exact steps in your terminal:

```bash
# 1. Set Python version (requires pyenv installed)
pyenv local 3.10.19

# 2. Recreate virtual environment to ensure a clean slate
rm -rf .venv
python -m venv .venv

# 3. Install build dependencies manually
# 'numpy' and 'cython<3' are needed for hdbscan build compilation
# 'poetry' is needed for some other dependencies
.venv/bin/pip install numpy "cython<3" poetry

# 4. Anchor critical core versions
# This prevents pip from backtracking to ancient versions of pandas/streamlit
.venv/bin/pip install "pandas>=1.5.0" streamlit

# 5. Install the rest without build isolation
# --no-build-isolation forces pip to use the numpy/cython we just installed
# rather than creating a temporary isolated build environment missing them
.venv/bin/pip install --no-build-isolation -r requirements-self-paced.txt
```

## Validation Results

### Installed Packages
A working environment is now established with the following key versions:

| Package | Version |
| :--- | :--- |
| **Python** | 3.10.19 |
| **pandas** | 2.3.3 |
| **numpy** | 2.2.6 |
| **streamlit** | 1.52.2 |
| **torch** | (Installed) |

### Verification
Run `pip list` to confirm the presence of core packages. The environment is now ready for `chapter0_fundamentals`.

> [!NOTE]
> If you encounter `ModuleNotFoundError` for visualization libraries (like `plotly` or `matplotlib`) in the future, simply run `pip install <package_name>` separately. The complex dependency graph made a single-shot install difficult, but the core is solid.
