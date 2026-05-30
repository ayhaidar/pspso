"""Compatibility shim for older editable-install workflows.

Project metadata lives in pyproject.toml. Prefer:

    uv sync --extra api --group dev

or, as a pip fallback:

    pip install -e ".[api]"
"""

from setuptools import setup


setup()
