from pathlib import Path

from setuptools import setup

README = Path(__file__).with_name("README.md")

setup(
    name="opti-hub",
    version="0.1.0",
    description="Registry and installer for SOTA optimizers for ML/LLM research",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    py_modules=["opti_hub"],
    python_requires=">=3.8",
    install_requires=[
        'tomli; python_version < "3.11"',
    ],
)
