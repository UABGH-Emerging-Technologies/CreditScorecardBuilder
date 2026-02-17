from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs", "mkdocstrings"]

style_packages = ["black", "flake8", "isort"]

test_packages = [
    "pytest>=6.0",
    "pytest-cov>=2.10",
    "pytest-mock>=3.0",
    "pytest-timeout>=1.4",
    "pytest-xdist>=2.0",  # For parallel test execution
]

dev_packages = ["mlflow", "pip-tools", "pandas"] + test_packages

# Define our package
setup(
    name="CreditScore",
    version=0.1,
    description="A generic repo for building out credit score models quickly.",
    author="Kameshwari S.",
    author_email="ksoundararajan@uabmc.edu",
    url="",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + dev_packages,
        "docs": docs_packages,
        "test": test_packages,
        "style": style_packages,
    },
)
