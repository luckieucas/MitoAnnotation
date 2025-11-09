"""
Setup script for mitoem2 package.
"""
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r", encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mitoem2",
    version="0.1.0",
    author="MitoEM2 Team",
    description="A comprehensive toolkit for mitochondria segmentation in EM images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weilab/mitoem2",
    packages=find_packages(exclude=["tests", "scripts", "docs", "examples", "src"]),
    package_data={
        "mitoem2": [
            "configs/*.yaml",
            "configs/*/*.yaml",
            "configs/*/*/*.yaml",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    entry_points={
        "console_scripts": [
            "mitoem2-train=mitoem2.scripts.train:main",
            "mitoem2-evaluate=mitoem2.scripts.evaluate:main",
            "mitoem2-inference=mitoem2.scripts.inference:main",
        ],
    },
)
