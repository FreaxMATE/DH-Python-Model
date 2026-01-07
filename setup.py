"""
Setup configuration for DH-model Python package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dhmodel",
    version="1.0.0",
    author="Annemarie Eckes-Shephard",
    description="Deleuze et Houllier 1998 model with some modifications (Python version)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dhmodel-py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    include_package_data=True,
    package_data={
        "dhmodel": ["../data/*.pkl", "../data/*.csv"],
    },
)
