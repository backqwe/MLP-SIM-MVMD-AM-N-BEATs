"""Setup configuration for MLP-SIM-MVMD-AM-N-BEATs package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlp-sim-mvmd-am-nbeats",
    version="0.1.0",
    author="backqwe",
    description=(
        "A deep learning framework combining MLP-SIM-MVMD and AM-N-BEATs "
        "for time series prediction"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/backqwe/MLP-SIM-MVMD-AM-N-BEATs",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
