#!/usr/bin/env python
import os

from setuptools import find_packages, setup

import lightning_transformers as ltf
from lightning_transformers import setup_tools

_PATH_ROOT = os.path.dirname(__file__)
long_description = setup_tools._load_readme_description(_PATH_ROOT, homepage=ltf.__homepage__, ver=ltf.__version__)

setup(
    name="lightning-transformers",
    version=ltf.__version__,
    description=ltf.__docs__,
    author=ltf.__author__,
    author_email=ltf.__author_email__,
    url=ltf.__homepage__,
    download_url="https://github.com/PyTorchLightning/lightning-transformers",
    license=ltf.__license__,
    packages=find_packages(exclude=["tests", "docs"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=setup_tools._load_requirements(_PATH_ROOT),
    entry_points={
        "console_scripts": [
            "pl-transformers-train=lightning_transformers.cli.train:hydra_entry",
            "pl-transformers-predict=lightning_transformers.cli.predict:hydra_entry",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/lightning-transformers/issues",
        "Documentation": "https://lightning-transformers.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/PyTorchLightning/lightning-transformers",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        # 'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
